from __future__ import annotations

import os
import tempfile
import threading
import webbrowser
from email.utils import parseaddr
from io import BytesIO
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file

from ai_reply import ReplySuggester, build_email_context
from auth import GmailAuth
from reader import GmailReader
from sender import GmailSender
from utils import (
    extract_plain_text_from_payload,
    extract_text_from_attachment_bytes,
    get_header,
    list_attachments_from_payload,
)

BASE_DIR = Path(__file__).resolve().parent
PORT = int(os.getenv("PORT", "5000"))
HOST = os.getenv("HOST", "127.0.0.1")

# Keep index.html in the same folder as main.py. No templates folder is required.
app = Flask(__name__, template_folder=str(BASE_DIR))

_auth: GmailAuth | None = None
_reader: GmailReader | None = None
_sender: GmailSender | None = None
_suggester: ReplySuggester | None = None


def _get_auth() -> GmailAuth:
    global _auth
    if _auth is None:
        _auth = GmailAuth()
    return _auth


def _get_reader() -> GmailReader:
    global _reader
    if _reader is None:
        _reader = GmailReader(_get_auth())
    return _reader


def _get_sender() -> GmailSender:
    global _sender
    if _sender is None:
        _sender = GmailSender(_get_auth())
    return _sender


def _get_suggester() -> ReplySuggester:
    global _suggester
    if _suggester is None:
        _suggester = ReplySuggester()
    return _suggester


def _parse_contact(header_value: str) -> tuple[str, str]:
    name, email = parseaddr(header_value or "")
    return (name.strip(), email.strip())


def _build_reply_inputs(msg: dict, forced_to: str | None = None) -> tuple[str, dict]:
    """
    Build the header-rich email_text plus structured identity fields
    for the reply suggester.
    """
    payload = msg.get("payload", {}) or {}
    headers = payload.get("headers", []) or []

    from_header = get_header(headers, "From") or ""
    to_header = get_header(headers, "To") or ""
    cc_header = get_header(headers, "Cc") or ""
    reply_to_header = get_header(headers, "Reply-To") or ""
    subject = get_header(headers, "Subject") or ""
    body_text = extract_plain_text_from_payload(payload) or msg.get("snippet", "") or ""

    email_text = build_email_context(
        body_text=body_text,
        from_header=from_header,
        to_header=to_header,
        subject=subject,
        cc_header=cc_header,
        reply_to_header=reply_to_header,
    )

    original_sender_name, original_sender_email = _parse_contact(from_header)
    original_receiver_name, original_receiver_email = _parse_contact(to_header)
    reply_target_name, reply_target_email = _parse_contact(reply_to_header or from_header)

    if forced_to:
        reply_target_email = forced_to.strip()
        if not reply_target_name:
            reply_target_name = original_sender_name

    reply_kwargs = {
        "replier_name": original_receiver_name,
        "replier_email": original_receiver_email,
        "original_sender_name": original_sender_name,
        "original_sender_email": original_sender_email,
        "original_receiver_name": original_receiver_name,
        "original_receiver_email": original_receiver_email,
        "reply_target_name": reply_target_name,
        "reply_target_email": reply_target_email,
        "use_agent": True,
    }
    return email_text, reply_kwargs


def _message_card_payload(msg: dict, forced_to: str | None = None) -> dict:
    payload = msg.get("payload", {}) or {}
    headers = payload.get("headers", []) or []
    reply_target = forced_to or get_header(headers, "Reply-To") or get_header(headers, "From") or ""

    return {
        "id": msg.get("id", ""),
        "threadId": msg.get("threadId", ""),
        "from": get_header(headers, "From") or "",
        "to": get_header(headers, "To") or "",
        "reply_to": get_header(headers, "Reply-To") or "",
        "subject": get_header(headers, "Subject") or "",
        "reply_target": reply_target,
    }


def _get_message_or_404(message_id: str) -> dict:
    message = _get_reader().get_message_by_id(message_id)
    if not message:
        raise ValueError("Message not found.")
    return message


def _json_error(message: str, status: int = 400):
    return jsonify({"ok": False, "error": message}), status


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/messages")
def api_messages():
    try:
        payload = request.get_json(silent=True) or {}
        n = int(payload.get("n", 5) or 5)
        n = max(1, min(n, 25))
        mark_as_read = bool(payload.get("mark_as_read", False))
        email_address = (payload.get("email_address") or "").strip()

        reader = _get_reader()
        if email_address:
            messages = reader.fetch_last_n_by_email_data(email_address=email_address, n=n, mark_as_read=mark_as_read)
        else:
            messages = reader.fetch_last_n_data(n=n, mark_as_read=mark_as_read)

        return jsonify(
            {
                "ok": True,
                "messages": messages,
                "forced_to": email_address or None,
            }
        )
    except Exception as exc:
        return _json_error(str(exc), 500)


@app.get("/api/attachment")
def api_attachment():
    """Preview or download one attachment from a fetched email."""
    try:
        message_id = (request.args.get("message_id") or "").strip()
        attachment_id = (request.args.get("attachment_id") or "").strip()
        filename = (request.args.get("filename") or "").strip()
        part_id = (request.args.get("part_id") or "").strip()
        index_value = (request.args.get("index") or "").strip()
        download = (request.args.get("download") or "0").strip() == "1"

        if not message_id:
            return _json_error("message_id is required.")

        attachment_index = None
        if index_value != "":
            try:
                attachment_index = int(index_value)
            except ValueError:
                attachment_index = None

        attachment = _get_reader().get_attachment_file(
            message_id=message_id,
            attachment_id=attachment_id,
            filename=filename,
            part_id=part_id,
            index=attachment_index,
        )
        if not attachment:
            return _json_error("Attachment not found.", 404)

        data = attachment.get("data") or b""
        if not isinstance(data, (bytes, bytearray)):
            return _json_error("Attachment data could not be decoded.", 500)

        return send_file(
            BytesIO(data),
            mimetype=str(attachment.get("mime_type") or "application/octet-stream"),
            as_attachment=download,
            download_name=str(attachment.get("filename") or "attachment"),
            max_age=0,
        )
    except Exception as exc:
        return _json_error(f"Attachment failed: {exc}", 500)


@app.post("/api/summarize-mail")
def api_summarize_mail():
    """Summarize the selected email message body/headers."""
    try:
        payload = request.get_json(silent=True) or {}
        message_id = (payload.get("message_id") or "").strip()
        if not message_id:
            return _json_error("message_id is required.")

        msg = _get_message_or_404(message_id)
        email_text, _reply_kwargs = _build_reply_inputs(msg=msg)
        headers = (msg.get("payload", {}) or {}).get("headers", []) or []
        subject = get_header(headers, "Subject") or "Email"

        summary = _get_suggester().summarize_mail(email_text=email_text, subject=subject)
        return jsonify({"ok": True, "summary": summary})
    except Exception as exc:
        return _json_error(f"Could not summarize email: {exc}", 500)


@app.post("/api/summarize-attachments")
def api_summarize_attachments():
    """Summarize each attachment separately when text can be extracted."""
    try:
        payload = request.get_json(silent=True) or {}
        message_id = (payload.get("message_id") or "").strip()
        if not message_id:
            return _json_error("message_id is required.")

        msg = _get_message_or_404(message_id)
        message_payload = msg.get("payload", {}) or {}
        attachments = list_attachments_from_payload(message_payload)
        if not attachments:
            return jsonify({"ok": True, "summaries": [], "message": "This email has no attachments."})

        summaries = []
        reader = _get_reader()
        suggester = _get_suggester()

        for attachment_meta in attachments:
            attachment = reader.get_attachment_file(
                message_id=message_id,
                attachment_id=attachment_meta.get("attachmentId", ""),
                filename=attachment_meta.get("filename", ""),
                part_id=attachment_meta.get("partId", ""),
                index=attachment_meta.get("index"),
            )

            filename = attachment_meta.get("filename") or "attachment"
            mime_type = attachment_meta.get("mimeType") or "application/octet-stream"
            size = attachment_meta.get("size") or 0

            if not attachment:
                summaries.append(
                    {
                        "filename": filename,
                        "mimeType": mime_type,
                        "size": size,
                        "summary": "Attachment could not be loaded from Gmail.",
                        "extractable": False,
                        "extraction_note": "Attachment could not be loaded from Gmail.",
                    }
                )
                continue

            data = attachment.get("data") or b""
            if not isinstance(data, (bytes, bytearray)):
                data = b""

            extracted_text, extraction_note = extract_text_from_attachment_bytes(
                bytes(data),
                filename=str(attachment.get("filename") or filename),
                mime_type=str(attachment.get("mime_type") or mime_type),
            )

            summary = suggester.summarize_attachment(
                filename=str(attachment.get("filename") or filename),
                mime_type=str(attachment.get("mime_type") or mime_type),
                text=extracted_text,
                extraction_note=extraction_note,
            )

            summaries.append(
                {
                    "filename": str(attachment.get("filename") or filename),
                    "mimeType": str(attachment.get("mime_type") or mime_type),
                    "size": int(attachment.get("size") or size or 0),
                    "summary": summary,
                    "extractable": bool(extracted_text.strip()),
                    "extraction_note": extraction_note,
                }
            )

        return jsonify({"ok": True, "summaries": summaries})
    except Exception as exc:
        return _json_error(f"Could not summarize attachments: {exc}", 500)


@app.post("/api/reply-suggestions")
def api_reply_suggestions():
    try:
        payload = request.get_json(silent=True) or {}
        message_id = (payload.get("message_id") or "").strip()
        forced_to = (payload.get("forced_to") or "").strip() or None
        if not message_id:
            return _json_error("message_id is required.")

        msg = _get_message_or_404(message_id)
        email_text, reply_kwargs = _build_reply_inputs(msg=msg, forced_to=forced_to)

        suggester = _get_suggester()
        suggestion_1, suggestion_2 = suggester.suggest_two(email_text=email_text, **reply_kwargs)
        message_meta = _message_card_payload(msg=msg, forced_to=forced_to)

        return jsonify(
            {
                "ok": True,
                "message": message_meta,
                "suggestions": {
                    "formal": suggestion_1,
                    "warm": suggestion_2,
                },
            }
        )
    except Exception as exc:
        return _json_error(f"Could not generate AI suggestions: {exc}", 500)


@app.post("/api/reply")
def api_reply():
    try:
        payload = request.get_json(silent=True) or {}
        message_id = (payload.get("message_id") or "").strip()
        reply_text = (payload.get("reply_text") or "").strip()
        forced_to = (payload.get("forced_to") or "").strip() or None

        if not message_id:
            return _json_error("message_id is required.")
        if not reply_text:
            return _json_error("reply_text cannot be empty.")

        sender = _get_sender()
        if forced_to:
            sent_id = sender.reply_to_address(
                original_message_id=message_id,
                to_address=forced_to,
                reply_text=reply_text,
            )
        else:
            sent_id = sender.reply(original_message_id=message_id, reply_text=reply_text)

        if not sent_id:
            return _json_error("Reply could not be sent.", 500)

        return jsonify({"ok": True, "message_id": sent_id})
    except Exception as exc:
        return _json_error(f"Reply failed: {exc}", 500)


@app.post("/api/send-email")
def api_send_email():
    temp_paths: list[str] = []
    try:
        to_value = (request.form.get("to") or "").strip()
        subject = (request.form.get("subject") or "").strip()
        body = (request.form.get("body") or "").strip()

        if not to_value:
            return _json_error("To is required.")

        uploaded_files = request.files.getlist("attachments")
        for uploaded in uploaded_files:
            if not uploaded or not uploaded.filename:
                continue

            suffix = Path(uploaded.filename).suffix
            handle = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            uploaded.save(handle.name)
            handle.close()
            temp_paths.append(handle.name)

        sent_id = _get_sender().send(
            to=to_value,
            subject=subject,
            body=body,
            attachments=temp_paths,
        )
        if not sent_id:
            return _json_error("Email could not be sent.", 500)

        return jsonify({"ok": True, "message_id": sent_id})
    except Exception as exc:
        return _json_error(f"Send failed: {exc}", 500)
    finally:
        for path in temp_paths:
            try:
                os.remove(path)
            except OSError:
                pass


@app.get("/api/health")
def api_health():
    return jsonify({"ok": True})


def _open_browser() -> None:
    url = f"http://{HOST}:{PORT}"
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()


def main() -> None:
    _open_browser()
    app.run(host=HOST, port=PORT, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
