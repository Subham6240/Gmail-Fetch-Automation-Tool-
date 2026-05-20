from __future__ import annotations

import os
from typing import Dict, List, Optional

from googleapiclient.errors import HttpError

from auth import GmailAuth
from utils import (
    decode_body_data_to_bytes,
    extract_plain_text_from_payload,
    find_attachment_part,
    get_header,
    list_attachments_from_payload,
    sanitize_filename,
)


# ---------------- LangChain mail categorizer ----------------
class MailCategorizer:
    """
    Categorize emails into one of: work, personal, spam, urgent

    Uses LangChain + OpenAI if available and OPENAI_API_KEY is set.
    Falls back to a simple keyword heuristic if LangChain/OpenAI isn't available.
    """

    ALLOWED = {"work", "personal", "spam", "urgent"}

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.0):
        self.model = model
        self.temperature = temperature
        self._chain = None

        # Lazy import so the rest of the tool still works even if langchain isn't installed
        try:
            from langchain_openai import ChatOpenAI
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.output_parsers import StrOutputParser

            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You are an email classifier. "
                        "Return EXACTLY one label from: work, personal, spam, urgent. "
                        "No extra words, punctuation, or explanation.",
                    ),
                    (
                        "human",
                        "Classify this email.\n\n"
                        "From: {from_addr}\n"
                        "To: {to_addr}\n"
                        "Subject: {subject}\n"
                        "Body:\n{body}\n",
                    ),
                ]
            )

            llm = ChatOpenAI(model=self.model, temperature=self.temperature)
            self._chain = prompt | llm | StrOutputParser()

        except Exception:
            self._chain = None

    def classify(self, from_addr: str, to_addr: str, subject: str, body: str) -> str:
        # If LangChain is ready and API key exists, use it
        if self._chain is not None and os.getenv("OPENAI_API_KEY"):
            try:
                # Keep token usage sane: truncate body if huge
                body_in = body if len(body) <= 5000 else body[:5000] + "\n...(truncated)..."
                out = self._chain.invoke(
                    {
                        "from_addr": from_addr or "",
                        "to_addr": to_addr or "",
                        "subject": subject or "",
                        "body": body_in or "",
                    }
                )
                label = (out or "").strip().lower()
                if label in self.ALLOWED:
                    return label
            except Exception:
                pass  # fall through to heuristic

        # Heuristic fallback (no extra deps)
        return self._heuristic(from_addr, subject, body)

    def _heuristic(self, from_addr: str, subject: str, body: str) -> str:
        text = f"{from_addr}\n{subject}\n{body}".lower()

        urgent_kw = ["urgent", "asap", "immediately", "deadline", "overdue", "action required"]
        spam_kw = ["unsubscribe", "winner", "prize", "lottery", "free", "buy now", "limited offer", "click here"]
        work_kw = ["meeting", "invoice", "project", "deadline", "contract", "interview", "hr", "client", "report"]

        if any(k in text for k in urgent_kw):
            return "urgent"
        if any(k in text for k in spam_kw):
            return "spam"
        if any(k in text for k in work_kw):
            return "work"
        return "personal"


class GmailReader:
    """
    Reads emails:
      - fetch_last_n: last n recent inbox mails (with optional mark-as-read)
      - fetch_last_n_by_email: last n mails filtered by an email address
      - get_attachment_file: fetches attachment bytes for preview/download
    Enforced display fields:
      - From, To, Subject, Body, category, and attachment metadata.
    """

    def __init__(self, auth: GmailAuth):
        self.service = auth.get_service()
        self.categorizer = MailCategorizer()

    def _fetch_full_messages(self, ids: List[str]) -> List[Dict]:
        full_msgs: List[Dict] = []
        for mid in ids:
            try:
                m = self.service.users().messages().get(userId="me", id=mid, format="full").execute()
                m["id"] = mid  # keep at top level
                full_msgs.append(m)
            except HttpError as e:
                print(f"⚠️ Could not fetch {mid}: {e}")
        return full_msgs

    def get_message_by_id(self, message_id: str) -> Optional[Dict]:
        try:
            message = self.service.users().messages().get(userId="me", id=message_id, format="full").execute()
            message["id"] = message_id
            return message
        except HttpError as e:
            print(f"⚠️ Could not fetch {message_id}: {e}")
            return None

    def get_attachment_file(
        self,
        message_id: str,
        *,
        attachment_id: str = "",
        filename: str = "",
        part_id: str = "",
        index: Optional[int] = None,
    ) -> Optional[Dict[str, object]]:
        """
        Return attachment bytes and metadata for a Gmail message attachment.

        The frontend passes message_id + attachmentId/filename/index. This method
        re-reads the message, finds the correct MIME part, then fetches bytes from
        users.messages.attachments.get() when Gmail stored the data separately.
        """
        message = self.get_message_by_id(message_id)
        if not message:
            return None

        payload = message.get("payload", {}) or {}
        part = find_attachment_part(
            payload,
            attachment_id=attachment_id,
            filename=filename,
            part_id=part_id,
            index=index,
        )
        if not part:
            return None

        body = part.get("body", {}) or {}
        resolved_attachment_id = body.get("attachmentId") or attachment_id
        data = ""

        if resolved_attachment_id:
            attachment = (
                self.service.users()
                .messages()
                .attachments()
                .get(userId="me", messageId=message_id, id=resolved_attachment_id)
                .execute()
            )
            data = attachment.get("data") or ""
        else:
            data = body.get("data") or ""

        raw_bytes = decode_body_data_to_bytes(data)
        resolved_filename = sanitize_filename(part.get("filename") or filename or "attachment")
        mime_type = part.get("mimeType") or "application/octet-stream"

        return {
            "filename": resolved_filename,
            "mime_type": mime_type,
            "data": raw_bytes,
            "size": len(raw_bytes),
        }

    def _serialize_message(self, msg: Dict, index: Optional[int] = None) -> Dict:
        payload = msg.get("payload", {}) or {}
        headers = payload.get("headers", []) or []

        from_addr = get_header(headers, "From") or ""
        to_addr = get_header(headers, "To") or ""
        subject = get_header(headers, "Subject") or ""
        reply_to = get_header(headers, "Reply-To") or ""
        cc = get_header(headers, "Cc") or ""
        body_text = extract_plain_text_from_payload(payload) or msg.get("snippet", "") or ""
        attachments = list_attachments_from_payload(payload)

        category = self.categorizer.classify(
            from_addr=from_addr,
            to_addr=to_addr,
            subject=subject,
            body=body_text,
        )

        return {
            "index": index,
            "id": msg.get("id", ""),
            "threadId": msg.get("threadId", ""),
            "from": from_addr,
            "to": to_addr,
            "reply_to": reply_to,
            "cc": cc,
            "subject": subject,
            "category": category,
            "body": body_text.strip(),
            "snippet": msg.get("snippet", "") or "",
            "labelIds": msg.get("labelIds", []) or [],
            "attachments": attachments,
            "attachmentCount": len(attachments),
        }

    def _print_minimal_message(self, index: int, msg: Dict):
        item = self._serialize_message(msg=msg, index=index)

        # ---- Only these fields (plus index + category + attachment names) ----
        print(f"\n[{index}]")
        print(f"From: {item['from']}")
        print(f"To: {item['to']}")
        print(f"Subject: {item['subject']}")
        print(f"Category: {item['category']}")
        if item["attachments"]:
            print("Attachments:")
            for attachment in item["attachments"]:
                print(f"- {attachment['filename']} ({attachment['mimeType']}, {attachment['size']} bytes)")
        print("Body:")
        print(item["body"] if item["body"] else "(no plain-text body found)")
        print("-" * 60)

    def _list_messages(self, query: str, n: int = 5, mark_as_read: bool = False) -> List[Dict]:
        try:
            listed = self.service.users().messages().list(userId="me", q=query, maxResults=n).execute()
            msgs_meta = listed.get("messages", []) or []
            if not msgs_meta:
                return []

            ids = [m["id"] for m in msgs_meta]
            full_msgs = self._fetch_full_messages(ids)

            if mark_as_read:
                for msg in full_msgs:
                    if "UNREAD" in msg.get("labelIds", []):
                        try:
                            self.service.users().messages().modify(
                                userId="me",
                                id=msg["id"],
                                body={"removeLabelIds": ["UNREAD"], "addLabelIds": []},
                            ).execute()
                            msg["labelIds"] = [lab for lab in (msg.get("labelIds", []) or []) if lab != "UNREAD"]
                        except HttpError as e:
                            print(f"⚠️ Could not mark as read for {msg['id']}: {e}")

            return full_msgs
        except HttpError as e:
            print(f"❌ Read error: {e}")
            return []

    def fetch_last_n_data(self, n: int = 5, mark_as_read: bool = False) -> List[Dict]:
        messages = self._list_messages(query="in:inbox", n=n, mark_as_read=mark_as_read)
        return [self._serialize_message(msg, index=i) for i, msg in enumerate(messages, start=1)]

    def fetch_last_n_by_email_data(self, email_address: str, n: int = 5, mark_as_read: bool = False) -> List[Dict]:
        if not email_address:
            return []
        query = f'(from:{email_address}) OR (to:{email_address})'
        messages = self._list_messages(query=query, n=n, mark_as_read=mark_as_read)
        return [self._serialize_message(msg, index=i) for i, msg in enumerate(messages, start=1)]

    # -------- Feature 1: Fetch last n inbox mails --------
    def fetch_last_n(self, n: int = 5, mark_as_read: bool = False) -> List[Dict]:
        full_msgs = self._list_messages(query="in:inbox", n=n, mark_as_read=mark_as_read)
        if not full_msgs:
            print("📭 No messages found.")
            return []

        for i, msg in enumerate(full_msgs, start=1):
            self._print_minimal_message(i, msg)

        return full_msgs

    # -------- Feature 2: Fetch last n mails filtered by an email address --------
    def fetch_last_n_by_email(self, email_address: str, n: int = 5, mark_as_read: bool = False) -> List[Dict]:
        if not email_address:
            print("Please provide an email address.")
            return []

        full_msgs = self._list_messages(
            query=f'(from:{email_address}) OR (to:{email_address})',
            n=n,
            mark_as_read=mark_as_read,
        )
        if not full_msgs:
            print("📭 No messages found for that address.")
            return []

        for i, msg in enumerate(full_msgs, start=1):
            self._print_minimal_message(i, msg)

        return full_msgs
