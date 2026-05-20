from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass
from email.utils import parseaddr
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Optional LangChain imports.
# The file stays importable even before dependencies are installed.
# ---------------------------------------------------------------------------
try:
    from langchain_classic.agents import AgentExecutor
    from langchain_classic.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
    from langchain_classic.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.tools import tool
    from langchain_openai import ChatOpenAI
except ImportError:  # pragma: no cover - graceful import fallback
    AgentExecutor = None
    ChatPromptTemplate = None
    ChatOpenAI = None
    MessagesPlaceholder = None
    OpenAIToolsAgentOutputParser = None
    format_to_openai_tool_messages = None

    def tool(*args, **kwargs):  # type: ignore[misc]
        """Fallback no-op decorator so this module can still be imported."""

        def decorator(func):
            func.name = kwargs.get("name") or func.__name__
            return func

        if args and callable(args[0]):
            return decorator(args[0])
        return decorator


# ---------------------------------------------------------------------------
# Structured email/contact helpers
# ---------------------------------------------------------------------------
HEADER_CANDIDATES = {
    "from": ["From", "Sender"],
    "to": ["To", "Delivered-To", "X-Original-To"],
    "reply_to": ["Reply-To"],
    "cc": ["Cc"],
    "subject": ["Subject"],
}

TONE_GUIDES: Dict[str, Dict[str, object]] = {
    "formal": {
        "tone": "formal",
        "greeting_style": "Use a respectful greeting if a name is available; otherwise keep it neutral.",
        "body_style": [
            "Sound polished, courteous, and professional.",
            "Use complete sentences and clear structure.",
            "Acknowledge the sender's request or update before responding.",
            "Avoid slang, emojis, and overly casual phrasing.",
        ],
        "signoff_examples": ["Best regards", "Kind regards", "Sincerely"],
    },
    "warm": {
        "tone": "warm",
        "greeting_style": "Use a friendly but still professional greeting.",
        "body_style": [
            "Sound approachable, appreciative, and supportive.",
            "Keep the wording human and natural.",
            "Show positive intent without becoming overly casual.",
            "Keep it concise and easy to read.",
        ],
        "signoff_examples": ["Best", "Warm regards", "Thanks again"],
    },
    "informal": {
        "tone": "informal",
        "greeting_style": "Use a casual greeting when the email context allows it.",
        "body_style": [
            "Sound relaxed and conversational.",
            "Keep sentences short and natural.",
            "Do not become rude or careless.",
            "Still answer the sender clearly.",
        ],
        "signoff_examples": ["Best", "Thanks", "Cheers"],
    },
    "concise": {
        "tone": "concise",
        "greeting_style": "Use a minimal greeting or skip it if that fits the context.",
        "body_style": [
            "Keep the reply brief and direct.",
            "Prioritize the answer, action, or confirmation.",
            "Avoid filler and repetition.",
            "Stay polite even when short.",
        ],
        "signoff_examples": ["Best", "Thanks", "Regards"],
    },
    "apologetic": {
        "tone": "apologetic",
        "greeting_style": "Use a respectful greeting and acknowledge the inconvenience quickly.",
        "body_style": [
            "Sound accountable and calm.",
            "Apologize clearly without overexplaining.",
            "State the next step or resolution if available.",
            "Avoid defensive language.",
        ],
        "signoff_examples": ["Best regards", "Sincerely", "Thank you for your patience"],
    },
    "assertive": {
        "tone": "assertive",
        "greeting_style": "Use a professional greeting and get to the point.",
        "body_style": [
            "Be clear, direct, and respectful.",
            "State boundaries, requests, or expectations explicitly.",
            "Do not sound hostile or aggressive.",
            "Keep the wording firm and solution-oriented.",
        ],
        "signoff_examples": ["Regards", "Best regards", "Thank you"],
    },
    "follow_up": {
        "tone": "follow_up",
        "greeting_style": "Use a polite greeting and reference the earlier conversation naturally.",
        "body_style": [
            "Remind the reader of the pending item without sounding accusatory.",
            "State the follow-up ask or status request clearly.",
            "Keep it short and easy to act on.",
            "Stay courteous throughout.",
        ],
        "signoff_examples": ["Best", "Regards", "Thanks"],
    },
    "appreciative": {
        "tone": "appreciative",
        "greeting_style": "Use a warm greeting and acknowledge the sender positively.",
        "body_style": [
            "Express thanks sincerely.",
            "Mention what you appreciate when possible.",
            "Keep the reply grounded and specific.",
            "Avoid exaggerated praise.",
        ],
        "signoff_examples": ["Thanks again", "Warm regards", "Best"],
    },
}

TONE_ALIASES = {
    "professional": "formal",
    "friendly": "warm",
    "casual": "informal",
    "brief": "concise",
    "short": "concise",
    "sorry": "apologetic",
    "firm": "assertive",
    "follow-up": "follow_up",
    "follow up": "follow_up",
    "grateful": "appreciative",
}


@dataclass
class ContactInfo:
    name: str = ""
    email: str = ""


@dataclass
class EmailMetadata:
    sender_name: str = ""
    sender_email: str = ""
    receiver_name: str = ""
    receiver_email: str = ""
    reply_to_name: str = ""
    reply_to_email: str = ""
    cc: str = ""
    subject: str = ""
    body_preview: str = ""


@dataclass
class ResolvedReplyContext:
    replier_name: str = ""
    replier_email: str = ""
    original_sender_name: str = ""
    original_sender_email: str = ""
    original_receiver_name: str = ""
    original_receiver_email: str = ""
    reply_target_name: str = ""
    reply_target_email: str = ""
    subject: str = ""
    cc: str = ""
    body_preview: str = ""


def _require_langchain() -> None:
    if not all(
        [
            AgentExecutor,
            ChatOpenAI,
            ChatPromptTemplate,
            MessagesPlaceholder,
            OpenAIToolsAgentOutputParser,
            format_to_openai_tool_messages,
        ]
    ):
        raise ImportError(
            "LangChain/OpenAI dependencies are missing. "
            "Install the packages from requirements.txt before using the email reply agent."
        )



def _normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "")).strip()



def _extract_header_value(email_text: str, header_names: Sequence[str]) -> str:
    """Extract a header from raw email text, including folded header lines."""
    for header in header_names:
        pattern = rf"(?im)^{re.escape(header)}:\s*(.+(?:\n[ \t].+)*)"
        match = re.search(pattern, email_text or "")
        if match:
            raw = match.group(1)
            unfolded = re.sub(r"\n[ \t]+", " ", raw)
            return unfolded.strip()
    return ""



def _extract_body(email_text: str) -> str:
    text = (email_text or "").strip()
    if not text:
        return ""
    parts = re.split(r"\n\s*\n", text, maxsplit=1)
    body = parts[1] if len(parts) == 2 else parts[0]
    return body.strip()



def _parse_contact(raw_value: str, default_name: str = "", default_email: str = "") -> ContactInfo:
    name, email = parseaddr(raw_value or "")
    name = _normalize_whitespace(name) or _normalize_whitespace(default_name)
    email = (email or default_email or "").strip()

    if not email and raw_value:
        email_match = re.search(r"[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}", raw_value, re.I)
        if email_match:
            email = email_match.group(0)

    if not name and email:
        local = email.split("@", 1)[0].replace(".", " ").replace("_", " ").replace("-", " ")
        inferred = _normalize_whitespace(local)
        if inferred:
            name = inferred.title()

    return ContactInfo(name=name, email=email)



def extract_email_metadata_structured(
    email_text: str,
    default_sender_email: str = "",
    default_receiver_email: str = "",
    default_sender_name: str = "",
    default_receiver_name: str = "",
) -> EmailMetadata:
    from_header = _extract_header_value(email_text, HEADER_CANDIDATES["from"])
    to_header = _extract_header_value(email_text, HEADER_CANDIDATES["to"])
    reply_to_header = _extract_header_value(email_text, HEADER_CANDIDATES["reply_to"])
    cc_header = _extract_header_value(email_text, HEADER_CANDIDATES["cc"])
    subject = _extract_header_value(email_text, HEADER_CANDIDATES["subject"])

    sender = _parse_contact(from_header, default_sender_name, default_sender_email)
    receiver = _parse_contact(to_header, default_receiver_name, default_receiver_email)
    reply_to = _parse_contact(reply_to_header, default_email=sender.email)

    body_preview = _extract_body(email_text)[:700]

    return EmailMetadata(
        sender_name=sender.name,
        sender_email=sender.email,
        receiver_name=receiver.name,
        receiver_email=receiver.email,
        reply_to_name=reply_to.name,
        reply_to_email=reply_to.email,
        cc=cc_header.strip(),
        subject=subject.strip(),
        body_preview=body_preview,
    )



def build_email_context(
    body_text: str,
    from_header: str = "",
    to_header: str = "",
    subject: str = "",
    cc_header: str = "",
    reply_to_header: str = "",
) -> str:
    """Create a header-rich string that grounding tools can parse reliably."""
    lines: List[str] = []
    if from_header:
        lines.append(f"From: {from_header}")
    if to_header:
        lines.append(f"To: {to_header}")
    if cc_header:
        lines.append(f"Cc: {cc_header}")
    if reply_to_header:
        lines.append(f"Reply-To: {reply_to_header}")
    if subject:
        lines.append(f"Subject: {subject}")
    lines.append("")
    lines.append((body_text or "").strip())
    return "\n".join(lines).strip()



def _canonical_tone(tone: str) -> str:
    tone_in = _normalize_whitespace(tone).lower()
    if tone_in in TONE_GUIDES:
        return tone_in
    if tone_in in TONE_ALIASES:
        return TONE_ALIASES[tone_in]
    return "formal"



def _tone_tool_name(tone: str) -> str:
    canonical = _canonical_tone(tone)
    return {
        "formal": "get_formal_tone_rules",
        "warm": "get_warm_tone_rules",
        "informal": "get_informal_tone_rules",
        "concise": "get_concise_tone_rules",
        "apologetic": "get_apologetic_tone_rules",
        "assertive": "get_assertive_tone_rules",
        "follow_up": "get_follow_up_tone_rules",
        "appreciative": "get_appreciative_tone_rules",
    }[canonical]



def _tone_guide_text(tone: str) -> str:
    guide = TONE_GUIDES[_canonical_tone(tone)]
    bullets = "\n".join(f"- {item}" for item in guide["body_style"])
    signoffs = ", ".join(guide["signoff_examples"])
    return (
        f"Tone: {guide['tone']}\n"
        f"Greeting style: {guide['greeting_style']}\n"
        f"Body guidance:\n{bullets}\n"
        f"Possible sign-offs: {signoffs}"
    )



def resolve_reply_context(
    email_text: str,
    replier_email: str = "",
    replier_name: str = "",
    original_sender_email: str = "",
    original_sender_name: str = "",
    original_receiver_email: str = "",
    original_receiver_name: str = "",
    reply_target_email: str = "",
    reply_target_name: str = "",
) -> ResolvedReplyContext:
    metadata = extract_email_metadata_structured(
        email_text=email_text,
        default_sender_email=original_sender_email,
        default_receiver_email=original_receiver_email,
        default_sender_name=original_sender_name,
        default_receiver_name=original_receiver_name,
    )

    inferred_reply_target_email = (
        (reply_target_email or "").strip()
        or metadata.reply_to_email
        or metadata.sender_email
        or (original_sender_email or "").strip()
    )
    inferred_reply_target_name = (
        _normalize_whitespace(reply_target_name)
        or metadata.reply_to_name
        or metadata.sender_name
        or _normalize_whitespace(original_sender_name)
    )

    inferred_replier_email = (
        (replier_email or "").strip()
        or (original_receiver_email or "").strip()
        or metadata.receiver_email
    )
    inferred_replier_name = (
        _normalize_whitespace(replier_name)
        or _normalize_whitespace(original_receiver_name)
        or metadata.receiver_name
    )

    return ResolvedReplyContext(
        replier_name=inferred_replier_name,
        replier_email=inferred_replier_email,
        original_sender_name=_normalize_whitespace(original_sender_name) or metadata.sender_name,
        original_sender_email=(original_sender_email or "").strip() or metadata.sender_email,
        original_receiver_name=_normalize_whitespace(original_receiver_name) or metadata.receiver_name,
        original_receiver_email=(original_receiver_email or "").strip() or metadata.receiver_email,
        reply_target_name=inferred_reply_target_name,
        reply_target_email=inferred_reply_target_email,
        subject=metadata.subject,
        cc=metadata.cc,
        body_preview=metadata.body_preview,
    )



def _coerce_identity_inputs(
    *,
    replier_email: str = "",
    replier_name: str = "",
    original_sender_email: str = "",
    original_sender_name: str = "",
    original_receiver_email: str = "",
    original_receiver_name: str = "",
    reply_target_email: str = "",
    reply_target_name: str = "",
    sender_email: str = "",
    sender_name: str = "",
    receiver_email: str = "",
    receiver_name: str = "",
) -> Dict[str, str]:
    """Support clearer new names while remaining backward-compatible with older calls."""
    return {
        "replier_email": (replier_email or receiver_email or "").strip(),
        "replier_name": _normalize_whitespace(replier_name or receiver_name),
        "original_sender_email": (original_sender_email or sender_email or "").strip(),
        "original_sender_name": _normalize_whitespace(original_sender_name or sender_name),
        "original_receiver_email": (original_receiver_email or receiver_email or "").strip(),
        "original_receiver_name": _normalize_whitespace(original_receiver_name or receiver_name),
        "reply_target_email": (reply_target_email or sender_email or original_sender_email or "").strip(),
        "reply_target_name": _normalize_whitespace(reply_target_name or sender_name or original_sender_name),
    }


# ---------------------------------------------------------------------------
# Tool definitions the LLM can bind to.
# ---------------------------------------------------------------------------
@tool

def list_supported_reply_tones() -> Dict[str, List[str]]:
    """Return the supported email reply tones that the agent can use."""
    return {"supported_tones": sorted(TONE_GUIDES.keys())}


@tool

def extract_sender_details(
    email_text: str,
    default_sender_email: str = "",
    default_sender_name: str = "",
) -> Dict[str, str]:
    """Extract sender details from the original incoming email."""
    metadata = extract_email_metadata_structured(
        email_text=email_text,
        default_sender_email=default_sender_email,
        default_sender_name=default_sender_name,
    )
    return {
        "sender_name": metadata.sender_name,
        "sender_email": metadata.sender_email,
        "reply_to_name": metadata.reply_to_name,
        "reply_to_email": metadata.reply_to_email,
    }


@tool

def extract_receiver_details(
    email_text: str,
    default_receiver_email: str = "",
    default_receiver_name: str = "",
) -> Dict[str, str]:
    """Extract receiver details from the original incoming email."""
    metadata = extract_email_metadata_structured(
        email_text=email_text,
        default_receiver_email=default_receiver_email,
        default_receiver_name=default_receiver_name,
    )
    return {
        "receiver_name": metadata.receiver_name,
        "receiver_email": metadata.receiver_email,
    }


@tool

def extract_email_metadata(
    email_text: str,
    default_sender_email: str = "",
    default_receiver_email: str = "",
    default_sender_name: str = "",
    default_receiver_name: str = "",
) -> Dict[str, str]:
    """Extract sender/receiver names and emails, reply-to, subject, cc, and a short body preview."""
    metadata = extract_email_metadata_structured(
        email_text=email_text,
        default_sender_email=default_sender_email,
        default_receiver_email=default_receiver_email,
        default_sender_name=default_sender_name,
        default_receiver_name=default_receiver_name,
    )
    return asdict(metadata)


@tool

def resolve_reply_target(
    email_text: str,
    explicit_reply_target_email: str = "",
    explicit_reply_target_name: str = "",
    default_original_sender_email: str = "",
    default_original_sender_name: str = "",
) -> Dict[str, str]:
    """Resolve where the reply should go. Reply-To wins over From when available."""
    metadata = extract_email_metadata_structured(
        email_text=email_text,
        default_sender_email=default_original_sender_email,
        default_sender_name=default_original_sender_name,
    )
    target_email = (
        (explicit_reply_target_email or "").strip()
        or metadata.reply_to_email
        or metadata.sender_email
        or (default_original_sender_email or "").strip()
    )
    target_name = (
        _normalize_whitespace(explicit_reply_target_name)
        or metadata.reply_to_name
        or metadata.sender_name
        or _normalize_whitespace(default_original_sender_name)
    )
    return {
        "reply_target_name": target_name,
        "reply_target_email": target_email,
        "resolution_rule": "Reply-To > From > explicit/default original sender",
    }


@tool

def resolve_replier_identity(
    email_text: str,
    explicit_replier_email: str = "",
    explicit_replier_name: str = "",
    default_original_receiver_email: str = "",
    default_original_receiver_name: str = "",
) -> Dict[str, str]:
    """Resolve the identity from which the reply is being drafted."""
    metadata = extract_email_metadata_structured(
        email_text=email_text,
        default_receiver_email=default_original_receiver_email,
        default_receiver_name=default_original_receiver_name,
    )
    replier_email = (
        (explicit_replier_email or "").strip()
        or (default_original_receiver_email or "").strip()
        or metadata.receiver_email
    )
    replier_name = (
        _normalize_whitespace(explicit_replier_name)
        or _normalize_whitespace(default_original_receiver_name)
        or metadata.receiver_name
    )
    return {
        "replier_name": replier_name,
        "replier_email": replier_email,
        "resolution_rule": "explicit replier > original receiver > To header",
    }


@tool

def get_formal_tone_rules() -> Dict[str, object]:
    """Return writing rules for a formal email reply."""
    return TONE_GUIDES["formal"]


@tool

def get_warm_tone_rules() -> Dict[str, object]:
    """Return writing rules for a warm email reply."""
    return TONE_GUIDES["warm"]


@tool

def get_informal_tone_rules() -> Dict[str, object]:
    """Return writing rules for an informal email reply."""
    return TONE_GUIDES["informal"]


@tool

def get_concise_tone_rules() -> Dict[str, object]:
    """Return writing rules for a concise email reply."""
    return TONE_GUIDES["concise"]


@tool

def get_apologetic_tone_rules() -> Dict[str, object]:
    """Return writing rules for an apologetic email reply."""
    return TONE_GUIDES["apologetic"]


@tool

def get_assertive_tone_rules() -> Dict[str, object]:
    """Return writing rules for an assertive email reply."""
    return TONE_GUIDES["assertive"]


@tool

def get_follow_up_tone_rules() -> Dict[str, object]:
    """Return writing rules for a follow-up email reply."""
    return TONE_GUIDES["follow_up"]


@tool

def get_appreciative_tone_rules() -> Dict[str, object]:
    """Return writing rules for an appreciative email reply."""
    return TONE_GUIDES["appreciative"]


EMAIL_REPLY_TOOLS = [
    list_supported_reply_tones,
    extract_sender_details,
    extract_receiver_details,
    extract_email_metadata,
    resolve_reply_target,
    resolve_replier_identity,
    get_formal_tone_rules,
    get_warm_tone_rules,
    get_informal_tone_rules,
    get_concise_tone_rules,
    get_apologetic_tone_rules,
    get_assertive_tone_rules,
    get_follow_up_tone_rules,
    get_appreciative_tone_rules,
]



def get_email_reply_tools() -> List[Callable]:
    """Return the full tool list so callers can pass it to llm.bind_tools(...)."""
    return list(EMAIL_REPLY_TOOLS)


# ---------------------------------------------------------------------------
# Agent wrapper (matches the notebook structure, but for email replies).
# ---------------------------------------------------------------------------
class EmailReplyAgent:
    """LangChain tool-calling agent for grounded email-reply generation."""

    def __init__(self, model: Optional[str] = None, temperature: float = 0.2, verbose: bool = False):
        _require_langchain()
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.temperature = temperature
        self.verbose = verbose
        self.llm = ChatOpenAI(model=self.model, temperature=self.temperature)
        self.tools = get_email_reply_tools()
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You write high-quality email replies. "
                    "Always ground your work with tools before drafting. "
                    "First call extract_email_metadata. Then call resolve_reply_target and resolve_replier_identity. "
                    "Then call the tone-rule tool that matches the requested tone. "
                    "Do not invent names, emails, dates, commitments, or facts not supported by the original email or provided context. "
                    "If information is missing, keep the wording neutral and natural. "
                    "Write as the replier, not as the original sender. "
                    "Return only the final email reply body unless the user explicitly asks for analysis."
                ),
                (
                    "human",
                    "Draft an email reply.\n\n"
                    "Requested tone: {tone}\n"
                    "Replier name: {replier_name}\n"
                    "Replier email: {replier_email}\n"
                    "Original sender name: {original_sender_name}\n"
                    "Original sender email: {original_sender_email}\n"
                    "Original receiver name: {original_receiver_name}\n"
                    "Original receiver email: {original_receiver_email}\n"
                    "Reply target name: {reply_target_name}\n"
                    "Reply target email: {reply_target_email}\n"
                    "Extra instructions: {extra_instructions}\n\n"
                    "Original email:\n{email_text}"
                ),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        self.agent = (
            {
                "tone": lambda x: x["tone"],
                "replier_name": lambda x: x["replier_name"],
                "replier_email": lambda x: x["replier_email"],
                "original_sender_name": lambda x: x["original_sender_name"],
                "original_sender_email": lambda x: x["original_sender_email"],
                "original_receiver_name": lambda x: x["original_receiver_name"],
                "original_receiver_email": lambda x: x["original_receiver_email"],
                "reply_target_name": lambda x: x["reply_target_name"],
                "reply_target_email": lambda x: x["reply_target_email"],
                "extra_instructions": lambda x: x["extra_instructions"],
                "email_text": lambda x: x["email_text"],
                "agent_scratchpad": lambda x: format_to_openai_tool_messages(x["intermediate_steps"]),
            }
            | self.prompt
            | self.llm.bind_tools(self.tools)
            | OpenAIToolsAgentOutputParser()
        )
        self.executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=self.verbose)

    def invoke(self, payload: Dict[str, str]) -> Dict[str, object]:
        return self.executor.invoke(payload)

    def draft_reply(
        self,
        email_text: str,
        tone: str = "formal",
        replier_email: str = "",
        replier_name: str = "",
        original_sender_email: str = "",
        original_sender_name: str = "",
        original_receiver_email: str = "",
        original_receiver_name: str = "",
        reply_target_email: str = "",
        reply_target_name: str = "",
        extra_instructions: str = "",
        # Backward-compatible aliases
        sender_email: str = "",
        sender_name: str = "",
        receiver_email: str = "",
        receiver_name: str = "",
    ) -> str:
        canonical_tone = _canonical_tone(tone)
        ids = _coerce_identity_inputs(
            replier_email=replier_email,
            replier_name=replier_name,
            original_sender_email=original_sender_email,
            original_sender_name=original_sender_name,
            original_receiver_email=original_receiver_email,
            original_receiver_name=original_receiver_name,
            reply_target_email=reply_target_email,
            reply_target_name=reply_target_name,
            sender_email=sender_email,
            sender_name=sender_name,
            receiver_email=receiver_email,
            receiver_name=receiver_name,
        )
        result = self.invoke(
            {
                "tone": canonical_tone,
                "replier_name": ids["replier_name"] or "(unknown)",
                "replier_email": ids["replier_email"] or "(unknown)",
                "original_sender_name": ids["original_sender_name"] or "(unknown)",
                "original_sender_email": ids["original_sender_email"] or "(unknown)",
                "original_receiver_name": ids["original_receiver_name"] or "(unknown)",
                "original_receiver_email": ids["original_receiver_email"] or "(unknown)",
                "reply_target_name": ids["reply_target_name"] or "(unknown)",
                "reply_target_email": ids["reply_target_email"] or "(unknown)",
                "extra_instructions": extra_instructions or "(none)",
                "email_text": email_text.strip(),
            }
        )
        return str(result["output"]).strip()

    def suggest_replies(
        self,
        email_text: str,
        tones: Sequence[str] = ("formal", "warm"),
        replier_email: str = "",
        replier_name: str = "",
        original_sender_email: str = "",
        original_sender_name: str = "",
        original_receiver_email: str = "",
        original_receiver_name: str = "",
        reply_target_email: str = "",
        reply_target_name: str = "",
        extra_instructions: str = "",
        sender_email: str = "",
        sender_name: str = "",
        receiver_email: str = "",
        receiver_name: str = "",
    ) -> Dict[str, str]:
        replies: Dict[str, str] = {}
        for tone in tones:
            canonical = _canonical_tone(tone)
            replies[canonical] = self.draft_reply(
                email_text=email_text,
                tone=canonical,
                replier_email=replier_email,
                replier_name=replier_name,
                original_sender_email=original_sender_email,
                original_sender_name=original_sender_name,
                original_receiver_email=original_receiver_email,
                original_receiver_name=original_receiver_name,
                reply_target_email=reply_target_email,
                reply_target_name=reply_target_name,
                extra_instructions=extra_instructions,
                sender_email=sender_email,
                sender_name=sender_name,
                receiver_email=receiver_email,
                receiver_name=receiver_name,
            )
        return replies


class ReplySuggester:
    """Generates tool-grounded email reply suggestions."""

    def __init__(self, model: Optional[str] = None, temperature: float = 0.2, verbose: bool = False):
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.temperature = temperature
        self.verbose = verbose
        self._agent: Optional[EmailReplyAgent] = None
        self._direct_chain = None
        self._summary_chain = None

    def _get_agent(self) -> EmailReplyAgent:
        if self._agent is None:
            self._agent = EmailReplyAgent(model=self.model, temperature=self.temperature, verbose=self.verbose)
        return self._agent

    def _get_direct_chain(self):
        _require_langchain()
        if self._direct_chain is None:
            llm = ChatOpenAI(model=self.model, temperature=self.temperature)
            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You write high-quality email replies. "
                        "Use the provided reply context and tone guide. "
                        "Do not invent facts, names, commitments, or dates. "
                        "If details are missing, keep the reply neutral. "
                        "Write as the replier. "
                        "Return only the email reply body.",
                    ),
                    (
                        "human",
                        "Requested tone: {tone}\n\n"
                        "Tone guide:\n{tone_guide}\n\n"
                        "Resolved reply context:\n{resolved_context}\n\n"
                        "Extra instructions: {extra_instructions}\n\n"
                        "Original email:\n{email_text}",
                    ),
                ]
            )
            self._direct_chain = prompt | llm
        return self._direct_chain

    def _fallback_summary(self, text: str, heading: str = "Summary") -> str:
        """Small non-AI fallback so the UI still returns something useful."""
        cleaned = _normalize_whitespace(text)
        if not cleaned:
            return "No readable text was available to summarize."

        pieces = re.split(r"(?<=[.!?])\s+|\n+", cleaned)
        bullets = []
        for piece in pieces:
            piece = piece.strip(" -•\t")
            if len(piece) < 20:
                continue
            bullets.append(piece[:240])
            if len(bullets) >= 5:
                break

        if not bullets:
            bullets = [cleaned[:600]]

        return (
            "AI summary unavailable, so here is an extracted preview:\n"
            + "\n".join(f"- {item}" for item in bullets)
        )

    def _get_summary_chain(self):
        _require_langchain()
        if self._summary_chain is None:
            llm = ChatOpenAI(model=self.model, temperature=0.1)
            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You summarize emails and email attachments for a user. "
                        "Be concise, factual, and useful. Do not invent details. "
                        "Use 3 to 6 bullets. Include deadlines, requested actions, dates, "
                        "money amounts, people, and decisions when present. "
                        "If the source text is insufficient, say so clearly.",
                    ),
                    (
                        "human",
                        "Source type: {source_type}\n"
                        "Title or filename: {title}\n"
                        "Extra note: {note}\n\n"
                        "Text to summarize:\n{text}",
                    ),
                ]
            )
            self._summary_chain = prompt | llm
        return self._summary_chain

    def summarize_text(
        self,
        *,
        text: str,
        source_type: str = "email",
        title: str = "",
        note: str = "",
    ) -> str:
        """Summarize arbitrary mail/attachment text using OpenAI when available."""
        text = (text or "").strip()
        if not text:
            return "No readable text was available to summarize."

        # Keep requests predictable and avoid huge prompt payloads.
        text_for_model = text[:24000]
        if len(text) > len(text_for_model):
            note = (note + " " if note else "") + "Input was truncated before summarization."

        if not os.getenv("OPENAI_API_KEY"):
            return self._fallback_summary(text_for_model, heading=title or source_type)

        try:
            chain = self._get_summary_chain()
            response = chain.invoke(
                {
                    "source_type": source_type,
                    "title": title or "(untitled)",
                    "note": note or "(none)",
                    "text": text_for_model,
                }
            )
            return str(getattr(response, "content", response)).strip()
        except Exception:
            return self._fallback_summary(text_for_model, heading=title or source_type)

    def summarize_mail(self, *, email_text: str, subject: str = "") -> str:
        """Summarize the visible email body and headers."""
        return self.summarize_text(
            text=email_text,
            source_type="email",
            title=subject or "Email",
            note="Summarize the email message itself, not its attachments.",
        )

    def summarize_attachment(
        self,
        *,
        filename: str,
        mime_type: str,
        text: str,
        extraction_note: str = "",
    ) -> str:
        """Summarize extracted attachment text."""
        if not text.strip():
            return extraction_note or "No readable text could be extracted from this attachment."

        return self.summarize_text(
            text=text,
            source_type=f"attachment ({mime_type or 'unknown type'})",
            title=filename or "attachment",
            note=extraction_note or "Summarize this attachment separately from the email body.",
        )

    def suggest_by_tone(
        self,
        email_text: str,
        tone: str = "formal",
        replier_email: str = "",
        replier_name: str = "",
        original_sender_email: str = "",
        original_sender_name: str = "",
        original_receiver_email: str = "",
        original_receiver_name: str = "",
        reply_target_email: str = "",
        reply_target_name: str = "",
        extra_instructions: str = "",
        use_agent: bool = True,
        # Backward-compatible aliases
        sender_email: str = "",
        sender_name: str = "",
        receiver_email: str = "",
        receiver_name: str = "",
    ) -> str:
        canonical = _canonical_tone(tone)
        ids = _coerce_identity_inputs(
            replier_email=replier_email,
            replier_name=replier_name,
            original_sender_email=original_sender_email,
            original_sender_name=original_sender_name,
            original_receiver_email=original_receiver_email,
            original_receiver_name=original_receiver_name,
            reply_target_email=reply_target_email,
            reply_target_name=reply_target_name,
            sender_email=sender_email,
            sender_name=sender_name,
            receiver_email=receiver_email,
            receiver_name=receiver_name,
        )

        if use_agent:
            return self._get_agent().draft_reply(
                email_text=email_text,
                tone=canonical,
                replier_email=ids["replier_email"],
                replier_name=ids["replier_name"],
                original_sender_email=ids["original_sender_email"],
                original_sender_name=ids["original_sender_name"],
                original_receiver_email=ids["original_receiver_email"],
                original_receiver_name=ids["original_receiver_name"],
                reply_target_email=ids["reply_target_email"],
                reply_target_name=ids["reply_target_name"],
                extra_instructions=extra_instructions,
            )

        resolved = resolve_reply_context(
            email_text=email_text,
            replier_email=ids["replier_email"],
            replier_name=ids["replier_name"],
            original_sender_email=ids["original_sender_email"],
            original_sender_name=ids["original_sender_name"],
            original_receiver_email=ids["original_receiver_email"],
            original_receiver_name=ids["original_receiver_name"],
            reply_target_email=ids["reply_target_email"],
            reply_target_name=ids["reply_target_name"],
        )
        chain = self._get_direct_chain()
        response = chain.invoke(
            {
                "tone": canonical,
                "tone_guide": _tone_guide_text(canonical),
                "resolved_context": asdict(resolved),
                "extra_instructions": extra_instructions or "(none)",
                "email_text": email_text,
            }
        )
        return response.content.strip()

    def suggest_many(
        self,
        email_text: str,
        tones: Sequence[str],
        replier_email: str = "",
        replier_name: str = "",
        original_sender_email: str = "",
        original_sender_name: str = "",
        original_receiver_email: str = "",
        original_receiver_name: str = "",
        reply_target_email: str = "",
        reply_target_name: str = "",
        extra_instructions: str = "",
        use_agent: bool = True,
        sender_email: str = "",
        sender_name: str = "",
        receiver_email: str = "",
        receiver_name: str = "",
    ) -> Dict[str, str]:
        return {
            _canonical_tone(tone): self.suggest_by_tone(
                email_text=email_text,
                tone=tone,
                replier_email=replier_email,
                replier_name=replier_name,
                original_sender_email=original_sender_email,
                original_sender_name=original_sender_name,
                original_receiver_email=original_receiver_email,
                original_receiver_name=original_receiver_name,
                reply_target_email=reply_target_email,
                reply_target_name=reply_target_name,
                extra_instructions=extra_instructions,
                use_agent=use_agent,
                sender_email=sender_email,
                sender_name=sender_name,
                receiver_email=receiver_email,
                receiver_name=receiver_name,
            )
            for tone in tones
        }

    def suggest_two(
        self,
        email_text: str,
        replier_email: str = "",
        replier_name: str = "",
        original_sender_email: str = "",
        original_sender_name: str = "",
        original_receiver_email: str = "",
        original_receiver_name: str = "",
        reply_target_email: str = "",
        reply_target_name: str = "",
        extra_instructions: str = "",
        use_agent: bool = True,
        # Backward-compatible aliases
        sender_email: str = "",
        sender_name: str = "",
        receiver_email: str = "",
        receiver_name: str = "",
    ) -> Tuple[str, str]:
        """
        Backward-compatible API used by main.py.

        Returns:
          suggestion_1: formal/professional
          suggestion_2: warm/friendly
        """
        replies = self.suggest_many(
            email_text=email_text,
            tones=("formal", "warm"),
            replier_email=replier_email,
            replier_name=replier_name,
            original_sender_email=original_sender_email,
            original_sender_name=original_sender_name,
            original_receiver_email=original_receiver_email,
            original_receiver_name=original_receiver_name,
            reply_target_email=reply_target_email,
            reply_target_name=reply_target_name,
            extra_instructions=extra_instructions,
            use_agent=use_agent,
            sender_email=sender_email,
            sender_name=sender_name,
            receiver_email=receiver_email,
            receiver_name=receiver_name,
        )
        return replies["formal"], replies["warm"]



def build_email_reply_agent(model: Optional[str] = None, temperature: float = 0.2, verbose: bool = False) -> EmailReplyAgent:
    """Convenience factory mirroring the notebook pattern."""
    return EmailReplyAgent(model=model, temperature=temperature, verbose=verbose)


__all__ = [
    "ReplySuggester",
    "EmailReplyAgent",
    "ResolvedReplyContext",
    "build_email_reply_agent",
    "build_email_context",
    "extract_email_metadata_structured",
    "resolve_reply_context",
    "get_email_reply_tools",
    "EMAIL_REPLY_TOOLS",
    "extract_sender_details",
    "extract_receiver_details",
    "extract_email_metadata",
    "resolve_reply_target",
    "resolve_replier_identity",
    "list_supported_reply_tones",
    "get_formal_tone_rules",
    "get_warm_tone_rules",
    "get_informal_tone_rules",
    "get_concise_tone_rules",
    "get_apologetic_tone_rules",
    "get_assertive_tone_rules",
    "get_follow_up_tone_rules",
    "get_appreciative_tone_rules",
]
