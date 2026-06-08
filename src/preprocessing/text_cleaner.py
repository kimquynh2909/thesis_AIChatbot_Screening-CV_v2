from __future__ import annotations

import html
import re
import unicodedata


def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFKC", text or "")


def strip_html(text: str) -> str:
    text = html.unescape(text or "")
    text = re.sub(r"<[^>]+>", " ", text)
    return text


def clean_text(text: str, preserve_case: bool = False) -> str:
    """Clean noisy resume or JD text while preserving technical tokens."""
    text = normalize_unicode(strip_html(text))
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    text = re.sub(r"[-=_]{4,}", " ", text)
    text = re.sub(r"[^\w\s.,;:()&/#%+\-@]", " ", text)
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    text = re.sub(r"([,.;:])(?=\S)", r"\1 ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s+", "\n", text)
    text = text.strip()
    return text if preserve_case else text.lower()


def clean_for_display(text: str) -> str:
    return clean_text(text, preserve_case=True)


def clean_for_matching(text: str) -> str:
    return clean_text(text, preserve_case=False)


def tokenize_words(text: str) -> list[str]:
    text = clean_for_matching(text)
    return re.findall(r"[a-zA-Z0-9+#.]+", text)
