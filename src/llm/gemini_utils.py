from __future__ import annotations

import os

DEFAULT_GEMINI_CHAT_MODEL = "gemini-2.5-flash"

PREFERRED_CHAT_MODELS = (
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-flash-latest",
    "gemini-2.5-flash-lite",
)


def get_google_api_key() -> str | None:
    return os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")


def resolve_gemini_chat_model(api_key: str, configured_model: str | None = None) -> str:
    configured = normalize_model_name(configured_model or os.getenv("GEMINI_MODEL") or DEFAULT_GEMINI_CHAT_MODEL)
    available = list_generate_content_models(api_key)
    if not available:
        return configured
    if configured in available:
        return configured
    for model_name in PREFERRED_CHAT_MODELS:
        if model_name in available:
            return model_name
    return sorted(available)[0]


def normalize_model_name(model_name: str) -> str:
    return model_name.removeprefix("models/").strip()


def list_generate_content_models(api_key: str) -> set[str]:
    try:
        import google.generativeai as genai
    except ImportError:
        return set()

    try:
        genai.configure(api_key=api_key)
        available = set()
        for model in genai.list_models():
            methods = getattr(model, "supported_generation_methods", []) or []
            if "generateContent" in methods:
                available.add(normalize_model_name(model.name))
        return available
    except Exception:
        return set()
