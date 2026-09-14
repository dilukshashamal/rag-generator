import re
import unicodedata


class ValidationError(ValueError):
    """Contains only an intentional user-safe validation message."""


INJECTION_PATTERNS = (
    r"ignore\s+(all\s+)?(previous|prior|system|above)\s+(instructions|prompts)",
    r"(reveal|show|print|repeat|give|expose).{0,40}(system\s+prompt|api\s*key|secrets?|credentials|server\s+files?|hidden\s+instructions)",
    r"(you are now|act as).{0,20}(system|developer|admin)",
    r"<\|?(system|im_start)\|?>|\[INST\]|</?system>",
    r"(execute|run).{0,20}(shell|command|script)|/etc/passwd|\.env\b",
)


def injection_flag(text: str) -> bool:
    normalized = unicodedata.normalize("NFKC", text)
    return any(re.search(pattern, normalized, re.I | re.S) for pattern in INJECTION_PATTERNS)


def validate_question(question: str) -> str:
    if len(question) > 1000:
        raise ValidationError("Questions must be at most 1,000 characters.")
    normalized = " ".join(unicodedata.normalize("NFKC", question).split())
    if not normalized:
        raise ValidationError("Enter a question before submitting.")
    if len(normalized) > 1000:
        raise ValidationError("Questions must be at most 1,000 characters.")
    if injection_flag(normalized):
        raise ValidationError("This request attempts to override instructions or access protected information.")
    return normalized
