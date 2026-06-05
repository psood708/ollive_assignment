import re
from typing import Callable, Optional, Tuple

_JAILBREAK_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"pretend\s+(you\s+are|to\s+be)",
    r"\bDAN\b",
    r"developer\s+mode",
    r"jailbreak",
    r"bypass\s+(your\s+)?(safety|restrictions|guidelines)",
]
_INPUT_REGEX = re.compile("|".join(_JAILBREAK_PATTERNS), re.IGNORECASE)
_SSN_REGEX = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_CC_REGEX = re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b")

SAFE_REPLY = "I'm sorry, I can't help with that."
_toxic_classifier = None
_USE_DEFAULT = object()  # sentinel: "load the real classifier"

def check_input(message: str) -> Tuple[bool, str]:
    if _INPUT_REGEX.search(message):
        return True, SAFE_REPLY
    return False, ""

def check_output(reply: str, _classifier=_USE_DEFAULT) -> Tuple[bool, str]:
    filtered = _SSN_REGEX.sub("[REDACTED]", reply)
    filtered = _CC_REGEX.sub("[REDACTED]", filtered)
    triggered = filtered != reply

    # _classifier=None means skip; _classifier=_USE_DEFAULT means load real one
    classifier = _get_toxic_classifier() if _classifier is _USE_DEFAULT else _classifier
    if classifier:
        try:
            result = classifier(filtered[:512])[0]
            if result["label"].lower() in ("toxic", "label_1") and result["score"] > 0.8:
                filtered = SAFE_REPLY + " [Response filtered for safety]"
                triggered = True
        except Exception:
            pass

    return triggered, filtered

def _get_toxic_classifier() -> Optional[Callable]:
    global _toxic_classifier
    if _toxic_classifier is None:
        try:
            from transformers import pipeline
            _toxic_classifier = pipeline(
                "text-classification",
                model="martin-ha/toxic-comment-model",
                device=-1,
            )
        except Exception:
            return None
    return _toxic_classifier
