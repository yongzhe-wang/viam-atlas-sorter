"""
One command naming several objects -> the objects, in order (user: "i want screw driver, and snack, and tape").

split_objects() drops the spoken lead-in, cuts at "," "and" "&" "then" "+", and spells each item the way the object
library does ("screw driver" -> "screwdriver"). Library names with a separator in them ("blue and black screwdriver")
stay whole. No model and no arm here: plain text in, plain text out.
"""
import re

from sorter.clip_embed import STOP

MAX_OBJECTS = 5
# Spoken lead-ins on top of clip_embed.STOP's command verbs; tried over and over ("please can you get me the tape")
LEAD = re.compile(r"^(please|pls|hey|ok|okay|now|also|and|then|can you|could you|would you|will you|i want|i wanna|"
                  r"i would like|i'd like|i need|give me|get me|bring me|hand me|grab me|pass me|find me|fetch me|"
                  r"fetch|bring|find)\b[\s,:]*")
ARTICLE = re.compile(r"^(the|a|an|some|my|that|this)\s+")
# The destination is always the blue square, and the politeness adds nothing to the object's name
TAIL = re.compile(r"\s+((in|into|to|onto|inside)\s+(the\s+)?(blue\s+)?(square|box)|please|for me|thanks|thank you)$")
SEPARATOR = re.compile(r"\s*(?:[,;&+]|\band\s+then\b|\band\b|\bthen\b)\s*")
_TOKEN = "\x00{}\x00"  # stands in for a library name with a separator in it while the text is cut


def _squash(text: str) -> str:
    """Letters and digits only, for comparing names regardless of spaces and case."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _strip(text: str) -> str:
    """Drop lead-ins, verbs and articles in front of an object's name, and what may follow it."""
    q, before = text.strip().lower().strip(" ,;:.!?"), None
    while q != before:
        before = q
        q = LEAD.sub("", q)
        if q == before:  # only then the verbs: STOP would leave "me the tape" of "get me the tape"
            q = re.sub(STOP, "", q)
        q = ARTICLE.sub("", q)
        q = TAIL.sub("", q).strip(" ,;:.!?")
    return q


def _spell(item: str, whole: dict, words: dict) -> str:
    """The item as the library spells it: a whole name ignoring spaces, else adjacent words merged into a known word."""
    if _squash(item) in whole:
        return whole[_squash(item)]
    parts, out, i = item.split(), [], 0
    while i < len(parts):
        for j in range(min(len(parts), i + 3), i + 1, -1):  # "screw driver" -> "screwdriver"
            merged = _squash("".join(parts[i:j]))
            if merged in words:
                out.append(words[merged])
                i = j
                break
        else:
            out.append(parts[i])
            i += 1
    return " ".join(out)


def split_objects(text: str, names=(), limit=MAX_OBJECTS) -> list:
    """
    The objects `text` names, in order, without repeats, at most `limit` (None: all, so a caller can report the rest).

    names: the library's names and aliases. A sentence that is one of them is one object, and one inside a list stays
    whole. Without names the text is still cut, only the spelling is left alone.
    """
    names = [n for n in names if n and n.strip()]
    whole = {_squash(n): n for n in names}
    words = {_squash(w): w.lower() for n in names for w in n.split() if _squash(w)}
    q = _strip(text or "")
    if _squash(q) in whole:
        return [whole[_squash(q)]]
    # Protect the names that a cut would break; spaces in them are optional ("blue and black screw driver")
    kept = []
    for n in sorted((n for n in names if SEPARATOR.search(n.lower())), key=len, reverse=True):
        pattern = r"\b" + r"\s*".join(re.escape(c) for c in _squash(n)) + r"\b"
        if re.search(pattern, q):
            q = re.sub(pattern, _TOKEN.format(len(kept)), q)
            kept.append(n)
    out, seen = [], set()
    for part in SEPARATOR.split(q):
        item = _strip(part)
        for k, n in enumerate(kept):
            item = item.replace(_TOKEN.format(k), n)
        item = _spell(item, whole, words) if item else ""
        if item and _squash(item) not in seen:
            seen.add(_squash(item))
            out.append(item)
    return out if limit is None else out[:limit]
