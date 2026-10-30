"""
One command naming several objects -> the objects, in order (user: "i want screw driver, and snack, and tape").

split_objects() drops the spoken lead-in, cuts at "," "and" "&" "then" "+", and spells each item the way the object
library does ("screw driver" -> "screwdriver"). Library names with a separator in them ("blue and black screwdriver")
stay whole. No model and no arm here: plain text in, plain text out.
"""
import re

from sorter.clip_embed import COLOURS, STOP

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
# Colour words: the query's (clip_embed.COLOURS) plus those only the library's names use
_COLOUR = "|".join(sorted(COLOURS | {"silver", "teal", "gray"}))
# "red and blue screwdriver": colours joined by a separator, then the words of the one noun they share. A trailing
# colour has no noun, so "red and blue" and "red and blue and tape" stay cut
_WORD = r"(?!(?:and|then)\b)[a-z][\w-]*"
SHARED_NOUN = re.compile(rf"\b((?:{_COLOUR})(?:\s*(?:,|&|\band\b)\s*(?:the\s+)?(?:{_COLOUR}))+)\s+({_WORD}(?:\s+{_WORD})*)")


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


def _colours(text: str) -> list:
    return re.findall(rf"\b(?:{_COLOUR})\b", text)


def _shared_noun(colours: list, noun: str, two_tone: dict, whole: dict, words: dict) -> list:
    """
    The objects meant by several colours in front of one noun, as the library has them.

    WHY: cut at "and", "black and yellow tape" left a bare "black", and the arm would take any black object.
    A two-tone name of the library is one object, in any word order ("green and black bar"). The noun with each colour
    being a library name, or a plural, means several ("red and blue screwdriver", "green and blue blocks"). Anything
    else describes one object; the matcher accepts either colour.
    """
    named = two_tone.get((frozenset(colours), _squash(noun)))
    if named:
        return [named]
    # A plural counts only if the library knows its singular: "scissors" stays as said
    head, _, last = noun.rpartition(" ")
    singular = last[:-1] if last.endswith("s") else ""
    plural = singular == "one" or _squash(singular) in words
    one = f"{head} {singular}".strip() if plural else noun
    apart = [f"{c} {one}" for c in colours]
    if plural or all(_squash(a) in whole for a in apart):
        return apart
    return [f"{' and '.join(colours)} {noun}"]


def split_objects(text: str, names=(), limit=MAX_OBJECTS) -> list:
    """
    The objects `text` names, in order, without repeats, at most `limit` (None: all, so a caller can report the rest).

    names: the library's names and aliases. A sentence that is one of them is one object, and one inside a list stays
    whole. Without names the text is still cut, only the spelling is left alone.
    """
    names = [n for n in names if n and n.strip()]
    whole = {_squash(n): n for n in names}
    words = {_squash(w): w.lower() for n in names for w in n.split() if _squash(w)}
    q = re.sub(r"(?<=\w)-and-(?=\w)", " and ", _strip(text or ""))  # "red-and-blue" says the same as "red and blue"
    if _squash(q) in whole:
        return [whole[_squash(q)]]
    # Protect the names that a cut would break; spaces in them are optional ("blue and black screw driver")
    kept = []
    for n in sorted((n for n in names if SEPARATOR.search(n.lower())), key=len, reverse=True):
        pattern = r"\b" + r"\s*".join(re.escape(c) for c in _squash(n)) + r"\b"
        if re.search(pattern, q):
            q = re.sub(pattern, _TOKEN.format(len(kept)), q)
            kept.append(n)
    # Colours sharing a noun become whole objects before the cut, for the same reason
    two_tone = {(frozenset(_colours(m[1])), _squash(m[2])): n for n in names
                for m in [SHARED_NOUN.fullmatch(n.lower())] if m}

    def resolve(m) -> str:
        objects = _shared_noun(_colours(m[1]), _strip(m[2]), two_tone, whole, words)
        kept.extend(objects)
        return ", ".join(_TOKEN.format(k) for k in range(len(kept) - len(objects), len(kept)))

    q = SHARED_NOUN.sub(resolve, q)
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
