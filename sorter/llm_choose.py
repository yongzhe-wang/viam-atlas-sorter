"""
Ask a vision-language model (Claude) which detected object a description means, when CLIP is unsure.

WHY: CLIP is weak on colour + shape words and cannot do relations ("the smaller cube", "the one left of the
cup", "something to write with"). Perception already gives exact object outlines from depth, so the model is
never asked for coordinates: we draw a numbered box on every detected object and ask for a number. The
geometry the grasp uses still comes from perception.

Off by default. Enable with env SORTER_LLM=fallback (default when enabled: only called when CLIP's best()
refuses or is ambiguous) or SORTER_LLM=always (every decision; slower, needs the network every time).
Needs ANTHROPIC_API_KEY. Optional: SORTER_LLM_MODEL (default claude-sonnet-5-5), SORTER_LLM_TIMEOUT_S (8).
Any failure (no key, no network, timeout, unparsable answer) returns None so the caller keeps CLIP's answer:
the LLM can only add picks, never break the offline path.

Contract: choose(rgb, objs, text) -> {"index", "status", "reason", "ms"} or None.
  status "ok"        index = position in objs of the object meant
  status "none"      nothing on the table matches (reason says why)
  status "ambiguous" several match (reason names them)
"""
import base64
import io
import json
import os
import re
import time
import urllib.error
import urllib.request

import numpy as np

API_URL = "https://api.anthropic.com/v1/messages"
DEFAULT_MODEL = "claude-sonnet-5-5"
MAX_IMAGE_PX = 1024  # longest side sent to the API: enough to read the labels, keeps the upload small
BOX_RGB = (255, 0, 255)  # magenta: rarely the colour of an object or the blue tape

PROMPT = """You are the perception helper of a robot arm. The photo is taken by a camera looking straight \
down at a table. Every object the robot can grasp has a magenta box with a white number on a magenta tag.

The user asked the robot to pick up: "{text}"

Decide which numbered object the user means. Use colour, shape, size, position and what the object is.
Answer with ONLY one JSON object on a single line, no other text:
{{"status": "ok", "number": <n>, "reason": "<short why>"}}            if exactly one numbered object fits
{{"status": "ambiguous", "numbers": [<n>, ...], "reason": "<short>"}}  if several fit about equally well
{{"status": "none", "reason": "<short why>"}}                          if no numbered object fits
Only use numbers that appear on the tags."""


def mode() -> str:
    """'off' | 'fallback' | 'always' from env SORTER_LLM; anything unknown counts as 'fallback'."""
    m = os.environ.get("SORTER_LLM", "off").strip().lower()
    if m in ("", "0", "off", "false", "no"):
        return "off"
    return "always" if m == "always" else "fallback"


def enabled() -> bool:
    return mode() != "off" and bool(os.environ.get("ANTHROPIC_API_KEY"))


def annotate(rgb: np.ndarray, objs: list) -> bytes:
    """JPEG of the frame with a numbered box (1..N) on every object's bbox, scaled to <= MAX_IMAGE_PX."""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.fromarray(rgb)
    s = min(1.0, MAX_IMAGE_PX / max(img.size))
    if s < 1.0:
        img = img.resize((round(img.width * s), round(img.height * s)))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default(size=22)
    except TypeError:  # Pillow < 10.1
        font = ImageFont.load_default()
    for n, o in enumerate(objs, start=1):
        u0, v0, u1, v1 = (round(c * s) for c in o.bbox)
        draw.rectangle((u0, v0, u1, v1), outline=BOX_RGB, width=3)
        label = str(n)
        tw, th = draw.textbbox((0, 0), label, font=font)[2:]
        ty = v0 - th - 6 if v0 - th - 6 >= 0 else v1 + 2
        draw.rectangle((u0, ty, u0 + tw + 8, ty + th + 6), fill=BOX_RGB)
        draw.text((u0 + 4, ty + 2), label, fill=(255, 255, 255), font=font)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=85)
    return buf.getvalue()


def parse(answer: str, n_objs: int) -> dict:
    """The model's reply -> {"index", "status", "reason"}; raises ValueError when it is not usable."""
    m = re.search(r"\{.*\}", answer, re.S)
    if not m:
        raise ValueError(f"no JSON in answer: {answer[:120]!r}")
    d = json.loads(m.group(0))
    status = str(d.get("status", "")).lower()
    reason = str(d.get("reason", ""))[:200]
    if status == "ok":
        n = int(d["number"])
        if not 1 <= n <= n_objs:
            raise ValueError(f"number {n} is not on a tag (1..{n_objs})")
        return {"index": n - 1, "status": "ok", "reason": reason}
    if status == "ambiguous":
        nums = [int(x) for x in d.get("numbers") or [] if 1 <= int(x) <= n_objs]
        if nums:
            reason = f"objects {', '.join(map(str, nums))} all fit" + (f": {reason}" if reason else "")
        return {"index": None, "status": "ambiguous", "reason": reason}
    if status == "none":
        return {"index": None, "status": "none", "reason": reason}
    raise ValueError(f"unknown status {status!r}")


def _call(jpeg: bytes, text: str) -> str:
    """One Messages API call; returns the reply text. Raises on HTTP or network errors."""
    body = {
        "model": os.environ.get("SORTER_LLM_MODEL", DEFAULT_MODEL),
        "max_tokens": 200,
        "messages": [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                         "data": base64.b64encode(jpeg).decode()}},
            {"type": "text", "text": PROMPT.format(text=text.replace('"', "'"))},
        ]}],
    }
    req = urllib.request.Request(API_URL, data=json.dumps(body).encode(), method="POST", headers={
        "content-type": "application/json",
        "x-api-key": os.environ["ANTHROPIC_API_KEY"],
        "anthropic-version": "2023-06-01",
    })
    with urllib.request.urlopen(req, timeout=float(os.environ.get("SORTER_LLM_TIMEOUT_S", 8))) as r:
        reply = json.loads(r.read())
    return "".join(b.get("text", "") for b in reply.get("content", []) if b.get("type") == "text")


LAST = {}  # last call's outcome, for the module's status: {"status", "reason", "ms", "error"}


def choose(rgb: np.ndarray, objs: list, text: str) -> dict:
    """
    Which of `objs` (perception.Obj, bboxes on `rgb`) the description `text` means, or None if the LLM is
    disabled or failed (the caller then keeps its own decision). Never raises.
    """
    if not enabled() or rgb is None or not objs:
        return None
    t0 = time.monotonic()
    try:
        out = parse(_call(annotate(rgb, objs), text), len(objs))
        out["ms"] = round((time.monotonic() - t0) * 1000)
        LAST.clear()
        LAST.update(status=out["status"], reason=out["reason"], ms=out["ms"], error=None, ts=time.time())
        return out
    except (urllib.error.URLError, OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as e:
        LAST.clear()
        LAST.update(status="error", error=f"{type(e).__name__}: {e}"[:200],
                    ms=round((time.monotonic() - t0) * 1000), ts=time.time())
        return None
