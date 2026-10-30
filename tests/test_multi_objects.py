"""split_objects: the user's sentence, spacing, single items, stray commas, aliases with 'and' in them."""
import sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
for n in ("torch", "open_clip"):
    sys.modules.setdefault(n, SimpleNamespace(set_num_threads=lambda k: None))
from sorter.multi_objects import split_objects
NAMES = ["black tape", "black tape roll", "roll of tape", "tape", "blue screwdriver", "Phillips screwdriver",
         "blue and black screwdriver", "blue-handle screwdriver", "red screwdriver", "granola bar", "black and green bar",
         "kind bar", "KIND bar", "candy", "snack bar", "utility knife", "box cutter", "silver box cutter", "blue block",
         "blue box", "yellow block", "green cube", "green square", "computer mouse", "mouse", "red knife"]
CASES = [
    ("i want screw driver, and snack, and tape", ["screwdriver", "snack", "tape"]),
    ("screw driver", ["screwdriver"]),
    ("tape", ["tape"]),
    ("grasp green square", ["green square"]),
    ("grasp the green square", ["green square"]),
    ("pick up the black tape", ["black tape"]),
    ("tape,, ,snack,", ["tape", "snack"]),
    (" , and ,", []),
    ("", []),
    ("blue and black screwdriver", ["blue and black screwdriver"]),
    ("blue and black screw driver", ["blue and black screwdriver"]),
    ("black and green bar", ["black and green bar"]),
    ("blue and black screwdriver, tape", ["blue and black screwdriver", "tape"]),
    ("tape and black and green bar and mouse", ["tape", "black and green bar", "mouse"]),
    ("tape and tape", ["tape"]),
    ("tape, Tape, TAPE", ["tape"]),
    ("screwdriver and screw driver", ["screwdriver"]),
    ("pick up the tape then the snack", ["tape", "snack"]),
    ("get me a snack & the mouse + yellow block", ["snack", "mouse", "yellow block"]),
    ("tape and then the mouse", ["tape", "mouse"]),
    ("please can you get me the tape and the mouse please", ["tape", "mouse"]),
    ("put the tape and the mouse into the blue square", ["tape", "mouse"]),
    ("boxcutter", ["box cutter"]),
    ("silver boxcutter", ["silver box cutter"]),
    ("Kind Bar", ["KIND bar"]),
    ("red screwdriver and blue screwdriver", ["red screwdriver", "blue screwdriver"]),
    ("the blue box", ["blue box"]),
    ("something shiny", ["something shiny"]),
    # Colours sharing a noun: never a bare colour in front of the rest (a bare "black" takes any black object)
    ("red and blue screwdriver", ["red screwdriver", "blue screwdriver"]),
    ("the red and the blue screw driver", ["red screwdriver", "blue screwdriver"]),
    ("tape and red and blue screwdriver", ["tape", "red screwdriver", "blue screwdriver"]),
    ("blue and yellow block, mouse", ["blue block", "yellow block", "mouse"]),
    ("red and blue screwdrivers please", ["red screwdriver", "blue screwdriver"]),
    ("black and yellow tape", ["black and yellow tape"]),
    ("green and yellow bars", ["green bar", "yellow bar"]),
    ("red and blue ones", ["red one", "blue one"]),
    ("red and blue one", ["red and blue one"]),
    ("black and blue scissors", ["black and blue scissors"]),
    ("red, blue and black mug", ["red and blue and black mug"]),
    ("green and black bar", ["black and green bar"]),
    ("black & green bar", ["black and green bar"]),
    ("black-and-green bar", ["black and green bar"]),
    ("red-and-blue screwdriver", ["red screwdriver", "blue screwdriver"]),
    ("black and blue screw driver and tape", ["blue and black screwdriver", "tape"]),
    # A colour with nothing after it stays an item of its own
    ("red and blue", ["red", "blue"]),
    ("red and blue and tape", ["red", "blue", "tape"]),
    ("red and tape", ["red", "tape"]),
    ("red and blue then tape", ["red", "blue", "tape"]),
    # "&" and "+" inside a word belong to its name (robot test: preview "M&M" -> "'m' is not in view")
    ("M&M", ["m&m"]),
    ("M&M's", ["m&m's"]),
    ("the m&m bag and tape", ["m&m bag", "tape"]),
    ("tape & mouse", ["tape", "mouse"]),
    ("tape &mouse", ["tape", "mouse"]),
    ("tape& mouse", ["tape", "mouse"]),
]
bad = 0
for text, want in CASES:
    got = split_objects(text, NAMES)
    if got != want:
        bad += 1
        print(f"FAIL {text!r}: {got} != {want}")
long = "tape, mouse, snack, red knife, green cube, yellow block, blue block"
assert len(split_objects(long, NAMES)) == 5 and len(split_objects(long, NAMES, limit=None)) == 7
assert split_objects("screw driver and tape") == ["screw driver", "tape"]  # no library: cut only
print(f"{len(CASES) - bad}/{len(CASES)} split cases ok; 7 objects -> first 5")
sys.exit(1 if bad else 0)
