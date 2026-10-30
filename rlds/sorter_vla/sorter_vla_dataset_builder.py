"""
TFDS builder: raw recorder episodes -> RLDS (TFRecord) dataset `sorter_vla`, Open X-Embodiment style.

Build (needs `pip install tensorflow tensorflow-datasets`), from this folder:
    SORTER_VLA_RAW=/path/to/dataset_raw tfds build                    # config "rgb" (default)
    SORTER_VLA_RAW=/path/to/dataset_raw tfds build --config rgbd      # + depth
Output: ~/tensorflow_datasets/sorter_vla/<config>/<version>/ (change with --data_dir).
Env: SORTER_VLA_RAW (default: <repo>/dataset_raw), SORTER_VLA_ONLY_SUCCESS=1 (drop failed episodes),
     SORTER_VLA_KINDS=grasp,reset (which episode kinds; default both).
Splits come from <raw>/splits.json ({"train": [sessions], "val": [...]}), else every session is "train".

Load:
    import tensorflow_datasets as tfds
    ds = tfds.load("sorter_vla/rgb", split="train")
    for ep in ds.take(1):
        for step in ep["steps"]:
            step["observation"]["image"], step["action"], step["language_instruction"]

The schema and the conversion live in convert.py (numpy only, tested without TensorFlow); this file only maps
convert.spec() onto TFDS feature types.
"""
import os
import sys
from pathlib import Path

import numpy as np
import tensorflow_datasets as tfds

sys.path.insert(0, str(Path(__file__).resolve().parent))
import convert  # noqa: E402

DEFAULT_RAW = Path(__file__).resolve().parents[2] / "dataset_raw"


def _feature(s):
    """convert.spec() leaf or dict -> TFDS feature."""
    if isinstance(s, dict):
        return tfds.features.FeaturesDict({k: _feature(v) for k, v in s.items()})
    kind = s[0]
    if kind == "text":
        return tfds.features.Text()
    if kind == "scalar":
        return tfds.features.Scalar(dtype=np.dtype(s[1]))
    if kind == "image":
        return tfds.features.Image(shape=s[1], dtype=np.dtype(s[2]), encoding_format="png" if s[2] == "uint16" else "jpeg")
    return tfds.features.Tensor(shape=s[1], dtype=np.dtype(s[2]))


class Builder(tfds.core.GeneratorBasedBuilder):
    """Wrist-camera pick-and-place episodes from the Viam x MongoDB sorter robot (xArm6 + G2 + RealSense)."""

    VERSION = tfds.core.Version(convert.SPEC_VERSION)
    RELEASE_NOTES = {"1.0.0": "First release: wrist RGB(-D), EE state, joints, EE delta actions, language."}
    BUILDER_CONFIGS = [
        tfds.core.BuilderConfig(name="rgb", description="Wrist colour 256x256."),
        tfds.core.BuilderConfig(name="rgbd", description="Wrist colour + depth (mm) 256x256."),
    ]

    def _depth(self) -> bool:
        return self.builder_config.name == "rgbd"

    def _info(self) -> tfds.core.DatasetInfo:
        sp = convert.spec(depth=self._depth())
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                "steps": tfds.features.Dataset({k: _feature(v) for k, v in sp["steps"].items()}),
                "episode_metadata": _feature(sp["episode_metadata"]),
            }),
            description=__doc__.split("\n\n")[0],
        )

    def _split_generators(self, dl_manager):
        raw = Path(os.environ.get("SORTER_VLA_RAW", DEFAULT_RAW))
        if not raw.exists():
            raise FileNotFoundError(f"no raw episodes at {raw}: set SORTER_VLA_RAW")
        return {name: self._generate_examples(raw, sessions) for name, sessions in convert.splits(raw).items()}

    def _generate_examples(self, raw: Path, sessions):
        only_success = os.environ.get("SORTER_VLA_ONLY_SUCCESS") == "1"
        kinds = os.environ.get("SORTER_VLA_KINDS")
        kinds = set(kinds.split(",")) if kinds else None
        for ep in convert.find_episodes(raw, sessions, only_success, kinds):
            yield f"{ep.parent.name}/{ep.name}", convert.episode_to_rlds(ep, depth=self._depth())
