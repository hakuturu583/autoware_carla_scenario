# SPDX-License-Identifier: Apache-2.0
"""Write a world's map as files, once, for drivers to read.

alpasim's services read a scene's vector map from its artifact, never off the
wire; this is the CARLA counterpart. The runtime takes the world's OpenDRIVE
(``carla.Map.to_opendrive()``), has roadgen convert it into whatever format a
driver reads, and writes the result under ``<map_dir>/<map_id>/``. Each step
then carries only what changes -- the traffic lights -- which
:mod:`carla_driver_interface.hdmap.files` resolves against these files.

Every set also holds the OpenDRIVE itself and roadgen's IR, and each format its
roadgen trace: the IR keeps OpenDRIVE's road and signal ids, and the trace says
which element of the format each IR element became, which is how a light named
by OpenDRIVE finds its stop line in, say, a Lanelet2 map.

Needs roadgen (``pip install 'carla-driver-interface[map]'``), imported only
when a map is written: a driver reading the files does not need it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import tempfile
from collections.abc import Iterable
from pathlib import Path

__all__ = ["MANIFEST_FILE", "export_map", "map_id_for", "sanitize_opendrive"]

logger = logging.getLogger(__name__)

#: The file listing what a map set holds; written last, so a set without it is
#: incomplete.
MANIFEST_FILE = "manifest.json"
SOURCE_FILE = "map.xodr"
IR_FILE = "map.ir.json"

#: Where each format is written, relative to the set's directory.
_FORMAT_PATHS = {
    "lanelet2": "lanelet2_map.osm",
    "opendrive": "roadgen.xodr",
    "osm": "openstreetmap.osm",
    "clipgt": "clipgt",
}

#: (pattern, replacement, why). roadgen reads OpenDRIVE 1.7 strictly, and the
#: RoadRunner exports behind CARLA's towns bend it in a few places; none of
#: these rewrites touch geometry.
_CARLA_FIXES: list[tuple[str, str, str]] = [
    (
        r"<userData>.*?</userData>|<userData\s*/>",
        "",
        "RoadRunner writes <userData><vectorScene/></userData>; 1.7 requires `code`",
    ),
    (
        r"<roadMark(?![^>]*\bcolor=)([^>]*)/>",
        r'<roadMark\1 color="standard"/>',
        "curb road marks carry no colour; `color` is required (default: standard)",
    ),
    (
        r'(<object\b[^>]*\btype=")-1(")',
        r"\1none\2",
        'objects with type="-1" are not a spec enum value',
    ),
    (
        r"<cornerLocal(?![^>]*\bheight=)([^>]*)/>",
        r'<cornerLocal\1 height="0"/>',
        "outline corners omit the required `height`",
    ),
    (
        r"<cornerRoad(?![^>]*\bheight=)([^>]*)/>",
        r'<cornerRoad\1 height="0"/>',
        "outline corners omit the required `height`",
    ),
]


def sanitize_opendrive(text: str) -> str:
    """CARLA's OpenDRIVE with the departures roadgen's strict reader rejects fixed."""
    for pattern, replacement, _why in _CARLA_FIXES:
        text = re.sub(pattern, replacement, text, flags=re.S)
    return text


def map_id_for(map_name: str, opendrive: str) -> str:
    """``<map name>-<first 12 hex digits of the OpenDRIVE's SHA-256>``.

    The digest makes the id name the map's content, so a driver holding an
    older copy of a map with the same name cannot mistake it for this one.
    """
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", map_name.rsplit("/", 1)[-1]) or "map"
    return f"{name}-{hashlib.sha256(opendrive.encode()).hexdigest()[:12]}"


def export_map(
    opendrive: str,
    map_name: str,
    map_dir: str | Path,
    formats: Iterable[str] = ("lanelet2",),
) -> str:
    """Write ``opendrive`` as ``formats`` under ``<map_dir>/<map_id>/``; returns the id.

    A complete set already holding every format is reused as it is. A set is
    built in a scratch directory beside its destination and moved into place
    whole, so a reader never sees one half written.
    """
    formats = tuple(dict.fromkeys(formats))
    unknown = sorted(set(formats) - set(_FORMAT_PATHS))
    if unknown:
        raise ValueError(f"unknown map format(s) {unknown}; known: {sorted(_FORMAT_PATHS)}")
    map_id = map_id_for(map_name, opendrive)
    root = Path(map_dir)
    target = root / map_id
    if _has_formats(target, formats):
        logger.info("map %s already written to %s", map_id, target)
        return map_id

    try:
        import roadgen
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise ImportError(
            "writing a map needs roadgen: pip install 'carla-driver-interface[map]'"
        ) from exc

    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root, prefix=f".{map_id}.") as scratch:
        out = Path(scratch)
        (out / SOURCE_FILE).write_text(opendrive, encoding="utf-8")
        sanitized = out / ".sanitized.xodr"
        sanitized.write_text(sanitize_opendrive(opendrive), encoding="utf-8")
        world_map = roadgen.read_opendrive(str(sanitized))
        sanitized.unlink()
        for warning in world_map.read_warnings():
            logger.info("roadgen read %s: %s", map_name, warning)

        world_map.export_ir(str(out / IR_FILE))
        written = {name: _export(world_map, name, out) for name in formats}
        manifest = {
            "map_id": map_id,
            "map_name": map_name,
            "roadgen_version": getattr(roadgen, "__version__", ""),
            "source": SOURCE_FILE,
            "ir": IR_FILE,
            "formats": written,
        }
        (out / MANIFEST_FILE).write_text(json.dumps(manifest, indent=1), encoding="utf-8")

        if target.exists():
            shutil.rmtree(target)
        out.rename(target)
        # The scratch directory is gone; recreate it so the context manager's
        # cleanup has something to remove.
        out.mkdir()
    logger.info("wrote map %s (%s) to %s", map_id, ", ".join(formats), target)
    return map_id


def _export(world_map, name: str, out: Path) -> dict[str, str]:
    """Write one format; returns its file and trace, relative to ``out``."""
    path = _FORMAT_PATHS[name]
    if name == "clipgt":
        clip_id = world_map.export_clipgt(str(out / path))
        return {"path": path, "trace": f"{path}/{clip_id}.clipgt.trace.json"}
    getattr(world_map, f"export_{name}")(str(out / path))
    return {"path": path, "trace": f"{path}.trace.json"}


def _has_formats(directory: Path, formats: tuple[str, ...]) -> bool:
    try:
        manifest = json.loads((directory / MANIFEST_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return set(formats) <= set(manifest.get("formats", {}))
