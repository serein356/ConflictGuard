"""Portable path resolution for benchmark images."""

from __future__ import annotations

import os
from pathlib import Path


def conflictgui_root() -> Path:
    """Return the configured benchmark root, defaulting to the current directory."""
    return Path(os.environ.get("CONFLICTGUI_ROOT", Path.cwd())).expanduser().resolve()


def normalize_image_path(image_path: str) -> str:
    """Resolve absolute paths and the relative paths used by CONFLICTGUI files."""
    if not image_path:
        return image_path

    android_root = os.environ.get("ANDROID_CONTROL_ROOT")
    if android_root and image_path.startswith("android_control_parsed/"):
        relative = image_path.removeprefix("android_control_parsed/")
        return str((Path(android_root).expanduser() / relative).resolve())

    path = Path(image_path).expanduser()
    if path.is_absolute():
        marker = "android_control_parsed"
        if android_root and marker in path.parts:
            marker_index = path.parts.index(marker)
            return str((Path(android_root).expanduser() / Path(*path.parts[marker_index + 1:])).resolve())

        configured_root = os.environ.get("CONFLICTGUI_ROOT")
        if configured_root and "CONFLICT" in path.parts:
            marker_index = path.parts.index("CONFLICT")
            return str((Path(configured_root).expanduser() / Path(*path.parts[marker_index + 1:])).resolve())
        return str(path)

    root = conflictgui_root()
    if image_path.startswith("../"):
        path = root / "data" / "rawdata" / image_path[3:]
    else:
        path = root / path
    return str(path.resolve())
