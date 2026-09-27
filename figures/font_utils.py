from __future__ import annotations

import os
import warnings
from pathlib import Path

from matplotlib import font_manager


def publication_serif(root: Path) -> str:
    candidates = []
    if os.environ.get("GTRMDA_FONT_DIR"):
        candidates.append(Path(os.environ["GTRMDA_FONT_DIR"]))
    candidates.extend([
        root / ".fonts" / "tnr",
        root.parent / ".fonts" / "tnr",
        root.parent.parent / ".fonts" / "tnr",
    ])
    for directory in candidates:
        if directory.exists():
            for font_path in sorted(directory.iterdir()):
                if font_path.suffix.lower() in {".ttf", ".otf"}:
                    font_manager.fontManager.addfont(font_path)
    registered = {font.name for font in font_manager.fontManager.ttflist}
    if "Times New Roman" in registered:
        return "Times New Roman"
    warnings.warn(
        "Times New Roman is unavailable; using Liberation Serif. Set "
        "GTRMDA_FONT_DIR to a directory containing legally obtained font files.",
        stacklevel=2,
    )
    return "Liberation Serif"

