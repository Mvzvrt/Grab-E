from __future__ import annotations

from pathlib import Path

_PACKAGE_DIR = Path(__file__).resolve().parent

for _built_pkg_dir in sorted(_PACKAGE_DIR.glob("build/lib*/mgc_core")):
    if _built_pkg_dir.is_dir():
        __path__.append(str(_built_pkg_dir))
