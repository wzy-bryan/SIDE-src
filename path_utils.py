import os
from pathlib import Path


def project_root() -> Path:
    """Return the SIDE project root.

    Priority:
    1. SIDE_ROOT environment variable
    2. parent directory of this file
    """
    return Path(os.environ.get("SIDE_ROOT", Path(__file__).resolve().parent)).resolve()


def data_root() -> Path:
    """Return the data root used by the original repo layout.

    The codebase historically referenced files under SIDE/acl_src and
    SIDE/icics_src_side. This helper keeps that structure but makes it
    relocatable.
    """
    return project_root().parent


def side_data_path(*parts: str) -> str:
    """Build a path under icics_src_side."""
    return str(project_root().joinpath(*parts))


def acl_data_path(*parts: str) -> str:
    """Build a path under SIDE/acl_src."""
    return str(data_root().joinpath("acl_src", *parts))


def ensure_parent(path: str) -> None:
    Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
