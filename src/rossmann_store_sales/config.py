"""Project configuration helpers."""

from __future__ import annotations

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_config_path() -> Path:
    return project_root() / "configs" / "project.toml"


def load_config(path: str | Path | None = None) -> dict:
    config_path = Path(path) if path else default_config_path()
    with config_path.open("rb") as fp:
        config = tomllib.load(fp)
    config["_project_root"] = str(project_root())
    return config


def resolve_project_path(config: dict, value: str | Path | None) -> Path:
    if value is None:
        return project_root()
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(config.get("_project_root", project_root())) / path
