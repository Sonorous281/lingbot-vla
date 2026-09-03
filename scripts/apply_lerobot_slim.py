"""Guarded, idempotent whole-file slimming for the pinned ``lerobot==0.4.2``.

The official ``lerobot==0.4.2`` wheel eagerly imports a chain that pulls in
hardware/serial modules (``lerobot.policies.__init__`` -> ``groot`` ->
``pretrained`` -> ``lerobot.configs.train`` -> ``lerobot.envs`` ->
``lerobot.robots`` -> ``lerobot.motors.motors_bus`` -> ``import serial``).
Those hardware deps are not part of the RoboTwin VLA inference runtime, so the
eager import breaks ``import deploy.lingbot_vla_policy``.

Three files in the vendored LeRobot copy sidestep this by deferring imports
(``policies/__init__.py`` and ``processor/__init__.py`` use lazy ``__getattr__``
exports; ``policies/pretrained.py`` guards ``TrainPipelineConfig`` behind
``TYPE_CHECKING``). These are behaviour-neutral: they only change *when* the
existing symbols are imported, not what they are. The slim copies are byte
identical to the previously-vendored ``vendor/lerobot`` sources, which were in
turn the official ``0.4.2`` wheel plus exactly these three slimming edits.

This module applies those three files on top of an installed ``lerobot==0.4.2``
so the runtime can depend on the official wheel instead of a vendored copy. It
mirrors the design of ``rlinf_robotwin.patches`` (``rlinf-robotwin-patch``):
package located via ``importlib``, version checked, idempotent, reinstall-safe,
and version-drift-detecting rather than silently no-op.

Run via the ``apply-lerobot-slim`` console script (or ``python -m``) after
installing ``lerobot==0.4.2`` (use ``--no-deps``; lerobot 0.4.2's own deps
pin ``torch<2.8`` which conflicts with the RoboTwin CUDA stack).
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

__all__ = ["SlimReport", "apply_lerobot_slim", "main"]

# The three slim files, relative to both the patch source dir and the installed
# ``lerobot`` package root.
_SLIM_FILES: tuple[str, ...] = (
    "policies/__init__.py",
    "policies/pretrained.py",
    "processor/__init__.py",
)

_EXPECTED_LEROBOT_VERSION = "0.4.2"


@dataclass
class SlimReport:
    """Result of one slim-file application."""

    relpath: str
    target: str
    status: str  # "patched" | "already_patched" | "not_found" | "missing_package"
    detail: str = ""

    def as_line(self) -> str:
        return (
            f"[lerobot-slim] {self.relpath} -> {self.target}: {self.status}"
            + (f" - {self.detail}" if self.detail else "")
        )


def _slim_root() -> Path:
    """Resolve the shipped slim-file source directory.

    Layout: ``<repo>/scripts/apply_lerobot_slim.py`` and
    ``<repo>/patches/lerobot_slim/``. Resolved from ``__file__`` so it works
    for editable/source installs. (Wheel packaging of the slim files is handled
    separately when the runtime moves to a non-editable install.)
    """

    here = Path(__file__).resolve().parent
    candidate = here.parent / "patches" / "lerobot_slim"
    if candidate.is_dir():
        return candidate
    # Fallback: maybe installed alongside the lingbotvla package.
    raise FileNotFoundError(
        f"lerobot slim source dir not found at {candidate}. Run from a source "
        "checkout of rlinf-lingbotvla, or ship patches/lerobot_slim with the wheel."
    )


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _resolve_lerobot_dir() -> Path | None:
    spec = importlib.util.find_spec("lerobot")
    if spec is None or spec.origin is None:
        return None
    return Path(spec.origin).resolve().parent


def apply_lerobot_slim() -> list[SlimReport]:
    """Apply the three slim files onto the installed ``lerobot`` package."""

    reports: list[SlimReport] = []
    lerobot_dir = _resolve_lerobot_dir()
    if lerobot_dir is None:
        return [
            SlimReport(
                relpath="*",
                target="lerobot",
                status="missing_package",
                detail="lerobot not installed; install lerobot==0.4.2 first",
            )
        ]

    # Version guard: warn loudly on drift, but still attempt (the slim files may
    # apply cleanly to a nearby point release; if they don't, the not_found
    # status below surfaces it).
    try:
        version = getattr(importlib.import_module("lerobot"), "__version__", "?")
    except Exception as exc:  # noqa: BLE001
        version = f"import-error:{exc}"
    if version != _EXPECTED_LEROBOT_VERSION:
        print(
            f"[lerobot-slim] WARNING: lerobot __version__={version!r}, expected "
            f"{_EXPECTED_LEROBOT_VERSION!r}. Slimming targets 0.4.2 exactly.",
            file=sys.stderr,
        )

    slim_root = _slim_root()
    for relpath in _SLIM_FILES:
        target = lerobot_dir / relpath
        slim_src = slim_root / relpath
        if not slim_src.is_file():
            reports.append(
                SlimReport(
                    relpath=relpath,
                    target=str(slim_src),
                    status="not_found",
                    detail="slim source file missing from patch dir",
                )
            )
            continue
        slim_text = slim_src.read_text(encoding="utf-8")
        if not target.is_file():
            reports.append(
                SlimReport(
                    relpath=relpath,
                    target=str(target),
                    status="not_found",
                    detail="target file absent in installed lerobot (version drift)",
                )
            )
            continue
        current = target.read_text(encoding="utf-8")
        if _sha256(current) == _sha256(slim_text):
            reports.append(
                SlimReport(
                    relpath=relpath, target=str(target), status="already_patched"
                )
            )
            continue
        target.write_text(slim_text, encoding="utf-8")
        reports.append(
            SlimReport(relpath=relpath, target=str(target), status="patched")
        )
    return reports


def main() -> int:
    """Console-script entry: apply all slim files and print a report."""

    reports = apply_lerobot_slim()
    hard_failures = 0
    for report in reports:
        print(report.as_line())
        if report.status in {"not_found", "missing_package"}:
            hard_failures += 1
    if hard_failures:
        print(
            f"\n{hard_failures} slim target(s) missing. This usually means the "
            "pinned lerobot==0.4.2 changed. Inspect the source and update "
            "patches/lerobot_slim.",
            file=sys.stderr,
        )
    return 1 if hard_failures else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
