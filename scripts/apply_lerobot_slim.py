"""Fail-closed, idempotent whole-file slimming for the pinned ``lerobot==0.4.2``.

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

Fail-closed contract
--------------------
The applier never silently overwrites an unrecognised file. For each target it
takes a three-way decision keyed on the file's sha256:

* matches the **patched** hash  -> ``already_patched`` (idempotent, no write)
* matches the **official 0.4.2** hash -> write the slim copy -> ``patched``
* anything else -> ``unexpected_hash`` (NO write; aborts with a non-zero exit)

The patched hash is also asserted against the shipped slim source file itself
(``patch_source_drift`` if it differs), so an accidental edit to
``patches/lerobot_slim/`` fails loudly instead of installing a different patch.
A version guard short-circuits the whole run to ``version_mismatch`` (no writes)
when ``lerobot.__version__ != "0.4.2"``.

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

# sha256 of each file as it ships in the official ``lerobot==0.4.2`` wheel
# (``lerobot-0.4.2-py3-none-any.whl``, wheel sha256
# ``af4e5c709522c8022703e10431c14ae65b2e967e7ab608f51a07c03c38cefe04``).
# A target matching this is the un-slimmed official source -> safe to patch.
_OFFICIAL_SHA: dict[str, str] = {
    "policies/__init__.py": (
        "1c7cb025cdb5e524da8b4fa2b633d502e2180b50d7ce535d8001853e6464329e"
    ),
    "policies/pretrained.py": (
        "4d5a42bd475c7beca2ec899b7fcf7a0d890edc4234d143e4397c9999edafc39e"
    ),
    "processor/__init__.py": (
        "10c1151144d60b32d7eff191931a7dc1f0057d00335131f13b664861fc3562bc"
    ),
}

# sha256 of each file *after* slimming (i.e. of the shipped slim copy in
# ``patches/lerobot_slim/``). A target matching this is already slimmed ->
# idempotent no-op. The slim source file itself must also hash to this value
# (else the patch source has drifted -> ``patch_source_drift``).
_PATCHED_SHA: dict[str, str] = {
    "policies/__init__.py": (
        "70ecece820102b15111588049ac4879abecd4575436ed4f42980c21193494cf4"
    ),
    "policies/pretrained.py": (
        "f3270d5b5b111e2c47c7462d3e26240cd8c285053d48702ea35f9f84cd414639"
    ),
    "processor/__init__.py": (
        "c9f75e5ba22d2a67e4965ae94fe0a94591f55f2936f43db669632295bc2e2729"
    ),
}

# Statuses that indicate the run failed and must not be treated as success.
_FAILURE_STATUSES = frozenset(
    {"missing_package", "version_mismatch", "not_found", "unexpected_hash", "patch_source_drift"}
)


@dataclass
class SlimReport:
    """Result of one slim-file application."""

    relpath: str
    target: str
    status: str  # see _FAILURE_STATUSES + {"patched", "already_patched"}
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
    """Apply the three slim files onto the installed ``lerobot`` package.

    Fail-closed: an unrecognised target hash or a version mismatch is reported
    and NO file is overwritten. Only a target whose hash matches the official
    ``0.4.2`` source is patched; a target already matching the patched hash is
    left untouched (idempotent).
    """

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

    # Version guard: fail-closed. Slimming is only defined for the exact 0.4.2
    # source these hashes were computed from; any other version is left as-is.
    try:
        version = getattr(importlib.import_module("lerobot"), "__version__", "?")
    except Exception as exc:  # noqa: BLE001
        version = f"import-error:{exc}"
    if version != _EXPECTED_LEROBOT_VERSION:
        for relpath in _SLIM_FILES:
            reports.append(
                SlimReport(
                    relpath=relpath,
                    target=str(lerobot_dir / relpath),
                    status="version_mismatch",
                    detail=(
                        f"lerobot __version__={version!r}, expected "
                        f"{_EXPECTED_LEROBOT_VERSION!r}; no files written"
                    ),
                )
            )
        return reports

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
        # The shipped slim source must hash to the known patched hash; if it
        # does not, the patch itself has drifted -> fail rather than install an
        # unreviewed patch.
        if _sha256(slim_text) != _PATCHED_SHA[relpath]:
            reports.append(
                SlimReport(
                    relpath=relpath,
                    target=str(slim_src),
                    status="patch_source_drift",
                    detail=(
                        "slim source sha256 does not match the pinned patched "
                        f"hash {_PATCHED_SHA[relpath]}; inspect patches/lerobot_slim"
                    ),
                )
            )
            continue
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
        current_sha = _sha256(current)
        if current_sha == _PATCHED_SHA[relpath]:
            reports.append(
                SlimReport(
                    relpath=relpath, target=str(target), status="already_patched"
                )
            )
            continue
        if current_sha == _OFFICIAL_SHA[relpath]:
            target.write_text(slim_text, encoding="utf-8")
            reports.append(
                SlimReport(relpath=relpath, target=str(target), status="patched")
            )
            continue
        # Unrecognised content: do NOT overwrite. Fail-closed.
        reports.append(
            SlimReport(
                relpath=relpath,
                target=str(target),
                status="unexpected_hash",
                detail=(
                    f"target sha256={current_sha} matches neither official "
                    f"({_OFFICIAL_SHA[relpath]}) nor patched "
                    f"({_PATCHED_SHA[relpath]}); file left unmodified"
                ),
            )
        )
    return reports


def main() -> int:
    """Console-script entry: apply all slim files and print a report."""

    reports = apply_lerobot_slim()
    hard_failures = 0
    for report in reports:
        print(report.as_line())
        if report.status in _FAILURE_STATUSES:
            hard_failures += 1
    if hard_failures:
        print(
            f"\n{hard_failures} slim target(s) failed (fail-closed: no unrecognised "
            "file was overwritten). This usually means the pinned lerobot==0.4.2 "
            "changed or patches/lerobot_slim drifted. Inspect the source and update "
            "patches/lerobot_slim.",
            file=sys.stderr,
        )
    return 1 if hard_failures else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
