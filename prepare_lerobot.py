"""Prepare the pinned LeRobot runtime source for a release wheel."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path

LEROBOT_WHEEL_URL = (
    "https://files.pythonhosted.org/packages/d5/58/"
    "f5f9d6c0451df3e5cf9c484b1eb6a3abb93016677630a615b925c9da189a/"
    "lerobot-0.4.2-py3-none-any.whl"
)
LEROBOT_WHEEL_SHA256 = (
    "af4e5c709522c8022703e10431c14ae65b2e967e7ab608f51a07c03c38cefe04"
)
LEROBOT_VERSION = "0.4.2"


def _prepare_lerobot(wheel: Path, destination: Path) -> None:
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    if digest != LEROBOT_WHEEL_SHA256:
        raise RuntimeError(
            "LeRobot wheel hash mismatch: "
            f"expected {LEROBOT_WHEEL_SHA256}, got {digest}"
        )
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(destination)
    licenses = list(destination.glob("lerobot-0.4.2.dist-info/licenses/LICENSE"))
    if len(licenses) != 1:
        raise RuntimeError("LeRobot wheel does not contain its expected LICENSE")
    shutil.copyfile(licenses[0], destination / "LICENSE")
    shutil.copyfile(licenses[0], destination / "lerobot" / "LICENSE")
    shutil.rmtree(destination / "lerobot-0.4.2.dist-info")
    version_module = destination / "lerobot" / "__version__.py"
    if not version_module.is_file():
        raise RuntimeError("LeRobot wheel does not contain lerobot/__version__.py")
    version_module.write_text(
        '"""Version bundled by rlinf-lingbotvla."""\n\n'
        f'__version__ = "{LEROBOT_VERSION}"\n',
        encoding="utf-8",
    )


def main() -> None:
    """Download or reuse the exact LeRobot wheel and prepare ignored sources."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheel", type=Path)
    args = parser.parse_args()

    package_root = Path(__file__).resolve().parent
    vendor_root = package_root / "vendor" / "lerobot"
    if vendor_root.exists():
        shutil.rmtree(vendor_root)
    vendor_root.mkdir(parents=True)

    wheel = package_root / "vendor" / "lerobot.whl"
    try:
        if args.wheel is None:
            with urllib.request.urlopen(LEROBOT_WHEEL_URL) as response:
                wheel.write_bytes(response.read())
        else:
            shutil.copyfile(args.wheel.expanduser().resolve(), wheel)
        _prepare_lerobot(wheel, vendor_root)
    finally:
        wheel.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
