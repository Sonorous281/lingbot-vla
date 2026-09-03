# rlinf-lingbotvla runtime install.
#
# lerobot==0.4.2 is installed with --no-deps because its own pins
# (torch<2.8, datasets>=4, gymnasium>=1.1) conflict with the RoboTwin CUDA
# stack. The slim script reapplies the three load-bearing import overrides
# (shipped in patches/lerobot_slim/) that remove lerobot's hardware-dep import
# chain, so --no-deps is safe for inference. See patches/lerobot_slim/README.md.

git submodule update --init --recursive --remote
pip install -e .
pip install --no-deps "lerobot==0.4.2"
python scripts/apply_lerobot_slim.py
pip install -e ./lingbotvla/models/vla/vision_models/lingbot-depth/ --no-deps
pip install -e ./lingbotvla/models/vla/vision_models/MoGe/
pip install flash-attn==2.8.3 --no-build-isolation
