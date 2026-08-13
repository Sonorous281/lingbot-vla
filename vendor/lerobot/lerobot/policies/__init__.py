# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from importlib import import_module


_LAZY_EXPORTS = {
    "ACTConfig": ("lerobot.policies.act.configuration_act", "ACTConfig"),
    "DiffusionConfig": (
        "lerobot.policies.diffusion.configuration_diffusion",
        "DiffusionConfig",
    ),
    "GrootConfig": ("lerobot.policies.groot.configuration_groot", "GrootConfig"),
    "PI0Config": ("lerobot.policies.pi0.configuration_pi0", "PI0Config"),
    "PI05Config": ("lerobot.policies.pi05.configuration_pi05", "PI05Config"),
    "SmolVLAConfig": (
        "lerobot.policies.smolvla.configuration_smolvla",
        "SmolVLAConfig",
    ),
    "SmolVLANewLineProcessor": (
        "lerobot.policies.smolvla.processor_smolvla",
        "SmolVLANewLineProcessor",
    ),
    "TDMPCConfig": ("lerobot.policies.tdmpc.configuration_tdmpc", "TDMPCConfig"),
    "VQBeTConfig": ("lerobot.policies.vqbet.configuration_vqbet", "VQBeTConfig"),
}


def __getattr__(name: str):
    try:
        module_name, attribute = _LAZY_EXPORTS[name]
    except KeyError as error:
        raise AttributeError(name) from error
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value

__all__ = [
    "ACTConfig",
    "DiffusionConfig",
    "PI0Config",
    "PI05Config",
    "SmolVLAConfig",
    "TDMPCConfig",
    "VQBeTConfig",
    "GrootConfig",
    "SmolVLANewLineProcessor",
]
