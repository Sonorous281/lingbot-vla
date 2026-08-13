#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""Public processor API with optional hardware integrations loaded lazily.

Inference users import core processor types from this package. Importing the
HIL processor eagerly also imports teleoperator and serial-hardware modules,
even though those modules are unrelated to policy inference.
"""

from importlib import import_module


_LAZY_EXPORTS = {
    "AddBatchDimensionProcessorStep": ("batch_processor", "AddBatchDimensionProcessorStep"),
    "batch_to_transition": ("converters", "batch_to_transition"),
    "create_transition": ("converters", "create_transition"),
    "transition_to_batch": ("converters", "transition_to_batch"),
    "EnvAction": ("core", "EnvAction"),
    "EnvTransition": ("core", "EnvTransition"),
    "PolicyAction": ("core", "PolicyAction"),
    "RobotAction": ("core", "RobotAction"),
    "RobotObservation": ("core", "RobotObservation"),
    "TransitionKey": ("core", "TransitionKey"),
    "MapDeltaActionToRobotActionStep": ("delta_action_processor", "MapDeltaActionToRobotActionStep"),
    "MapTensorToDeltaActionDictStep": ("delta_action_processor", "MapTensorToDeltaActionDictStep"),
    "DeviceProcessorStep": ("device_processor", "DeviceProcessorStep"),
    "make_default_processors": ("factory", "make_default_processors"),
    "make_default_robot_action_processor": ("factory", "make_default_robot_action_processor"),
    "make_default_robot_observation_processor": ("factory", "make_default_robot_observation_processor"),
    "make_default_teleop_action_processor": ("factory", "make_default_teleop_action_processor"),
    "Numpy2TorchActionProcessorStep": ("gym_action_processor", "Numpy2TorchActionProcessorStep"),
    "Torch2NumpyActionProcessorStep": ("gym_action_processor", "Torch2NumpyActionProcessorStep"),
    "AddTeleopActionAsComplimentaryDataStep": ("hil_processor", "AddTeleopActionAsComplimentaryDataStep"),
    "AddTeleopEventsAsInfoStep": ("hil_processor", "AddTeleopEventsAsInfoStep"),
    "GripperPenaltyProcessorStep": ("hil_processor", "GripperPenaltyProcessorStep"),
    "ImageCropResizeProcessorStep": ("hil_processor", "ImageCropResizeProcessorStep"),
    "InterventionActionProcessorStep": ("hil_processor", "InterventionActionProcessorStep"),
    "RewardClassifierProcessorStep": ("hil_processor", "RewardClassifierProcessorStep"),
    "TimeLimitProcessorStep": ("hil_processor", "TimeLimitProcessorStep"),
    "JointVelocityProcessorStep": ("joint_observations_processor", "JointVelocityProcessorStep"),
    "MotorCurrentProcessorStep": ("joint_observations_processor", "MotorCurrentProcessorStep"),
    "NormalizerProcessorStep": ("normalize_processor", "NormalizerProcessorStep"),
    "UnnormalizerProcessorStep": ("normalize_processor", "UnnormalizerProcessorStep"),
    "hotswap_stats": ("normalize_processor", "hotswap_stats"),
    "VanillaObservationProcessorStep": ("observation_processor", "VanillaObservationProcessorStep"),
    "ActionProcessorStep": ("pipeline", "ActionProcessorStep"),
    "ComplementaryDataProcessorStep": ("pipeline", "ComplementaryDataProcessorStep"),
    "DataProcessorPipeline": ("pipeline", "DataProcessorPipeline"),
    "DoneProcessorStep": ("pipeline", "DoneProcessorStep"),
    "IdentityProcessorStep": ("pipeline", "IdentityProcessorStep"),
    "InfoProcessorStep": ("pipeline", "InfoProcessorStep"),
    "ObservationProcessorStep": ("pipeline", "ObservationProcessorStep"),
    "PolicyActionProcessorStep": ("pipeline", "PolicyActionProcessorStep"),
    "PolicyProcessorPipeline": ("pipeline", "PolicyProcessorPipeline"),
    "ProcessorKwargs": ("pipeline", "ProcessorKwargs"),
    "ProcessorStep": ("pipeline", "ProcessorStep"),
    "ProcessorStepRegistry": ("pipeline", "ProcessorStepRegistry"),
    "RewardProcessorStep": ("pipeline", "RewardProcessorStep"),
    "RobotActionProcessorStep": ("pipeline", "RobotActionProcessorStep"),
    "RobotProcessorPipeline": ("pipeline", "RobotProcessorPipeline"),
    "TruncatedProcessorStep": ("pipeline", "TruncatedProcessorStep"),
    "PolicyActionToRobotActionProcessorStep": (
        "policy_robot_bridge",
        "PolicyActionToRobotActionProcessorStep",
    ),
    "RobotActionToPolicyActionProcessorStep": (
        "policy_robot_bridge",
        "RobotActionToPolicyActionProcessorStep",
    ),
    "RenameObservationsProcessorStep": ("rename_processor", "RenameObservationsProcessorStep"),
    "TokenizerProcessorStep": ("tokenizer_processor", "TokenizerProcessorStep"),
}


def __getattr__(name: str):
    try:
        module_name, attribute = _LAZY_EXPORTS[name]
    except KeyError as error:
        raise AttributeError(name) from error
    value = getattr(import_module(f"{__name__}.{module_name}"), attribute)
    globals()[name] = value
    return value


__all__ = list(_LAZY_EXPORTS)
