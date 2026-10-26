"""GPU-host model operator using immutable camera bundles and typed ZeroMQ requests.

Only allowlisted observations cross into model requests. The controller accepts
finite numerical inputs while physics continues in its own process.
"""

from __future__ import annotations

import base64
import hashlib
import json
import shutil
import time
from functools import partial
from pathlib import Path
from threading import Event, Timer
from typing import Any, Literal

import numpy as np

from humanoid_agent.artifact_io import atomic_json
from humanoid_agent.canonical_robot import JOINT_NAMES, canonical_urdf_path
from humanoid_agent.dex3 import hand_joint_limits
from humanoid_agent.facility_map import FACILITY_MAP_DESCRIPTION
from humanoid_agent.hand_control import HAND_CONTROL_PROTOCOL, PICO_STREAMER_SHA256, SOLVER_SHA256
from humanoid_agent.model_contracts import ModelRequest
from humanoid_agent.model_decision_validation import validate_initial_isaac_targets
from humanoid_agent.observation_history import (
    OBSERVATION_INTERFACE_PROTOCOL,
    RECENT_ACTION_HISTORY_LENGTH,
    collect_visual_history,
    image_pair_content,
)
from humanoid_agent.posture_control import (
    TURN_BUDGET_SECONDS_PER_STEP,
    PlannedSteppedTurnTeleopIsaacChunk,
    PlannedTeleopIsaacChunk,
)
from humanoid_agent.robot_feedback import (
    concise_robot_feedback,
    model_observation_payload,
    previous_action_feedback,
)
from humanoid_agent.runtime.observations import IsaacObservation
from humanoid_agent.runtime.operator_transport import (
    OperatorClient,
    OperatorState,
    QueryOperatorState,
    RequestObservation,
    SubmitCommand,
)
from humanoid_agent.runtime.timing import CommandTimingEvidence
from humanoid_agent.sensor_feedback import FRAME_DESCRIPTION, DepthSnapshot, EncoderKinematics
from humanoid_agent.sensor_tool_client import SensorRequestStopped, request_with_sensor_tools
from humanoid_agent.sonic_model_command import prepare_planner_publication
from humanoid_agent.sonic_planner_input import MOTION_PUBLICATION_ADAPTER, SonicMotionPublication
from humanoid_agent.task_model_client import TaskModelClient
from humanoid_agent.task_planning import TaskPlan


class LocalOperatorConnection:
    """Use ZeroMQ for live coordination and copy immutable camera artifacts locally."""

    def __init__(self, output: Path, commands: Path) -> None:
        self.output = output
        self.commands = commands
        self.client = OperatorClient(commands)
        self.state = OperatorState()

    def refresh(self, deadline: float | None = None) -> bool:
        state = self.client.request(
            QueryOperatorState(), time.monotonic() + 0.2 if deadline is None else deadline
        )
        if state is None:
            return False
        self.state = state
        return True

    def fetch(self, source: Path, destination: Path, *, directory: bool = False) -> None:
        if directory:
            shutil.copytree(source, destination)
        else:
            shutil.copyfile(source, destination)

    def publish_command(
        self, publication: SonicMotionPublication, timing: CommandTimingEvidence | None = None
    ) -> None:
        self._request(SubmitCommand(publication=publication, timing=timing))

    def request_observation(self, identifier: str) -> None:
        self._request(RequestObservation(identifier=identifier))

    def _request(self, request: SubmitCommand | RequestObservation) -> None:
        deadline = (
            self.state.clock.deadline_monotonic_seconds
            if self.state.clock
            else time.monotonic() + 900
        )
        state = self.client.request(request, deadline)
        if state is None:
            raise TimeoutError("Operator request was not acknowledged before the episode deadline")
        self.state = state

    def wait_identifier(
        self, kind: Literal["motion", "observation"], identifier: str, deadline: float
    ) -> bool:
        while time.monotonic() < deadline:
            if not self.refresh(min(deadline, time.monotonic() + 0.2)):
                continue
            if self.state.finished:
                return False
            if kind == "motion" and self.state.command_status is not None:
                status = self.state.command_status
                if status.identifier == identifier and status.state == "rejected":
                    raise ValueError(
                        f"Simulator rejected command {identifier}: {status.admission.reason}"
                    )
            receipt = (
                self.state.motion_finished if kind == "motion" else self.state.observation_ready
            )
            if receipt is not None and receipt.identifier == identifier:
                return True
            time.sleep(0.02)
        return False


def facility_map_request_content(facility_map: Path) -> list[dict[str, str]]:
    """The same frozen map bytes accompany every request in an episode."""
    encoded = base64.b64encode(facility_map.read_bytes()).decode()
    return [
        {"type": "input_text", "text": json.dumps({"static_facility_map": "image follows"})},
        {"type": "input_image", "image_url": f"data:image/png;base64,{encoded}"},
    ]


def run_isaac_model_operator(arguments: Any) -> None:
    arguments.output.mkdir(parents=True, exist_ok=False)
    # Match the physics owner's 900-second startup watchdog.
    # Simulator installation/loading is not charged to model inference time.
    deadline = time.monotonic() + 900
    stop_requested = Event()
    deadline_timer: Timer | None = None
    history: list[dict[str, Any]] = []
    high_level_plan: dict[str, Any] | None = None
    action_count = 0
    identifier = "initial"
    try:
        if not arguments.external_camera:
            raise ValueError(
                "The current observation interface requires paired onboard/external RGB"
            )
        decision_schema = (
            PlannedSteppedTurnTeleopIsaacChunk
            if arguments.stepped_turns
            else PlannedTeleopIsaacChunk
        )
        if arguments.physics_directory is None or arguments.command_directory is None:
            raise ValueError("The operator requires local physics and command directories")
        if arguments.task_instruction is None:
            raise ValueError("The operator requires the episode owner's task instruction")
        files = LocalOperatorConnection(arguments.physics_directory, arguments.command_directory)
        facility_map_content = (
            facility_map_request_content(arguments.facility_map)
            if arguments.facility_map is not None
            else []
        )
        client = TaskModelClient(
            arguments.provider,
            service_tier=arguments.api_service_tier,
            reasoning_effort=arguments.reasoning_effort,
            request_timeout_seconds=arguments.api_request_timeout_seconds,
        )
        encoder = EncoderKinematics()
        atomic_json(
            arguments.output / "operator-configuration.json",
            {
                "model": client.model,
                "reasoning_effort": client.reasoning_effort,
                "max_output_tokens": arguments.model_max_output_tokens,
                "api_path": client.api_path,
                "recorded_retry_policy": arguments.api_retry_policy.value,
                **client.inference_configuration(),
                "validation_feedback_protocol": "schema-and-start-frame-correction-v1",
                "invalid_motion_handling": "Reject, record and return validation feedback within the original deadline; no automatic normalization",
                "physics_during_inference": True,
                "observations": "onboard RGB-D, named encoders, IMU gravity/gyro, robot-only FK"
                + (", fixed external RGB camera" if arguments.external_camera else ""),
                "task_world_state_supplied": False,
                "walking_endpoint": "native planner idle after finite command expiry",
                "task_instruction": arguments.task_instruction,
                "facility_map_sha256": hashlib.sha256(
                    arguments.facility_map.read_bytes()
                ).hexdigest()
                if arguments.facility_map is not None
                else None,
                "motion_mode": arguments.mode,
                "hand_control_protocol": HAND_CONTROL_PROTOCOL,
                "hand_mapping_solver_sha256": SOLVER_SHA256,
                "hand_mapping_streamer_sha256": PICO_STREAMER_SHA256,
                "orientation_control": "four numerical head/wrist quaternion keyframes; wxyz; pelvis heading frame",
                "physics_directory": str(arguments.physics_directory),
            },
        )
        if not files.wait_identifier("observation", "initial", deadline):
            return
        clock_path = arguments.output / "episode-clock.json"
        episode_clock = files.state.clock
        if episode_clock is None:
            raise ValueError("Initial camera acknowledgment lacks its episode clock")
        atomic_json(clock_path, episode_clock.model_dump(mode="json"))
        deadline = episode_clock.deadline_monotonic_seconds
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        operator_budget = remaining
        deadline_timer = Timer(operator_budget, stop_requested.set)
        deadline_timer.daemon = True
        deadline_timer.start()
        atomic_json(
            arguments.output / "operator-startup.json",
            {
                "episode_id": episode_clock.episode_id,
                "sampled_simulator_elapsed_seconds": episode_clock.elapsed_wall_seconds,
                "operator_budget_seconds": operator_budget,
                "deadline_monotonic_seconds": deadline,
                "clock_sampling": "Absolute physics deadline on the shared host monotonic clock",
            },
        )
        while time.monotonic() < deadline:
            if not files.wait_identifier("observation", identifier, deadline):
                break
            destination = arguments.output / identifier
            files.fetch(files.output / "observations" / identifier, destination, directory=True)
            record = IsaacObservation.parse_requested_snapshot(
                (destination / "observation.json").read_text(),
                identifier=identifier,
                clock=episode_clock,
            )
            observation_identity = record.observation_identity
            received_at = time.monotonic()
            sensors = record.robot_sensors.in_joint_order(JOINT_NAMES)
            record.robot_sensors = sensors
            kinematics = encoder.measure(sensors)
            snapshot = DepthSnapshot(
                np.load(destination / "depth.npy"),
                kinematics,
                record.simulation_seconds,
                record.elapsed_wall_seconds,
                optical_points_m=np.load(destination / "optical-points-m.npy"),
            )
            pairs = collect_visual_history(
                destination, files.output / "observations", include_episode_start=True
            )
            prompt_generated_elapsed_seconds = time.monotonic() - (
                episode_clock.deadline_monotonic_seconds
                - episode_clock.deadline_elapsed_wall_seconds
            )
            atomic_json(
                destination / "visual-history.json",
                {
                    "protocol": OBSERVATION_INTERFACE_PROTOCOL,
                    "prompt_generated_elapsed_seconds": prompt_generated_elapsed_seconds,
                    "pairs": [pair.model_dump(mode="json") for pair in pairs],
                },
            )
            atomic_json(destination / "kinematic-feedback.json", kinematics)
            feedback = concise_robot_feedback(record, kinematics, prompt_generated_elapsed_seconds)
            if record.action_frame is not None:
                if (
                    not history
                    or history[-1]["action_id"] != record.action_frame.execution.action_id
                ):
                    raise ValueError("Previous-action feedback differs from recent action history")
                previous_command = MOTION_PUBLICATION_ADAPTER.validate_json(
                    (
                        arguments.output / record.action_frame.execution.action_id / "command.json"
                    ).read_text()
                )
                history[-1]["feedback"] = previous_action_feedback(
                    previous_command, record.action_frame.execution, record, kinematics
                )
            hand_limits = (
                hand_joint_limits(canonical_urdf_path())
                if record.hand_joint_sensors is not None
                else None
            )
            prompt = {
                "task": arguments.task_instruction,
                **(
                    {"facility_map": FACILITY_MAP_DESCRIPTION}
                    if arguments.facility_map is not None
                    else {}
                ),
                "episode_start_images": "Images labelled episode/start are unchanged camera observations of your initial location. Use them as a fixed visual reference when returning. They are historical RGB, not current sensors or localization; depth queries use the current/end onboard image only.",
                "control": {
                    "interface": "You are the teleoperator. Emit numerical head/hand keyframes and released SONIC walking/posture commands. No autonomous task skills exist.",
                    "frame": FRAME_DESCRIPTION,
                    "first_keyframe": "Copy observation.robot_feedback.copy_current_pose.tracking_points_m exactly into the first pose. Four poses interpolate over duration_seconds; x forward, y left, z up.",
                    "orientations": "Emit four orientations alongside the four poses. Each has left_wrist, right_wrist and head unit quaternions in [w,x,y,z] order, expressing body-to-pelvis-heading rotations. Copy observation.robot_feedback.copy_current_pose.tracking_orientations_wxyz into the first orientation. Rotations interpolate by shortest-path SLERP, then hold the last value. Identity points the wrist/body local axes along heading axes. No automatic aiming or IK is supplied.",
                    "whole_body_coupling": "SONIC tracks the body targets jointly. Changing head/hand targets can move the base even without a walking request or while an endpoint is held. A held reference is not a guarantee that the robot has stopped. Identity rotations are not necessarily a neutral pose; use the measured orientations as your reference and choose consistent position/orientation targets.",
                    "camera_motion": "Actively improve your view with the head position AND head orientation keyframes alongside wrists and locomotion. The G1 has no actuated neck: the head target is torso_link at offset [0,0,0.35] metres, with torso orientation. SONIC must physically lean, lower or turn the robot. Positive rotation about heading +y pitches forward/down (wxyz quaternion [cos(a/2),0,sin(a/2),0]). Move the head point consistently with torso lean; coordinate wrist targets and squat/kneel posture as needed. The camera is rigidly mounted at the calibrated torso transform and renders measured motion only, never requested targets. Inspect measured head orientation, camera optical axis and the next image to check the achieved view.",
                    "walking": "movement_direction and facing_yaw_radians are relative to the current heading. Only standing mode permits walking. A native stop transition follows finite walking commands. The pinned upstream ROS2 slow-walk profile uses nominal 0.3 m/s forward and 0.2 m/s backward. Commissioning in this setup found very little forward translation at 0.15 m/s even over five seconds. Prefer the upstream nominal speeds for forward/backward movement and verify progress in images; requested speed times duration is not a guaranteed displacement. No speed is silently clipped or replaced.",
                    "timing": "Choose any positive duration_seconds within the remaining episode budget shown by the observation clocks. Duration rounds to the nearest 0.02-second controller tick, with a minimum of one tick. Four keyframes span that duration. The episode deadline can truncate motion without speeding up keyframes. You receive new feedback and think again after execution; you cannot interrupt an active motion. Physics continues while you think; feedback is a frozen snapshot. Request completion only on your final motion; evaluation follows execution.",
                    **(
                        {
                            "turning": f"facing_yaw_radians is limited to ±45° (±0.785 rad); larger facing changes made the released planner spin and fall. To turn further, set turn_in_place_degrees to one of ±45, ±90, ±135 or ±180 with walking false, standing posture and facing_yaw_radians 0; otherwise set it to 0. The robot then turns in place in released-planner facing steps of at most 45°, each held until the measured heading settles, starting from the heading measured when the command activates. A turn command lasts up to {TURN_BUDGET_SECONDS_PER_STEP:g} s per 45° step plus 1 s regardless of duration_seconds (measured: about 8-11 s for 90°, 15 s for 180°); keep head and hand keyframes near the current pose. Verify the achieved heading in the next images.",
                        }
                        if arguments.stepped_turns
                        else {
                            "turning": "Large facing changes are approached gradually: the controller keeps the commanded facing within 15° of the measured heading until it reaches the requested facing_yaw_radians, as the released gamepad's stick turning does. A 90° change takes several seconds and continues through the held endpoint after the motion ends. Verify the achieved heading in the next images.",
                        }
                        if arguments.leashed_facing
                        else {}
                    ),
                    "postures": "standing, squat, kneel_both, kneel_single; body_height_m controls static posture via the released planner",
                    "fingers": {
                        "available": record.hand_joint_sensors is not None,
                        "commands": "Use left_grasp/right_grasp, each null or {action: open|close, start_seconds, transition_seconds}. These are the upstream Pico binary trigger's open/middle-close preset; do not output individual finger angles. start_seconds is relative to motion activation, independent of wrist/body keyframe timing. Delay close until your planned approach is complete and choose an explicit transition of at least 0.02 s (default 0.4 s); the full transition must fit duration_seconds. Our time ramp starts from measured fingers at admission; before the scheduled transition it holds that measured pose, matching a native hold. Null preserves the previous native hand target. Native measured-state smoothing remains active. You control placement, closure, lift, hold and release timing; no object targeting, attachment, automatic contact gate or grasp planner is supplied.",
                    },
                },
                "observation_interface_protocol": OBSERVATION_INTERFACE_PROTOCOL,
                "observation": model_observation_payload(feedback, pairs),
                "recent_action_history_length": RECENT_ACTION_HISTORY_LENGTH,
                "plan": high_level_plan,
                "recent_decisions": history,
                "request": "First make a high-level plan."
                if high_level_plan is None
                else "Update the subtask list and emit your next numerical motion chunk.",
            }
            atomic_json(destination / "prompt.json", prompt)
            request_started_at = time.monotonic()
            response = request_with_sensor_tools(
                client,
                destination / "model.response.json",
                snapshot,
                stop_requested,
                ModelRequest.model_validate(
                    {
                        "input": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "input_text", "text": json.dumps(prompt)},
                                    *facility_map_content,
                                    *image_pair_content(pairs, destination),
                                ],
                            }
                        ],
                        "text_format": TaskPlan if high_level_plan is None else decision_schema,
                        "max_output_tokens": arguments.model_max_output_tokens,
                    }
                ),
                api_retry_policy=arguments.api_retry_policy,
                validate_output=partial(
                    validate_initial_isaac_targets,
                    kinematics=kinematics,
                    hand_limits=hand_limits,
                    maximum_duration_seconds=(
                        observation_identity.episode_deadline_elapsed_wall_seconds
                        - observation_identity.captured_elapsed_wall_seconds
                    ),
                ),
                deadline_monotonic_seconds=deadline,
            )
            response_received_at = time.monotonic()
            atomic_json(
                destination / "model-timing.json",
                {
                    "observation_id": observation_identity.identifier,
                    "episode_id": observation_identity.episode_id,
                    "model_request_wall_seconds": response_received_at - request_started_at,
                    "local_snapshot_receive_to_response_seconds": response_received_at
                    - received_at,
                    "snapshot_capture_to_receive_seconds": received_at
                    - (
                        episode_clock.deadline_monotonic_seconds
                        - observation_identity.episode_deadline_elapsed_wall_seconds
                        + observation_identity.captured_elapsed_wall_seconds
                    ),
                    "observation_received_monotonic_seconds": received_at,
                    "request_started_monotonic_seconds": request_started_at,
                    "response_received_monotonic_seconds": response_received_at,
                },
            )
            response_path = destination / "model.response.json"
            atomic_json(response_path, response.model_dump())
            if response.output_parsed is None:
                raise ValueError("Model returned no parsed operator decision")
            if high_level_plan is None:
                high_level_plan = response.output_parsed.model_dump()
                atomic_json(arguments.output / "high-level-plan.json", high_level_plan)
            else:
                decision = decision_schema.model_validate(response.output_parsed.model_dump())
                history.append(
                    {
                        "action_id": identifier,
                        "duration_seconds": decision.duration_seconds,
                        "active_subtask_id": decision.active_subtask_id,
                        "observed_progress": decision.observed_progress,
                        "feedback": None,
                    }
                )
                history[:] = history[-RECENT_ACTION_HISTORY_LENGTH:]
                action_count += 1
                high_level_plan = {
                    "subtasks": [subtask.model_dump() for subtask in decision.subtasks]
                }
                publication = prepare_planner_publication(
                    decision,
                    record.observation_identity,
                    kinematics,
                    identifier,
                    hashlib.sha256(response_path.read_bytes()).hexdigest(),
                    hand_limits=hand_limits,
                    leashed_facing=arguments.leashed_facing,
                )
                command_path = destination / "command.json"
                atomic_json(command_path, publication.model_dump())
                if time.monotonic() >= deadline:
                    break
                timing_evidence = CommandTimingEvidence(
                    identifier=publication.identifier,
                    episode_id=publication.episode_id,
                    observation_id=publication.observation_id,
                    source_response_sha256=publication.source_response_sha256,
                    observation_received_monotonic_seconds=received_at,
                    request_started_monotonic_seconds=request_started_at,
                    response_received_monotonic_seconds=response_received_at,
                    publication_prepared_monotonic_seconds=time.monotonic(),
                )
                timing_path = destination / "command-timing.json"
                atomic_json(timing_path, timing_evidence.model_dump())
                files.publish_command(publication, timing_evidence)
                if not files.wait_identifier("motion", identifier, deadline):
                    break
                if decision.request_task_completion:
                    break
            identifier = f"feedback_{action_count:03d}"
            request_path = arguments.output / "observation-request.json"
            atomic_json(request_path, {"identifier": identifier})
            files.request_observation(identifier)
    except SensorRequestStopped:
        atomic_json(
            arguments.output / "operator-stop.json",
            {
                "reason": "episode_budget_exhausted_between_model_requests",
                "recorded_by": "operator",
            },
        )
    except BaseException as error:
        atomic_json(
            arguments.output / "operator-error.json",
            {
                "exception_type": type(error).__name__,
                "message": str(error),
                "recorded_by": "operator",
            },
        )
        raise
    finally:
        if deadline_timer is not None:
            deadline_timer.cancel()
        atomic_json(
            arguments.output / "operator-summary.json",
            {
                "motion_decisions": action_count,
                "latest_plan": high_level_plan,
                "task_success": "See independent physics/task-outcome.json; no success inferred here",
            },
        )
