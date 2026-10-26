# Architecture

[Documentation guide](README.md) · [Storage round](storage-round.md) ·
[Operations](reproduction.md)

## Runtime ownership

One command owns a complete episode on the GPU host:
[`humanoid-agent`](../src/humanoid_agent/cli.py). The episode owner starts one
batched physics
worker and, for model episodes, one operator per robot. Physics owns each
native SONIC controller and SDK worker, plus a camera worker. Components talk
over local files, loopback transport and private Unix sockets on that host;
rsync moves code and completed evidence outside the control loop.

<!-- markdownlint-disable MD013 -->

<!-- markdownlint-disable MD013 -->

| Component | Responsibility | Implementation |
| --- | --- | --- |
| Episode parent | Freeze inputs and source, preflight, deadlines, cleanup and terminal records | [`group.py`](../src/humanoid_agent/runtime/group.py), [`episode.py`](../src/humanoid_agent/runtime/episode.py) |
| Physics owner | Admit commands, step Newton, publish controller sensors and score the task | [`shared.py`](../src/humanoid_agent/runtime/shared.py) |
| Native controller | Released SONIC inference, histories, planner and motor commands | [`controller.py`](../src/humanoid_agent/runtime/controller.py) |
| SDK bridge | Isolate vendor DDS serialization and exchange motor and sensor packets | [`sdk.py`](../src/humanoid_agent/runtime/sdk.py) |
| Camera worker | Render frozen pose snapshots without advancing task physics | [`camera.py`](../src/humanoid_agent/runtime/camera.py) |
| Model operator | Assemble allowlisted observations, call model APIs and publish finite targets | [`operator.py`](../src/humanoid_agent/runtime/operator.py) |
| Evaluator | Independent drawer, fall and return-home checks | [`evaluation.py`](../src/humanoid_agent/runtime/evaluation.py) |

<!-- markdownlint-enable MD013 -->

<!-- markdownlint-enable MD013 -->

Physics enqueues owned CPU snapshots and consumes validated messages through
bounded local queues. A physics-owned I/O process handles typed ZeroMQ
request/reply, Foxglove recording and camera-result copying. Operator commands
and observation requests use local IPC sockets serviced at up to 100 Hz,
independently of the evidence queue. Camera poses, readiness and capture
completion use bounded ZeroMQ push/pull messages that bind batch, environment,
episode and observation identities. Image and depth files are immutable
model-input artifacts, never completion signals. Queue saturation or worker
failure ends execution with an infrastructure error; evidence is never
silently dropped.

Sensor transfers, actuation, task checks and deadline checks stay in the
physics loop. Fields due together share one packed GPU readback into reused
pinned staging buffers, then are copied into independently owned CPU arrays,
so a later transfer cannot overwrite evidence; the cache lasts one physics
tick. Motor targets use persistent pinned host and device buffers, with changed
positions, velocities, torques and gains sharing one upload. The camera worker
finishes its batched image and geometry readbacks before PNG encoding.

Each SDK and controller pair runs in its own loopback-only network namespace.
Operators keep Internet access for model APIs. Render poses and evaluator state
never cross the model's observation boundary.

## Physics and robot definition

Newton GPU physics is a deliberate choice, motivated by the goal of more
accurate physical simulation; keep it when adopting libraries or refactoring.
This is a design objective, not a measured accuracy result.

[`environment.py`](../src/humanoid_agent/runtime/environment.py) derives its
simulation
configuration from Isaac Lab's `G1FlatEnvCfg` and selects `cuda:0`, 5 ms
physics steps, one solver substep, CUDA graphs and fixed contact capacity. It
runs its own thin episode loop, not the example's policy, rewards or
randomized resets. Storage scenes add the room, cabinets, lockers, passive task
drawers, drawer contact sensors and the flat ground collider described in
[Storage round](storage-round.md#scene-and-assets).

The deployed Isaac Lab checkout is the official
[`v3.0.0` release](https://github.com/isaac-sim/IsaacLab/releases/tag/v3.0.0),
commit `e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4`, with Newton 1.6.1 and
Warp 1.17.0; the installed packages are `isaaclab` 28.0.0, `isaaclab-newton`
9.1.4 and `isaaclab-ov` 5.1.5. The study environment was checked on October
22, 2026. Each episode retains the physics environment's lockfile; it
identifies
dependencies without proving them conflict-free.

[`robot.py`](../src/humanoid_agent/runtime/robot.py) reads the hash-pinned
SONIC robot
configuration and translates its actuator fields to Newton without importing
SONIC's own simulator.
[`canonical_robot.py`](../src/humanoid_agent/canonical_robot.py)
verifies the shared URDF and defines joint order and tracking-point offsets;
MuJoCo supplies encoder-only forward kinematics from that URDF without stepping
dynamics. Storage scenes enable the articulated Dex3 hands. The
[controller requirements](reproduction.md#controller-and-asset-requirements)
compares released and local
parameters.

## Native controller

The native deployment is pinned to SONIC revision
`45c03facbf848e0c743091b7d9c17952096ff908`; launch verifies source, executable
and deployment-asset hashes. SONIC owns control and planning, and the project
does not reimplement its learned policy. The policy runs at 50 Hz, local
physics at 200 Hz, and physics refreshes planner publications at about 30 Hz;
the SDK forwards the newest queued sample without a second planner timer. The
launcher applies the pinned README's tested v1.1 ankle-pitch gain scaling
(hardware indices 4 and 10, Kp and Kd × 1.5) without modifying the executable.
The [reproduction guide](reproduction.md) holds the
deployment and bridge checks.

The SDK worker processes at most 32 incoming packets per pass, coalescing
sensor and planner updates within that batch, and serializes motor receipts
only when the DDS source timestamp changes. Motor state occupies one locked,
memory-mapped slot shared with physics, so a backlog cannot hide a fresh
command. A busy read keeps the existing device targets for that tick; a reader
busy for one second fails. Planner publication receipts stay ordered,
validated and loss-intolerant.

## Controller checks

`native-controller-check`, or `--episode-kind controller-check`, runs the same
episode owner, physics loop, native processes, task measurements, deadlines and
records as a model episode, with a scripted source instead of an operator:
`ShutdownReferenceControl`
([`shutdown_reference.py`](../src/humanoid_agent/runtime/shutdown_reference.py)),
`StandingControlSource`
([`scripted_control.py`](../src/humanoid_agent/runtime/scripted_control.py)) or
`TurnCheckControl`
([`stepped_turn.py`](../src/humanoid_agent/runtime/stepped_turn.py)).
[`diagnostic.py`](../src/humanoid_agent/runtime/diagnostic.py) requires a
storage scene
and sets the deadline. Scripted episodes use the `sonic-native-planner-v2` wire
contract, reject model ingress and cannot count as model success. Camera-off
checks omit the render worker.
[Operations](reproduction.md#run-the-storage-round)
lists the commands.

## Shared groups and GPU scheduling

The default episode is the common runtime with one environment. The `group`
role gives the same owner one to sixteen explicit provider assignments, with
one Newton scene, one tiled camera worker and a separate native SONIC instance,
command ledger, evaluator, observation stream and terminal record per robot.
For more than one environment the owner runs NVIDIA's foreground MPS control
process with a private pipe and retained logs, stops CUDA clients before MPS
and rejects inherited MPS overrides. One environment keeps default driver
scheduling. The protocol records `execution_topology` and `gpu_scheduling`, so
the two cannot be pooled silently.

## Observations

The onboard RGB-D camera uses HomeBody's released torso-to-D435 mount (position
[0.0576, 0.0175, 0.4299] m, 47.6° downward pitch) at 640×480, from
[`robot_camera.py`](../src/humanoid_agent/robot_camera.py). Rendering
reconstructs the
measured root and joint state and composes that mount with the resulting torso
pose; requested head targets never set camera poses. The G1 has no actuated
neck, so the robot changes its view through locomotion and torso posture. In
the cabinet inspection task the fixed
external RGB camera is high on the west wall between C and D, looking east at
home; the lockers hide A and B and C and D sit behind it. Storage scenes have
no wrist cameras. Calibration is virtual, and ideal depth and IMUs do not
establish hardware realism.

The observation interface is `paired-action-history-initial-reference-v2`
([`observation_history.py`](../src/humanoid_agent/observation_history.py)).
Each request
carries synchronized onboard and external images at the previous action's
start, midpoint and current/end, plus the original `episode/start` pair as a
fixed visual home reference; the first request has one current pair. Identical
pairs are uploaded once with all their timestamps. Selected-pixel depth queries
read only the current onboard image and preserve invalid readings.

The prompt also carries the static facility map, encoder/FK head and wrist
poses in the heading-normalized pelvis frame, body tilt, observation age and
measured hand opening, and feedback on the previous actions comparing requested
endpoints with measured FK. Feedback covers the last three actions and the
model's own revisable plan; there is no retrieval or summary agent. FK excludes
room translation and heading, so neither is observable.

Rendering, image copying and API input persistence run in other processes while
physics continues. History captures do not wake the operator; only the
current/end pair publishes observation-ready. Every API attempt retains its
exact input, image bytes, raw response and timing. No simulator pose, drawer
position, private contact or evaluator flag enters model observations.

## Numerical action contract

The only model interface is `--mode hybrid`, schema `sonic-native-planner-v3`
([`sonic_model_command.py`](../src/humanoid_agent/sonic_model_command.py)).
Each chunk
gives four head and wrist position and unit-quaternion keyframes in the
heading-normalized pelvis frame (x forward, y left, z up, metres, wxyz), with a
walking flag, speed, movement direction, facing yaw and a locomotion posture
(`standing`, `squat`, `kneel_both`, `kneel_single`). The first keyframe must
agree with measured tracking values. The model chooses any positive duration
within its remaining budget, rounded to 20 ms control ticks; at admission the
command is capped at the remaining wall time, truncating rather than speeding
up motion. An active command cannot be interrupted by the model.

[`sonic_planner_input.py`](../src/humanoid_agent/sonic_planner_input.py)
interpolates
positions linearly and orientations by SciPy SLERP. Tracking stays relative to
the live pelvis heading; locomotion direction is anchored to the observation's
heading. With the scene's `leashed_facing`, the planner timeline holds the
commanded facing within 15° of the measured heading until it reaches the
requested facing. The optional stepped in-place turn tool (`stepped_turn_tool`)
is disabled in the storage scenes. After the horizon the last tracking pose is
held and locomotion returns to idle; a held reference does not prove the robot
is still. Native SONIC owns lower-body context, replanning and blending, and its
released builder owns wire encoding.

Commands are bound to an episode, source response and observation
([`command_admission.py`](../src/humanoid_agent/command_admission.py)).
Admission
rejects wrong episodes, stale observations, expired inputs and identifier
conflicts; accepted duplicates do not restart motion, and rejections keep their
reason. Schema and start-frame errors return correction feedback within the
original deadline. No target normalization, IK or autonomous skill replaces the
model's output. Each decision also carries the revisable subtask plan and the
completion request; the legacy optional `inspection_report` field stays in the
schema so requests match the recorded ones and is never scored.

### Dex3 hands

Each chunk may carry nullable `left_grasp` and `right_grasp` commands with
`action` (`open` or `close`), `start_seconds` and `transition_seconds`
([`hand_control.py`](../src/humanoid_agent/hand_control.py)). Null keeps the
previous
hand reference, which the SDK relay republishes explicitly. Targets are the
pinned PICO streamer's trigger presets, computed by importing SONIC's
`G1GripperInverseKinematicsSolver` from the configured checkout; no copy of the
solver or preset table is kept. The timed ramp from measured finger positions
is a local adaptation (`pico-trigger-timed-ramp-v1`), default 0.4 s and at
least 20 ms. The unchanged native Dex3 driver clamps each publication to
±0.25 rad of measured joints. The storage round needs no grasp: drawers are
closed by pushing.

## Native model APIs

[`task_model_client.py`](../src/humanoid_agent/task_model_client.py) maps
`sol` and
`astra` to OpenAI Responses, `opus` to Anthropic Messages and `gemini` to
Google Interactions through their first-party SDKs, with SDK retries disabled.
Native reasoning and tool-continuation blocks are preserved within depth-tool
rounds. [`model_contracts.py`](../src/humanoid_agent/model_contracts.py)
defines the
validated `ModelRequest` and normalized `TaskGeneration`; provider-native blocks
stay opaque so signatures and future fields survive recording. Responses are
retained before robot-domain validation, which can request a correction without
altering rejected values or extending the deadline.

Standard service tier and a 180-second request timeout are the defaults, and
each request is capped by the remaining episode deadline. There is no
automatic retry or tier fallback. Service tier and reasoning settings are part
of the experimental condition; [Research protocol](reproduction.md) defines how
operational outcomes are interpreted.

## Typed task evidence

[`task_outcome.py`](../src/humanoid_agent/task_outcome.py) keeps the completion
protocol `agent-declaration-then-independent-verification-v1`: success needs
both the model's declaration and the physical criterion. The recorded outcome
is a typed shutdown record tagged with its `task_protocol`
(`drawer-shutdown-v2` for the storage round), with per-drawer evidence;
unknown tags fail closed. Infrastructure outcomes stay separate;
[Records](records.md) owns the schemas.
