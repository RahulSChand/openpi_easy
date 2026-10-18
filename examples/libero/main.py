import collections
import dataclasses
import json
import logging
import math
import pathlib
import re

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_spatial"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task
    num_tasks: int = 0  # Number of tasks to evaluate; 0 evaluates the full suite
    initial_state_offset: int = 0  # First initial state to use (for partitioning evaluation workers)
    max_steps: int = 0  # Policy-controlled step limit; 0 uses the suite default

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero/videos"  # Path to save videos
    save_success_videos: bool = True
    save_state_traces: bool = True

    seed: int = 7  # Random Seed (for reproducibility)


def eval_libero(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    if args.num_tasks > 0:
        num_tasks_in_suite = min(args.num_tasks, num_tasks_in_suite)
    logging.info(f"Task suite: {args.task_suite_name}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name.startswith("libero_spatial"):
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name.startswith("libero_object"):
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name.startswith("libero_goal"):
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name.startswith("libero_10"):
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name.startswith("libero_90"):
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")
    if args.max_steps > 0:
        max_steps = args.max_steps
    logging.info(f"Policy-controlled step limit: {max_steps}")

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")

            # Reset environment
            env.reset()
            action_plan = collections.deque()

            # Set initial states
            initial_state_idx = args.initial_state_offset + episode_idx
            if initial_state_idx >= len(initial_states):
                raise ValueError(
                    f"Initial state index {initial_state_idx} is out of range for {len(initial_states)} states."
                )
            obs = env.set_init_state(initial_states[initial_state_idx])

            # Setup
            t = 0
            policy_steps = 0
            replay_images = []
            trace = _new_trace(env, obs, initial_states[initial_state_idx])
            success = False

            logging.info(f"Starting episode {task_episodes + 1}...")
            while t < max_steps + args.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < args.num_steps_wait:
                        _record_transition_start(trace, env, obs, LIBERO_DUMMY_ACTION, is_policy_step=False)
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        _record_transition_end(trace, reward, done)
                        t += 1
                        continue

                    # Get preprocessed image
                    # IMPORTANT: rotate 180 degrees to match train preprocessing
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )

                    # Save preprocessed image for replay video
                    replay_images.append(np.concatenate((img, wrist_img), axis=1))

                    if not action_plan:
                        # Finished executing previous action chunk -- compute new chunk
                        # Prepare observations dict
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": np.concatenate(
                                (
                                    obs["robot0_eef_pos"],
                                    _quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"],
                                )
                            ),
                            "prompt": str(task_description),
                        }

                        # Query model to get action
                        action_chunk = client.infer(element)["actions"]
                        assert len(action_chunk) >= args.replan_steps, (
                            f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        )
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()

                    # Execute action in environment
                    _record_transition_start(trace, env, obs, action, is_policy_step=True)
                    obs, reward, done, info = env.step(action.tolist())
                    _record_transition_end(trace, reward, done)
                    policy_steps += 1
                    if done:
                        success = True
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

            task_episodes += 1
            total_episodes += 1

            _record_final_state(trace, env, obs)
            suffix = "success" if success else "failure"
            task_segment = re.sub(r"[^a-zA-Z0-9]+", "_", task_description).strip("_").lower()
            stem = (
                f"rollout_{initial_state_idx:03d}_{task_segment}_"
                f"policy-step_{policy_steps:03d}_env-step_{len(trace['actions']):03d}_{suffix}"
            )
            output_dir = pathlib.Path(args.video_out_path)
            if replay_images and (args.save_success_videos or not success):
                imageio.mimwrite(output_dir / f"{stem}.mp4", [np.asarray(x) for x in replay_images], fps=10)
            if args.save_state_traces:
                _save_trace(
                    output_dir / f"{stem}.npz",
                    trace,
                    {
                        "task_suite_name": args.task_suite_name,
                        "task_id": task_id,
                        "task_description": task_description,
                        "task_problem_folder": task.problem_folder,
                        "task_bddl_file": task.bddl_file,
                        "initial_state_idx": initial_state_idx,
                        "seed": args.seed,
                        "success": success,
                        "policy_steps": policy_steps,
                        "environment_steps": len(trace["actions"]),
                        "num_steps_wait": args.num_steps_wait,
                        "max_policy_steps": max_steps,
                        "replan_steps": args.replan_steps,
                    },
                )

            # Log current results
            logging.info(f"Success: {success}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        # Log final results
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")
        env.close()

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _new_trace(env, obs, initial_state):
    return {
        "actions": [],
        "is_policy_step": [],
        "rewards": [],
        "dones": [],
        "sim_states": [],
        "body_positions": [],
        "body_quaternions": [],
        "contact_pairs": [],
        "observations": collections.defaultdict(list),
        "initial_state": np.asarray(initial_state),
        "body_names": [_safe_model_name(env.sim.model.body_id2name(i), "body", i) for i in range(env.sim.model.nbody)],
        "geom_names": [_safe_model_name(env.sim.model.geom_id2name(i), "geom", i) for i in range(env.sim.model.ngeom)],
    }


def _safe_model_name(name, prefix, index):
    return name if name is not None else f"{prefix}_{index}"


def _record_state(trace, env, obs):
    trace["sim_states"].append(np.asarray(env.sim.get_state().flatten()).copy())
    trace["body_positions"].append(np.asarray(env.sim.data.body_xpos).copy())
    trace["body_quaternions"].append(np.asarray(env.sim.data.body_xquat).copy())
    trace["contact_pairs"].append(
        np.asarray(
            [(env.sim.data.contact[i].geom1, env.sim.data.contact[i].geom2) for i in range(env.sim.data.ncon)],
            dtype=np.int32,
        ).reshape(-1, 2)
    )
    for key, value in obs.items():
        if key.endswith(("_image", "_depth")):
            continue
        array = np.asarray(value)
        if np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.bool_):
            trace["observations"][key].append(array.copy())


def _record_transition_start(trace, env, obs, action, *, is_policy_step):
    _record_state(trace, env, obs)
    trace["actions"].append(np.asarray(action, dtype=np.float32))
    trace["is_policy_step"].append(is_policy_step)


def _record_transition_end(trace, reward, done):
    trace["rewards"].append(reward)
    trace["dones"].append(done)


def _record_final_state(trace, env, obs):
    _record_state(trace, env, obs)


def _save_trace(path, trace, metadata):
    num_states = len(trace["sim_states"])
    max_contacts = max((len(pairs) for pairs in trace["contact_pairs"]), default=0)
    contact_pairs = np.full((num_states, max_contacts, 2), -1, dtype=np.int32)
    contact_counts = np.zeros(num_states, dtype=np.int32)
    for state_idx, pairs in enumerate(trace["contact_pairs"]):
        contact_counts[state_idx] = len(pairs)
        contact_pairs[state_idx, : len(pairs)] = pairs

    arrays = {
        "actions": np.stack(trace["actions"]),
        "is_policy_step": np.asarray(trace["is_policy_step"], dtype=np.bool_),
        "rewards": np.asarray(trace["rewards"], dtype=np.float32),
        "dones": np.asarray(trace["dones"], dtype=np.bool_),
        "sim_states": np.stack(trace["sim_states"]),
        "body_positions": np.stack(trace["body_positions"]),
        "body_quaternions": np.stack(trace["body_quaternions"]),
        "contact_pairs": contact_pairs,
        "contact_counts": contact_counts,
        "initial_state": trace["initial_state"],
        "body_names": np.asarray(trace["body_names"]),
        "geom_names": np.asarray(trace["geom_names"]),
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
    }
    for key, values in trace["observations"].items():
        try:
            arrays[f"observation__{key}"] = np.stack(values)
        except ValueError:
            logging.warning(f"Skipping variable-shaped observation field {key!r}")
    np.savez_compressed(path, **arrays)


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
