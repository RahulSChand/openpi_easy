"""Smoke-test a privileged deterministic recovery controller in LIBERO.

This targets LIBERO-Object task 0. It restores an official initial state,
displaces the target object to create a recoverable failure state, saves and
restores that exact MuJoCo state, and then runs a closed-loop OSC controller.
No images or learned policy are used by the controller.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from libero.libero import benchmark, get_libero_path
from libero.libero.envs.env_wrapper import ControlEnv


OPEN_GRIPPER = -1.0
CLOSE_GRIPPER = 1.0
FAILURE_OFFSETS = ((0.06, 0.04), (-0.04, 0.05), (0.05, -0.03))


class PrivilegedPickPlaceOracle:
    def __init__(self, env: ControlEnv, observation: dict):
        self.env = env
        self.observation = observation
        self.actions: list[np.ndarray] = []
        self.states: list[np.ndarray] = []

    def _step(self, action: np.ndarray) -> None:
        self.states.append(self.env.get_sim_state().copy())
        self.actions.append(np.asarray(action, dtype=np.float32))
        self.observation, _, _, _ = self.env.step(action)

    def hold(self, gripper: float, steps: int) -> None:
        for _ in range(steps):
            self._step(np.array([0, 0, 0, 0, 0, 0, gripper], dtype=np.float32))

    def move_to(
        self,
        target_position: np.ndarray,
        gripper: float,
        *,
        max_action: float,
        max_steps: int = 240,
        tolerance: float = 0.008,
    ) -> bool:
        for _ in range(max_steps):
            error = np.asarray(target_position) - self.observation["robot0_eef_pos"]
            if np.linalg.norm(error) < tolerance:
                return True
            translation = np.clip(error / 0.05, -max_action, max_action)
            self._step(np.concatenate((translation, np.zeros(3), [gripper])))
        return False

    def recover(self, object_name: str, target_region_name: str) -> dict:
        object_state = self.env.env.object_states_dict[object_name]
        target_region = self.env.env.object_states_dict[target_region_name]

        object_position = object_state.get_geom_state()["pos"].copy()
        stages = {}
        stages["pregrasp"] = self.move_to(
            np.array([object_position[0], object_position[1], 0.18]),
            OPEN_GRIPPER,
            max_action=0.18,
        )
        stages["descend"] = self.move_to(
            np.array([object_position[0], object_position[1], object_position[2] + 0.028]),
            OPEN_GRIPPER,
            max_action=0.18,
        )
        self.hold(CLOSE_GRIPPER, 25)

        grasp_offset = self.observation["robot0_eef_pos"] - object_state.get_geom_state()["pos"]
        stages["lift"] = self.move_to(
            np.array([
                self.observation["robot0_eef_pos"][0],
                self.observation["robot0_eef_pos"][1],
                0.28,
            ]),
            CLOSE_GRIPPER,
            max_action=0.18,
        )

        target_position = target_region.get_geom_state()["pos"].copy()
        desired_object_position = np.array([target_position[0], target_position[1], 0.175])
        desired_eef_position = desired_object_position + grasp_offset
        stages["transfer"] = self.move_to(
            np.array([desired_eef_position[0], desired_eef_position[1], 0.28]),
            CLOSE_GRIPPER,
            max_action=0.18,
        )
        stages["lower"] = self.move_to(
            desired_eef_position,
            CLOSE_GRIPPER,
            max_action=0.12,
        )
        self.hold(OPEN_GRIPPER, 50)

        return {
            "stages_reached": stages,
            "success": bool(self.env.check_success()),
            "contained": bool(target_region.check_contain(object_state)),
            "contact": bool(target_region.check_contact(object_state)),
            "final_object_position": object_state.get_geom_state()["pos"].tolist(),
            "final_target_position": target_region.get_geom_state()["pos"].tolist(),
            "oracle_steps": len(self.actions),
        }


def reanchor_osc(env: ControlEnv) -> None:
    controller = env.robots[0].controller
    if hasattr(controller, "goal_pos"):
        controller.goal_pos = controller.ee_pos.copy()
    if hasattr(controller, "goal_ori"):
        controller.goal_ori = controller.ee_ori_mat.copy()


def make_failure(env: ControlEnv, observation: dict, object_name: str, offset: tuple[float, float]):
    joint_name = env.env.get_object(object_name).joints[0]
    qpos = env.sim.data.get_joint_qpos(joint_name).copy()
    qpos[0] += offset[0]
    qpos[1] += offset[1]
    env.sim.data.set_joint_qpos(joint_name, qpos)
    env.sim.data.set_joint_qvel(joint_name, np.zeros(6))
    env.sim.forward()
    observation = env.regenerate_obs_from_state(env.get_sim_state())
    for _ in range(10):
        observation, _, _, _ = env.step([0, 0, 0, 0, 0, 0, OPEN_GRIPPER])
    return observation, env.get_sim_state().copy()


def run_episode(task_suite, task_id: int, episode: int, output_dir: Path) -> dict:
    task = task_suite.get_task(task_id)
    bddl_path = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = ControlEnv(
        bddl_file_name=str(bddl_path),
        use_camera_obs=False,
        has_renderer=False,
        has_offscreen_renderer=False,
        control_freq=20,
    )
    try:
        env.seed(7)
        observation = env.reset()
        observation = env.set_init_state(task_suite.get_task_init_states(task_id)[episode])
        for _ in range(10):
            observation, _, _, _ = env.step([0, 0, 0, 0, 0, 0, OPEN_GRIPPER])

        object_name, destination_name = env.obj_of_interest
        target_region_name = next(
            name
            for name, site in env.env.object_sites_dict.items()
            if site.parent_name == destination_name and "contain" in name
        )
        offset = FAILURE_OFFSETS[episode % len(FAILURE_OFFSETS)]
        observation, failure_state = make_failure(env, observation, object_name, offset)
        failure_object_position = env.env.object_states_dict[object_name].get_geom_state()["pos"].copy()

        # Exercise the same restore path needed for policy-induced failure snapshots.
        observation = env.regenerate_obs_from_state(failure_state)
        reanchor_osc(env)

        oracle = PrivilegedPickPlaceOracle(env, observation)
        result = oracle.recover(object_name, target_region_name)
        result.update(
            {
                "episode": episode,
                "task": task.language,
                "object": object_name,
                "target_region": target_region_name,
                "failure_offset_xy": list(offset),
                "failure_object_position": failure_object_position.tolist(),
            }
        )
        np.savez_compressed(
            output_dir / f"episode_{episode:03d}.npz",
            failure_state=failure_state,
            states=np.asarray(oracle.states),
            actions=np.asarray(oracle.actions),
        )
        return result
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/libero/deterministic_c_smoke"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    suite = benchmark.get_benchmark_dict()["libero_object"]()
    results = [run_episode(suite, args.task_id, i, args.output_dir) for i in range(args.episodes)]
    successes = sum(result["success"] for result in results)
    summary = {
        "episodes": len(results),
        "successes": successes,
        "success_rate": successes / len(results),
        "results": results,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
