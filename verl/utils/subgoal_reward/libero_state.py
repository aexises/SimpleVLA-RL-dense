from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np


@dataclass
class LiberoState:
    task_name: str | None = None
    instruction: str | None = None
    gripper_position: np.ndarray | None = None
    gripper_open: float | None = None
    object_position: np.ndarray | None = None
    target_position: np.ndarray | None = None
    object_in_gripper: bool | None = None
    success: bool = False
    raw_info: dict[str, Any] = field(default_factory=dict)


class LiberoStateExtractor:
    """Best-effort LIBERO / robosuite state extraction.

    The extractor intentionally keeps all environment-specific probing here so
    callers can fall back cleanly when an observation does not expose enough
    structured state for dense subgoals.
    """

    def extract(
        self,
        env: Any = None,
        obs: Mapping[str, Any] | None = None,
        info: Mapping[str, Any] | None = None,
        task_metadata: Mapping[str, Any] | None = None,
        done: bool = False,
    ) -> LiberoState:
        obs = obs or {}
        info = dict(info or {})
        task_metadata = task_metadata or {}

        gripper_position = self._vec3(obs.get("robot0_eef_pos"))
        gripper_open = self._gripper_open(obs)
        object_position = self._object_position(obs, gripper_position, task_metadata)
        target_position = self._target_position(obs, task_metadata)
        success = self._success(env, info, done)

        return LiberoState(
            task_name=self._str_from(task_metadata, "task_name", "task_suite_name"),
            instruction=self._str_from(task_metadata, "instruction", "task_description", "language"),
            gripper_position=gripper_position,
            gripper_open=gripper_open,
            object_position=object_position,
            target_position=target_position,
            object_in_gripper=self._object_in_gripper(obs, info, object_position, gripper_position, gripper_open),
            success=success,
            raw_info=info,
        )

    def _str_from(self, mapping: Mapping[str, Any], *keys: str) -> str | None:
        for key in keys:
            value = mapping.get(key)
            if value is not None:
                return str(value)
        return None

    def _vec3(self, value: Any) -> np.ndarray | None:
        if value is None:
            return None
        try:
            arr = np.asarray(value, dtype=np.float32).reshape(-1)
        except Exception:
            return None
        if arr.size < 3 or not np.all(np.isfinite(arr[:3])):
            return None
        return arr[:3]

    def _gripper_open(self, obs: Mapping[str, Any]) -> float | None:
        qpos = obs.get("robot0_gripper_qpos")
        if qpos is None:
            qpos = obs.get("gripper_qpos")
        if qpos is None:
            return None
        try:
            value = float(np.asarray(qpos, dtype=np.float32).reshape(-1).mean())
        except Exception:
            return None
        # LIBERO gripper qpos is usually small; normalize only enough for
        # thresholding without assuming a fixed robot XML.
        return float(np.clip(value / 0.04, 0.0, 1.0))

    def _candidate_pos_keys(self, obs: Mapping[str, Any]) -> list[str]:
        blocked = ("robot", "eef", "gripper", "camera", "image", "quat")
        keys = []
        for key, value in obs.items():
            lower = key.lower()
            if not lower.endswith("_pos"):
                continue
            if any(token in lower for token in blocked):
                continue
            if self._vec3(value) is not None:
                keys.append(key)
        return keys

    def _object_position(
        self,
        obs: Mapping[str, Any],
        gripper_position: np.ndarray | None,
        task_metadata: Mapping[str, Any],
    ) -> np.ndarray | None:
        object_name = task_metadata.get("object_name")
        if object_name:
            for key in (f"{object_name}_pos", str(object_name)):
                value = self._vec3(obs.get(key))
                if value is not None:
                    return value

        candidates = [(key, self._vec3(obs[key])) for key in self._candidate_pos_keys(obs)]
        candidates = [(key, value) for key, value in candidates if value is not None]
        if not candidates:
            object_state = np.asarray(obs.get("object-state", []), dtype=np.float32).reshape(-1)
            if object_state.size >= 3:
                return object_state[:3]
            return None

        if gripper_position is None:
            return candidates[0][1]
        return min(candidates, key=lambda item: float(np.linalg.norm(item[1] - gripper_position)))[1]

    def _target_position(
        self,
        obs: Mapping[str, Any],
        task_metadata: Mapping[str, Any],
    ) -> np.ndarray | None:
        for key in ("target_position", "target_pos", "goal_position", "goal_pos"):
            value = self._vec3(task_metadata.get(key))
            if value is not None:
                return value
            value = self._vec3(obs.get(key))
            if value is not None:
                return value

        target_tokens = ("target", "goal", "zone", "region", "site")
        for key, value in obs.items():
            lower = key.lower()
            if any(token in lower for token in target_tokens):
                vec = self._vec3(value)
                if vec is not None:
                    return vec
        return None

    def _object_in_gripper(
        self,
        obs: Mapping[str, Any],
        info: Mapping[str, Any],
        object_position: np.ndarray | None,
        gripper_position: np.ndarray | None,
        gripper_open: float | None,
    ) -> bool | None:
        for source in (info, obs):
            for key in ("object_in_gripper", "in_gripper", "grasped", "grasp_success", "contact"):
                if key in source:
                    return bool(source[key])
        if object_position is None or gripper_position is None or gripper_open is None:
            return None
        return bool(np.linalg.norm(object_position - gripper_position) <= 0.04 and gripper_open < 0.5)

    def _success(self, env: Any, info: Mapping[str, Any], done: bool) -> bool:
        for key in ("success", "is_success", "complete"):
            if key in info:
                return bool(info[key])
        if env is not None:
            checker = getattr(env, "_check_success", None)
            if callable(checker):
                try:
                    return bool(checker())
                except Exception:
                    pass
        return bool(done)
