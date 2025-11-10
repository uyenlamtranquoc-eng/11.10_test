from collections import deque
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import torch


class ReplayBuffer:
    """Simple FIFO replay buffer for off-policy RL."""

    def __init__(self, capacity: int, obs_dim: int):
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self.buffer = deque(maxlen=self.capacity)

    def push(self, state: np.ndarray, action, reward: float, next_state: np.ndarray, done: bool, info: Optional[Dict[str, Any]] = None) -> None:
        """Push transition. Action can be scalar int or array-like (e.g., MultiDiscrete of shape [6])."""
        a = np.asarray(action)
        if a.ndim == 0:
            a = a.astype(np.int64)
        else:
            a = a.astype(np.int64).reshape(-1)
        # 基础回放忽略 info
        self.buffer.append((state.astype(np.float32), a, float(reward), next_state.astype(np.float32), bool(done)))

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states_t = torch.from_numpy(np.stack(states))
        # 统一规范动作到二维 [B, H]（H 可为 1）
        first = actions[0]
        first_arr = np.asarray(first)
        if first_arr.ndim >= 1:
            # 将每个动作压平为 1D，再 stack 成 [B, H]
            norm_actions = [np.asarray(a, dtype=np.int64).reshape(-1) for a in actions]
            actions_t = torch.from_numpy(np.stack(norm_actions)).long()
        else:
            # 纯标量动作：构造 [B, 1]
            actions_t = torch.from_numpy(np.array(actions, dtype=np.int64).reshape(-1, 1)).long()
        rewards_t = torch.tensor(rewards, dtype=torch.float32)
        next_states_t = torch.from_numpy(np.stack(next_states))
        dones_t = torch.tensor(dones, dtype=torch.float32)
        return (states_t, actions_t, rewards_t, next_states_t, dones_t)

    def __len__(self) -> int:
        return len(self.buffer)


 


# Removed PrioritizedReplayBuffer; module now provides only HierarchicalReplayBuffer


class HierarchicalReplayBuffer:
    """Hierarchical (stratified) replay buffer.
    Buckets transitions into strata (e.g., reward sign, action-change magnitude, terminal).
    Sampling draws proportionally from buckets to ensure diverse regimes (e.g., signal-driven regimes).
    """

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        sample_weights: Optional[Dict[str, float]] = None,
        delta_threshold_mps: float = 2.0,
    ) -> None:
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self.delta_thr = float(delta_threshold_mps)
        self.sample_weights: Dict[str, float] = dict(sample_weights or {})
        self.buckets: Dict[str, deque] = {}
        self.bucket_caps: Dict[str, int] = {}
        self.last_action_speeds: Optional[Tuple[float, ...]] = None

    def _ensure_bucket(self, key: str) -> None:
        if key not in self.buckets:
            # 动态按均分容量；后续可根据 sample_weights 调整
            # 简单策略：新桶分配 capacity // 8，最多 8 桶，若不足则至少 1
            default_cap = max(1, int(self.capacity // 8))
            self.buckets[key] = deque(maxlen=default_cap)
            self.bucket_caps[key] = default_cap
            if key not in self.sample_weights:
                self.sample_weights[key] = 1.0

    def _stratify(self, reward: float, info: Optional[Dict[str, Any]], done_flag: bool) -> str:
        # 依据 reward 正负 + 动作速度变化大小分层
        sign = 'pos' if float(reward) >= 0.0 else 'neg'
        change = 'unknown'
        try:
            speeds = None
            if info and 'action_speeds_mps' in info:
                speeds = tuple(map(float, info.get('action_speeds_mps')))
            if speeds is not None and self.last_action_speeds is not None:
                deltas = [abs(a - b) for a, b in zip(speeds, self.last_action_speeds)]
                mean_delta = float(np.mean(deltas))
                change = 'large' if mean_delta >= self.delta_thr else 'small'
            elif speeds is not None:
                change = 'small'
        except Exception:
            change = 'unknown'
        key = f'{sign}_{change}'
        if done_flag:
            key = 'terminal'
        return key

    def push(
        self,
        state: np.ndarray,
        action,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        info: Optional[Dict[str, Any]] = None,
    ) -> None:
        a = np.asarray(action)
        if a.ndim == 0:
            a = a.astype(np.int64)
        else:
            a = a.astype(np.int64).reshape(-1)
        transition = (state.astype(np.float32), a, float(reward), next_state.astype(np.float32), bool(done))
        key = self._stratify(float(reward), info, bool(done))
        self._ensure_bucket(key)
        self.buckets[key].append(transition)
        # 更新 last_action_speeds：仅当 info 给出时
        try:
            if info and ('action_speeds_mps' in info):
                self.last_action_speeds = tuple(map(float, info.get('action_speeds_mps')))
            if done:
                self.last_action_speeds = None
        except Exception:
            pass

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # 依据 sample_weights 分配每桶采样数量；若某桶不足则回退到其他桶
        active = [(k, self.buckets[k]) for k in self.buckets.keys() if len(self.buckets[k]) > 0]
        assert len(active) > 0, 'buffer underflow'
        weights = np.array([self.sample_weights.get(k, 1.0) for k, _ in active], dtype=np.float32)
        weights = weights / (weights.sum() + 1e-8)
        counts = np.maximum(1, (weights * batch_size).astype(int))
        # 修正总数到 batch_size
        diff = batch_size - int(counts.sum())
        while diff != 0:
            idx = int(np.argmax(weights)) if diff > 0 else int(np.argmin(weights))
            counts[idx] += 1 if diff > 0 else -1
            diff = batch_size - int(counts.sum())
        picked: List[Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]] = []
        for (k, dq), c in zip(active, counts):
            c = min(int(c), len(dq))
            if c <= 0:
                continue
            idxs = np.random.choice(len(dq), size=c, replace=False)
            for i in idxs:
                picked.append(dq[i])
        # 若仍不足，则从所有桶补齐
        need = batch_size - len(picked)
        if need > 0:
            pool = []
            for _, dq in active:
                pool.extend(list(dq))
            extra = np.random.choice(len(pool), size=need, replace=False)
            for i in extra:
                picked.append(pool[i])

        states, actions, rewards, next_states, dones = zip(*picked)
        states_t = torch.from_numpy(np.stack(states))
        first_arr = np.asarray(actions[0])
        if first_arr.ndim >= 1:
            norm_actions = [np.asarray(a, dtype=np.int64).reshape(-1) for a in actions]
            actions_t = torch.from_numpy(np.stack(norm_actions)).long()
        else:
            actions_t = torch.from_numpy(np.array(actions, dtype=np.int64).reshape(-1, 1)).long()
        rewards_t = torch.tensor(rewards, dtype=torch.float32)
        next_states_t = torch.from_numpy(np.stack(next_states))
        dones_t = torch.tensor(dones, dtype=torch.float32)
        return (states_t, actions_t, rewards_t, next_states_t, dones_t)

    def __len__(self) -> int:
        return int(sum(len(dq) for dq in self.buckets.values()))