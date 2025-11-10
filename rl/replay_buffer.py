from collections import deque
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import torch


# Note: Uniform ReplayBuffer removed; module provides HierarchicalReplayBuffer only.


 


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

    def _rebalance_caps(self) -> None:
        """
        Rebalance per-bucket capacities so that sum(cap_i) == total capacity,
        distributed proportionally to sample_weights. Ensures at least 1 per bucket.
        """
        keys = list(self.buckets.keys())
        if not keys:
            return
        weights = np.array([float(self.sample_weights.get(k, 1.0)) for k in keys], dtype=np.float32)
        weights = weights / (weights.sum() + 1e-8)
        caps = np.maximum(1, np.floor(weights * float(self.capacity)).astype(int))
        # fix rounding to match total capacity
        diff = int(self.capacity) - int(caps.sum())
        while diff != 0:
            idx = int(np.argmax(weights)) if diff > 0 else int(np.argmin(weights))
            caps[idx] += 1 if diff > 0 else -1
            diff = int(self.capacity) - int(caps.sum())
        # apply new caps
        for k, new_cap in zip(keys, caps.tolist()):
            old_cap = int(self.bucket_caps.get(k, 0))
            if old_cap != int(new_cap):
                old_dq = self.buckets[k]
                new_dq = deque(maxlen=int(new_cap))
                # keep most recent transitions up to new cap
                if len(old_dq) > 0:
                    # deque keeps oldest first; extend last new_cap elements
                    start = max(0, len(old_dq) - int(new_cap))
                    for i in range(start, len(old_dq)):
                        new_dq.append(old_dq[i])
                self.buckets[k] = new_dq
                self.bucket_caps[k] = int(new_cap)

    def _ensure_bucket(self, key: str) -> None:
        if key not in self.buckets:
            # 初始化桶并进行容量再平衡（总容量保持为 self.capacity）
            # 若无权重，默认 1.0
            self.buckets[key] = deque(maxlen=1)
            self.bucket_caps[key] = 1
            if key not in self.sample_weights:
                self.sample_weights[key] = 1.0
            # 根据当前权重分配容量
            self._rebalance_caps()

    def _stratify(self, reward: float, info: Optional[Dict[str, Any]], done_flag: bool) -> str:
        # 终止优先
        if done_flag:
            return 'terminal'

        # 依据环境暴露的信号窗口标志进行信号驱动分桶（优先于默认策略）
        try:
            if info and ('signal_window_flags' in info):
                flags = info.get('signal_window_flags') or {}
                def _any_flag(name: str) -> bool:
                    for dirn in ('eb', 'wb'):
                        d = flags.get(dirn, {})
                        if isinstance(d, dict):
                            for _, fv in d.items():
                                try:
                                    if bool(fv.get(name, False)):
                                        return True
                                except Exception:
                                    pass
                    return False
                is_green = _any_flag('is_green_now')
                is_near = _any_flag('near_green')
                if is_green:
                    return 'signal_green'
                if is_near:
                    return 'signal_near_green'
                # 若存在标志但不在绿窗/近绿窗，则归入红窗类
                return 'signal_red'
        except Exception:
            pass

        # 回退：依据 reward 正负 + 动作速度变化大小分层
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
        return f'{sign}_{change}'

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