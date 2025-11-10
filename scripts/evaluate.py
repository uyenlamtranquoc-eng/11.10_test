import os
import sys
import argparse
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

import numpy as np
import torch

# package root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from envs.city_vsl_env import CityVSLEnv  # noqa
from rl.sac_agent import MultiHeadSACAgent  # 多头离散SAC（基础实现）

# common utils
from utils.seed_utils import set_seed as _common_set_seed
from utils.config_utils import load_yaml_bom_safe, validate_env_config
from utils.route_utils import (
    get_route_files_value,
    strip_pen_suffix,
    generate_penetrated_routes_from_base,
    ensure_route_penetration,
)


@dataclass
class EvaluationMetrics:
    """评估指标数据类（基础版，仅保留使用到的指标）"""
    rewards: List[float]
    queues: List[float]
    throughputs: List[float]
    
    def get_means(self) -> Dict[str, float]:
        return {
            'reward': float(np.mean(self.rewards)) if self.rewards else 0.0,
            'queue': float(np.mean(self.queues)) if self.queues else 0.0,
            'throughput': float(np.mean(self.throughputs)) if self.throughputs else 0.0,
        }
    
    def get_stds(self) -> Dict[str, float]:
        return {
            'reward': float(np.std(self.rewards)) if self.rewards else 0.0,
            'queue': float(np.std(self.queues)) if self.queues else 0.0,
            'throughput': float(np.std(self.throughputs)) if self.throughputs else 0.0,
        }
    
    def get_counts(self) -> Dict[str, int]:
        return {
            'reward': len(self.rewards),
            'queue': len(self.queues),
            'throughput': len(self.throughputs),
        }


def set_seed(seed: int = 0, worker_id: int = 0):
    """统一与训练的随机种子设置，保证评估可重现性。"""
    return _common_set_seed(seed, worker_id)


# 删除本地兼容转发函数，统一使用 utils 模块中的公共函数


def _find_latest_sac_checkpoint(root_dir: str) -> str:
    """查找SAC检查点，优先加载"最优模型"：
    选择优先级：
    1) best_sac.pt（显式最佳命名）
    2) sac.pt（约定为最佳模型）
    3) 最新的 sac_ep*.pt（周期备份）
    4) 回退到 sac.pt
    """
    ckpt_dir = os.path.join(root_dir, "checkpoints")
    pattern = "sac_ep"
    fallback_name = "sac.pt"
    best_name = "best_sac.pt"
    if not os.path.isdir(ckpt_dir):
        return os.path.join(root_dir, "checkpoints", fallback_name)
    # 1) 显式最佳命名的 best_*.pt
    best_path = os.path.join(ckpt_dir, best_name)
    if os.path.exists(best_path):
        return best_path
    # 2) 约定最佳模型文件 sac.pt
    best_conv_path = os.path.join(ckpt_dir, fallback_name)
    if os.path.exists(best_conv_path):
        return best_conv_path
    latest_path = None
    latest_ep = -1
    for fn in os.listdir(ckpt_dir):
        if fn.startswith(pattern) and fn.endswith(".pt"):
            try:
                num = int(fn.replace(pattern, "").replace(".pt", ""))
            except Exception:
                num = -1
            if num > latest_ep:
                latest_ep = num
                latest_path = os.path.join(ckpt_dir, fn)
    if latest_path:
        return latest_path
    fallback = os.path.join(ckpt_dir, fallback_name)
    return fallback


# 删除旧的路由生成函数副本，依赖 utils.route_utils.generate_penetrated_routes_from_base


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", type=str, default="false")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--policy", type=str, default="both", choices=["best", "baseline", "both"], help="评估策略：best=加载最优模型，baseline=无控制，both=两者都评估")
    parser.add_argument("--device", type=str, default=None, help="评估设备：auto/cuda/cpu，默认auto")
    parser.add_argument("--repeat", type=int, default=1, help="重复评估次数（不同随机种子）")
    parser.add_argument("--seed_base", type=int, default=0, help="随机种子起点")
    args = parser.parse_args()

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    cfg = load_yaml_bom_safe(os.path.join(root, "config", "env.yaml"))
    # 配置合法性检查（硬约束 + 建议）
    try:
        validate_env_config(cfg, root)
    except Exception as e:
        raise SystemExit(f"[config] 配置校验失败：{e}")
    use_gui = str(args.gui).lower() in ("1", "true", "yes")
    # 设备解析：命令行优先，其次配置文件，默认 auto
    dev_pref = (args.device or str(cfg.get("device", "auto")).lower())
    if dev_pref == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif dev_pref in ("cuda", "cpu"):
        device = dev_pref
        if device == "cuda" and not torch.cuda.is_available():
            print("[warn] --device=cuda 指定，但当前环境未检测到可用 CUDA，回退到 CPU")
            device = "cpu"
    else:
        print(f"[warn] 未识别的设备 '{dev_pref}'，使用 auto 策略")
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"[device] 评估使用设备: {device}")
    # 在进行路由准备之前设置随机种子，保证渗透路由的生成可重现
    try:
        set_seed(int(args.seed_base))
    except Exception:
        # 兜底：即使设置失败也继续，但可能影响重现性
        pass

    # 评估阶段使用训练时的 sumocfg；若其指向的渗透路由缺失或为空，则按 penetration 自动生成
    sumo_cfg_rel = cfg.get("sumo_cfg_relpath", "sumo/grid1x3.sumocfg")
    sumo_cfg_path = os.path.join(root, sumo_cfg_rel)

    # 路由准备：确保 .rou.penXX.xml 存在且非空
    try:
        route_rel = get_route_files_value(sumo_cfg_path)
        sumo_dir = os.path.dirname(sumo_cfg_path)
        route_path = os.path.join(sumo_dir, route_rel)
        base_rel = strip_pen_suffix(route_rel)
        base_path = os.path.join(sumo_dir, base_rel)
        pen_rate = float(cfg.get("penetration", 0.0))

        need_gen = False
        if not os.path.exists(route_path):
            need_gen = True
        else:
            try:
                size = os.path.getsize(route_path)
                need_gen = size < 100
            except Exception:
                need_gen = True

        if need_gen:
            print(f"[routes] regenerating penetrated routes for evaluation: {os.path.basename(base_path)} -> {os.path.basename(route_path)} | CAV rate={pen_rate}")
            generate_penetrated_routes_from_base(base_path, route_path, pen_rate, cfg)
            try:
                ensure_route_penetration(route_path, pen_rate, cav_type_id=str(cfg.get("cav_type_id", "CAV")))
            except Exception as _e:
                print(f"[routes] penetration check failed: {_e}")
        else:
            print(f"[routes] using existing route file: {route_rel}")
            # 即使文件存在也进行渗透率一致性检查，若不匹配则重建
            try:
                ok = ensure_route_penetration(route_path, pen_rate, cav_type_id=str(cfg.get("cav_type_id", "CAV")))
            except Exception as _e:
                print(f"[routes] penetration check failed: {_e}")
                ok = False
            if not ok:
                print(f"[routes] mismatch detected; regenerating routes to target penetration={pen_rate}")
                generate_penetrated_routes_from_base(base_path, route_path, pen_rate, cfg)
                try:
                    ensure_route_penetration(route_path, pen_rate, cav_type_id=str(cfg.get("cav_type_id", "CAV")))
                except Exception as _e:
                    print(f"[routes] penetration check failed after regeneration: {_e}")
    except Exception as e:
        print(f"[routes] route preparation skipped due to error: {e}")

    lane_groups = cfg.get(
        "lane_groups",
        {"upstream": ["J0_J1_0", "J0_J1_1", "J0_J1_2"], "mid": ["J1_J2_0", "J1_J2_1", "J1_J2_2"], "down": ["J2_E_0", "J2_E_1", "J2_E_2"]},
    )
    tls_ids = cfg.get("tls_ids", ["J0", "J1", "J2"])

    speeds_kph = cfg.get("discrete_speeds_kph")
    if not speeds_kph:
        mps = cfg.get("action_speeds_mps", [8.33, 10.0, 12.0, 14.0])
        speeds_kph = [round(v * 3.6) for v in mps]

    decision_interval = int(cfg.get("decision_interval", 5))

    # 策略集合
    _pol = args.policy
    policies = [_pol] if _pol != "both" else ["baseline", "best"]
    # 按参数设置评估回合数，保证至少 1 回合
    try:
        num_episodes = int(args.episodes)
    except (TypeError, ValueError):
        num_episodes = 1
    num_episodes = max(1, num_episodes)
    # 统一决策周期语义（按秒）：评估时按秒折算决策步数
    from math import ceil
    max_steps = max(1, int(ceil(int(cfg.get("max_sim_seconds", 3600)) / float(max(int(cfg.get("decision_interval", 5)), 1)))))
    print(f"[timing] decision_interval={int(cfg.get('decision_interval',5))}s | max_sim_seconds={int(cfg.get('max_sim_seconds',3600))}s => max_decision_steps={max_steps}")
    
    # 存储所有策略的评估结果
    policy_results = {}

    for pol in policies:
        print(f"[eval] policy={pol}")
        if pol == "baseline":
            print(f"[baseline] 启用固定限速baseline：保持相同CAV比例，使用固定档位动作")
        cav_type = str(cfg.get("cav_type_id", "CAV"))
        # 奖励裁剪配置
        _rc_min = cfg.get("reward_clip_min", None)
        _rc_max = cfg.get("reward_clip_max", None)
        env = CityVSLEnv(
            sumo_cfg_path=sumo_cfg_path,
            use_gui=use_gui,
            backend=str(cfg.get("backend", "auto")),
            lane_groups=lane_groups,
            lane_groups_wb=cfg.get("lane_groups_wb", {"upstream": [], "mid": [], "down": []}),
            tls_ids=tls_ids,
            discrete_speeds_kph=speeds_kph,
            decision_interval=decision_interval,
            max_sim_seconds=int(cfg.get("max_sim_seconds", 3600)),
            max_delta_kph=float(cfg.get("max_delta_kph", 10.0)),
            cav_type_id=cav_type,
            reward_speed_weight=float(cfg.get("reward_speed_weight", 1.0)),
            reward_congestion_weight=float(cfg.get("reward_congestion_weight", 1.0)),
            reward_delay_weight=float(cfg.get("reward_delay_weight", 1.0)),
            reward_queue_overflow_weight=float(cfg.get("reward_queue_overflow_weight", 1.0)),
            reward_stops_weight=float(cfg.get("reward_stops_weight", 0.5)),
            reward_fuel_weight=float(cfg.get("reward_fuel_weight", 0.1)),
            reward_smoothness_weight=float(cfg.get("reward_smoothness_weight", 0.2)),
            reward_coordination_weight=float(cfg.get("reward_coordination_weight", 0.5)),
            reward_throughput_weight=float(cfg.get("reward_throughput_weight", 0.3)),
            reward_demand_robust_weight=float(cfg.get("reward_demand_robust_weight", 0.3)),
            reward_change_penalty=float(cfg.get("reward_change_penalty", 0.1)),
            reward_queue_overflow_threshold=float(cfg.get("reward_queue_overflow_threshold", 0.8)),
            reward_delay_max_ratio_extra=float(cfg.get("reward_delay_max_ratio_extra", 3.0)),
            reward_use_real_fuel=bool(cfg.get("reward_use_real_fuel", False)),
            warmup_steps=int(cfg.get("warmup_steps", 5)),
            reward_clip_min=_rc_min,
            reward_clip_max=_rc_max,
            # 保持与训练一致的观测归一化配置
            norm_cfg=cfg.get("norm", None),
        )

        obs, _ = env.reset()
        assert env.observation_space is not None, "observation_space cannot be None"
        obs_dim = env.observation_space.shape[0]  # type: ignore
        # MultiDiscrete: nvec（长度=6，每段的档位数）；若为单维离散则退化为 1 头
        if hasattr(env.action_space, "nvec"):
            nvec = list(map(int, env.action_space.nvec))  # type: ignore
        else:
            nvec = [int(env.action_space.n)]  # type: ignore

        agent = None
        ckpt_path = None
        if pol == "best":
            # 创建最小版 SAC Agent
            agent = MultiHeadSACAgent(
                obs_dim=obs_dim,
                nvec=nvec,
                lr=float(cfg.get("learning_rate", 2e-4)),
                actor_lr=float(cfg.get("actor_lr", cfg.get("learning_rate", 2e-4))),
                critic_lr=float(cfg.get("critic_lr", cfg.get("learning_rate", 3e-4))),
                gamma=float(cfg.get("gamma", 0.995)),
                target_update_interval=int(cfg.get("sac_target_update_interval", cfg.get("target_update_interval", 1))),
                device=device,
                tau=float(cfg.get("sac_tau", 0.005)),
                alpha=float(cfg.get("sac_alpha", 0.15)),
                hidden_size=int(cfg.get("hidden_size", 256)),
                max_grad_norm=float(cfg.get("grad_clip_norm", 5.0)),
                # 与训练保持一致的结构开关与自动温度配置
                auto_alpha=bool(cfg.get("sac_auto_alpha", True)),
                alpha_lr=float(cfg.get("sac_alpha_lr", cfg.get("actor_lr", cfg.get("learning_rate", 2e-4)))),
                target_entropy_scale=float(cfg.get("sac_target_entropy_scale", 1.0)),
                use_joint_action_constraint=bool(cfg.get("use_joint_action_constraint", True)),
                actor_use_layer_norm=bool(cfg.get("actor_use_layer_norm", True)),
            )
            ckpt_path = args.ckpt if args.ckpt else _find_latest_sac_checkpoint(root)
            ckpt_path = os.path.join(root, ckpt_path) if not os.path.isabs(ckpt_path) else ckpt_path
            if os.path.exists(ckpt_path):
                agent.load(ckpt_path)
                print(f"[eval] loaded checkpoint: {ckpt_path}")
            else:
                print(f"[eval] checkpoint not found, evaluating with randomly-initialized agent: {ckpt_path}")

        rewards = []
        queues = []
        throughputs = []

        # 用多种随机种子重复评估
        # 计算baseline固定速度对应的离散档位索引（最近邻）
        import numpy as _np
        baseline_kph = float(cfg.get("baseline_fixed_speed_kph", float(speeds_kph[-1])))
        try:
            _idx = int(min(range(len(speeds_kph)), key=lambda i: abs(float(speeds_kph[i]) - baseline_kph)))
        except Exception:
            _idx = int(len(speeds_kph) - 1)
        baseline_action = _np.full(len(nvec), _idx, dtype=_np.int64)

        for rep in range(int(args.repeat)):
            seed = int(args.seed_base) + rep
            set_seed(seed)
            for ep in range(1, num_episodes + 1):
                obs, _ = env.reset()
                ep_reward = 0.0
                for _ in range(max_steps):
                    if agent is None:
                        # baseline：固定限速策略，使用固定档位动作
                        action = baseline_action
                    else:
                        # SAC确定性动作选择
                        action = agent.select_action(obs, deterministic=True)
                    next_obs, reward, terminated, truncated, _info = env.step(action)
                    ep_reward += reward
                    obs = next_obs
                    if terminated or truncated:
                        break

                rewards.append(ep_reward)
                m = env.get_metrics()
                queues.append(m.get("avg_queue_veh", 0.0))
                throughputs.append(m.get("throughput_veh_per_hour", 0.0))

                print(
                    f"[EVAL {pol} seed={seed} EP {ep:03d}] reward={ep_reward:.2f} | queue={m.get('avg_queue_veh',0.0):.2f} | flow={m.get('throughput_veh_per_hour',0.0):.1f} veh/h"
                )

        # 保存评估结果
        metrics = EvaluationMetrics(
            rewards=rewards,
            queues=queues,
            throughputs=throughputs,
        )
        policy_results[pol] = metrics

        # 汇总输出
        def _avg(x):
            return float(np.mean(x)) if x else 0.0

        print(
            f"[EVAL {pol}] avg_reward={_avg(rewards):.2f} | avg_queue={_avg(queues):.2f} | avg_flow={_avg(throughputs):.1f} veh/h"
        )

        env.close()
if __name__ == "__main__":
    main()
