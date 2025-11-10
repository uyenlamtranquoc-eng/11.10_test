import os
import sys
from collections import deque
import argparse
import datetime

import numpy as np
import torch

# package root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from envs.city_vsl_env import CityVSLEnv  # noqa
from rl.sac_agent import MultiHeadSACAgent  # 多头离散SAC（基础实现）
from rl.replay_buffer import HierarchicalReplayBuffer  # 分层采样回放
import pickle

# common utils
from utils.seed_utils import set_seed as _common_set_seed
from utils.config_utils import load_yaml_bom_safe, validate_env_config
from utils.route_utils import (
    get_route_files_value,
)


def set_seed(seed: int = 0, worker_id: int = 0):
    """包装通用 set_seed，保持训练脚本的日志输出风格。"""
    actual = _common_set_seed(seed, worker_id)
    print(f"[随机种子] 设置种子: 基础={seed}, worker={worker_id}, 实际={actual}")
    return actual
    
# 基础版：仅保存/加载智能体检查点（.pt），移除与环境绑定的完整检查点机制



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", type=str, default="false")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--device", type=str, default=None, help="训练设备：auto/cuda/cpu，默认auto")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径（默认config/env.yaml）")
    parser.add_argument("--output_dir", type=str, default=None, help="输出目录（默认outputs/run_YYYYMMDD_HHMMSS）")
    parser.add_argument("--seed", type=int, default=0, help="随机种子（默认0）")
    parser.add_argument("--resume", type=str, default=None, help="恢复训练的检查点路径")
    args = parser.parse_args()

    # 设置随机种子
    set_seed(args.seed)

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    # 支持自定义配置文件路径
    cfg_path = args.config if args.config else os.path.join(root, "config", "env.yaml")
    cfg = load_yaml_bom_safe(cfg_path)
    # 配置合法性检查（硬约束 + 建议）
    try:
        validate_env_config(cfg, root)
    except Exception as e:
        raise SystemExit(f"[config] 配置校验失败：{e}")

    use_gui = str(args.gui).lower() in ("1", "true", "yes")
    # 设备解析：命令行优先，其次 env.yaml，默认 auto
    dev_pref = (args.device or str(cfg.get("device", "auto"))).lower()
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
    print(f"[device] 使用设备: {device}")

    # 使用 env.yaml 的路径；默认回退到 grid1x3
    sumo_cfg_rel = cfg.get("sumo_cfg_relpath", "sumo/grid1x3.sumocfg")
    # 如果想做多渗透率，可以在 cfg 里写多个 cfg，再根据 route_suffix 切
    sumo_cfg_path = os.path.join(root, sumo_cfg_rel)
    # 使用固定的路由文件，不再在训练时按渗透率生成或重建。
    try:
        route_rel = get_route_files_value(sumo_cfg_path)
        sumo_dir = os.path.dirname(sumo_cfg_path)
        route_path = os.path.join(sumo_dir, route_rel)
        if not os.path.exists(route_path):
            raise SystemExit(
                f"[routes] 路由文件不存在: {route_rel} | 请先用 scripts/generate_structured_routes.py 生成并在 sumo 配置中指向该文件"
            )
        try:
            size = os.path.getsize(route_path)
        except Exception:
            size = 0
        if size < 100:
            raise SystemExit(
                f"[routes] 路由文件内容异常(可能为空): {route_rel} | 请用 scripts/generate_structured_routes.py 重新生成"
            )
        print(f"[routes] using route file: {route_rel}")
    except Exception as e:
        raise SystemExit(f"[routes] 路由文件检查失败：{e}")

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

    # 奖励裁剪配置
    _rc_min = cfg.get("reward_clip_min", None)
    _rc_max = cfg.get("reward_clip_max", None)

    # 为多环境并行支持，获取worker ID
    worker_id = int(os.environ.get('WORKER_ID', '0'))
    
    # 设置独立种子
    actual_seed = set_seed(args.seed, worker_id)
    
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
        # 论文式硬约束：一次不超过10km/h
        max_delta_kph=float(cfg.get("max_delta_kph", 10.0)),
        cav_type_id=str(cfg.get("cav_type_id", "CAV")),
        # 从 env.yaml 接入奖励权重与暖机步数
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
        norm_cfg=cfg.get("norm", None),
    )

    obs, _ = env.reset()
    # 类型注解：确保observation_space不为None
    assert env.observation_space is not None, "observation_space cannot be None"
    obs_dim = env.observation_space.shape[0]  # type: ignore
    # MultiDiscrete: nvec（长度=6，每段的档位数）
    if hasattr(env.action_space, "nvec"):
        nvec = list(map(int, env.action_space.nvec))  # type: ignore
    else:
        nvec = [int(env.action_space.n)]  # type: ignore

    # 构建最小版 SAC Agent（多头 MultiDiscrete）

    agent = MultiHeadSACAgent(
        obs_dim=obs_dim,
        nvec=nvec,
        lr=float(cfg.get("learning_rate", 5e-4)),
        actor_lr=float(cfg.get("actor_lr", cfg.get("learning_rate", 5e-4))),
        critic_lr=float(cfg.get("critic_lr", cfg.get("learning_rate", 5e-4))),
        gamma=float(cfg.get("gamma", 0.995)),
        target_update_interval=int(cfg.get("sac_target_update_interval", cfg.get("target_update_interval", 1))),
        device=device,
        tau=float(cfg.get("sac_tau", 0.005)),
        alpha=float(cfg.get("sac_alpha", 0.15)),
        hidden_size=int(cfg.get("hidden_size", 256)),
        max_grad_norm=float(cfg.get("grad_clip_norm", 5.0)),
        auto_alpha=bool(cfg.get("sac_auto_alpha", True)),
        alpha_lr=float(cfg.get("sac_alpha_lr", cfg.get("actor_lr", cfg.get("learning_rate", 5e-4)))),
        target_entropy_scale=float(cfg.get("sac_target_entropy_scale", 1.0)),
        use_joint_action_constraint=bool(cfg.get("use_joint_action_constraint", True)),
        actor_use_layer_norm=bool(cfg.get("actor_use_layer_norm", True)),
        actor_context_samples=int(cfg.get("actor_context_samples", 1)),
        target_context_samples=int(cfg.get("target_context_samples", 1)),
    )
    print(f"[SAC] 使用 Polyak 软更新，tau={float(cfg.get('sac_tau', 0.005)):.4f}")

    # 经验回放：使用分层回放（Stratified Replay）
    buffer = HierarchicalReplayBuffer(
        capacity=int(cfg.get("buffer_size", 200000)),
        obs_dim=obs_dim,
        sample_weights=cfg.get("hier_sample_weights", None),
        delta_threshold_mps=float(cfg.get("hier_delta_threshold_mps", 2.0)),
    )
    print("[replay] 使用分层经验回放 (stratified)")
    batch_size = int(cfg.get("batch_size", 128))

    # 预填充经验池（随机策略）
    prefill_steps = int(cfg.get("prefill_steps", 0) or 0)
    if prefill_steps > 0:
        print(f"[prefill] start prefill_steps={prefill_steps}")
        filled = 0
        obs, _ = env.reset()
        while filled < prefill_steps:
            try:
                action = env.action_space.sample()
            except Exception:
                # 回退：若环境不支持 sample，则使用均匀随机
                import numpy as _np
                action = _np.array([int(_np.random.randint(0, n)) for n in nvec], dtype=_np.int64)
            next_obs, reward, terminated, truncated, _info = env.step(action)
            # 将实际执行的动作（裁剪/缩放后）写入回放
            try:
                if _info and ('executed_action_levels' in _info):
                    executed = np.asarray(_info['executed_action_levels'], dtype=np.int64).reshape(-1)
                elif _info and ('action_speeds_mps' in _info) and hasattr(env, 'speed_levels_mps') and (env.speed_levels_mps is not None):
                    # 兼容老版本：根据速度映射回离散索引（就近匹配）
                    levels = np.asarray(getattr(env, 'speed_levels_mps'))
                    speeds = np.asarray(_info['action_speeds_mps'], dtype=np.float32).reshape(-1)
                    # 对每段取最近档位索引
                    diffs = np.abs(speeds[:, None] - levels[None, :])
                    executed = np.argmin(diffs, axis=1).astype(np.int64)
                else:
                    executed = np.asarray(action, dtype=np.int64).reshape(-1)
            except Exception:
                executed = np.asarray(action, dtype=np.int64).reshape(-1)
            # 分层经验回放（使用已执行动作）
            buffer.push(obs, executed, reward, next_obs, bool(terminated or truncated), _info)  # type: ignore
            obs = next_obs if not (terminated or truncated) else env.reset()[0]
            filled += 1
        print(f"[prefill] done filled={filled}")

    global_steps = 0

    num_episodes = int(args.episodes or cfg.get("episodes", 10))
    # 周期备份频率：每 N 回合保存一次 *_ep{ep}.pt；0/None 表示关闭
    save_every = int(cfg.get("save_every_n_episodes", 0) or 0)
    # 统一决策周期语义（按秒）：按 step-length 折算，每个决策步覆盖 decision_interval 秒
    max_sim_seconds = int(cfg.get("max_sim_seconds", 3600))
    from math import ceil
    max_decision_steps_per_ep = max(1, int(ceil(max_sim_seconds / float(max(decision_interval, 1)))))
    print(f"[timing] decision_interval={decision_interval}s | max_sim_seconds={max_sim_seconds}s => max_decision_steps={max_decision_steps_per_ep}")

    save_dir = args.output_dir if args.output_dir else os.path.join(root, "checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    reward_window = deque(maxlen=20)
    # CSV 指标输出（每回合）
    metrics_dir = args.output_dir if args.output_dir else env.run_output_dir
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_csv = os.path.join(metrics_dir, "train_metrics.csv")
    if not os.path.exists(metrics_csv):
        with open(metrics_csv, "w", encoding="utf-8") as f:
            f.write(
                "ep,algo,tau,reward,avg20,alpha,"
                "avg_queue_veh,throughput_veh_per_hour,arrived_total,sim_seconds,"
                "avg_delay_norm,avg_stops_norm,avg_speed_fluct_norm,"
                "gate_mean_eb,gate_mean_wb,coupling_clamp_events,local_factor_mean,"
                "r_efficiency_avg,r_throughput_avg,r_coordination_avg,r_congestion_avg,r_delay_avg,r_queue_overflow_avg,"
                "r_stops_avg,r_fuel_avg,r_smoothness_avg,r_demand_robust_avg,r_signal_compliance_avg,r_green_pass_avg,"
                "td_error_mean,actor_loss\n"
            )
    best_avg20 = -1e18
    best_ckpt = None

    # 如果指定了恢复检查点，则加载
    start_episode = 1
    if args.resume and os.path.exists(args.resume):
        try:
            agent.load(args.resume)
            print(f"[恢复] 从检查点恢复训练: {args.resume}")
        except Exception as e:
            print(f"[恢复] 加载检查点失败: {e}, 从头开始训练")

    for ep in range(start_episode, num_episodes + 1):
        obs, _ = env.reset()
        ep_reward = 0.0

        for di_step in range(max_decision_steps_per_ep):
            try:
                # SAC 随机动作选择
                action = agent.select_action(obs, deterministic=False)
                next_obs, reward, terminated, truncated, info = env.step(action)

                done_flag = bool(terminated or truncated)
                # 将实际执行的动作（裁剪/缩放后）写入回放
                try:
                    if info and ('executed_action_levels' in info):
                        executed = np.asarray(info['executed_action_levels'], dtype=np.int64).reshape(-1)
                    elif info and ('action_speeds_mps' in info) and hasattr(env, 'speed_levels_mps') and (env.speed_levels_mps is not None):
                        levels = np.asarray(getattr(env, 'speed_levels_mps'))
                        speeds = np.asarray(info['action_speeds_mps'], dtype=np.float32).reshape(-1)
                        diffs = np.abs(speeds[:, None] - levels[None, :])
                        executed = np.argmin(diffs, axis=1).astype(np.int64)
                    else:
                        executed = np.asarray(action, dtype=np.int64).reshape(-1)
                except Exception:
                    executed = np.asarray(action, dtype=np.int64).reshape(-1)
                # 回放记录使用已执行动作
                buffer.push(obs, executed, reward, next_obs, done_flag, info)  # type: ignore

                if len(buffer) >= batch_size:
                    batch = buffer.sample(batch_size)
                    metrics = agent.update(batch)
                    loss = metrics.get('total_loss', None)
                else:
                    loss = None

                ep_reward += reward
                obs = next_obs

                # 计步
                global_steps += 1

                if terminated or truncated:
                    break
                    
            except Exception as e:
                print(f"[异常] 回合{ep}步{di_step+1}发生未捕获异常: {e}")
                break

        reward_window.append(ep_reward)
        avg20 = float(np.mean(reward_window)) if reward_window else ep_reward
        print(
            f"[EP {ep:03d}] reward={ep_reward:.2f} | avg20={avg20:.2f} | alpha={getattr(agent, 'temperature', 0.0):.3f} | heads={len(nvec)} | nvec={nvec}"
        )

        # 追加 CSV 指标
        m = env.get_metrics()
        alpha_val = getattr(agent, "temperature", 0.0)
        # 从智能体指标中读取门控均值（若启用联合约束），否则为0
        g_eb = 0.0
        g_wb = 0.0
        try:
            g_eb = float(getattr(agent, 'metrics', {}).get('gate_mean_eb', 0.0) or 0.0)
            g_wb = float(getattr(agent, 'metrics', {}).get('gate_mean_wb', 0.0) or 0.0)
        except Exception:
            pass
        with open(metrics_csv, "a", encoding="utf-8") as f:
            f.write(
                f"{ep},sac,{float(cfg.get('sac_tau', 0.0) or 0.0):.6f},{ep_reward:.6f},{avg20:.6f},{alpha_val:.6f},"
                f"{m.get('avg_queue_veh',0.0):.6f},{m.get('throughput_veh_per_hour',0.0):.6f},{int(m.get('arrived_total',0) or 0):d},{float(m.get('sim_seconds',0.0) or 0.0):.6f},"
                f"{m.get('avg_delay_norm',0.0):.6f},{m.get('avg_stops_norm',0.0):.6f},{m.get('avg_speed_fluct_norm',0.0):.6f},"
                f"{g_eb:.6f},{g_wb:.6f},{int(m.get('coupling_clamp_events',0) or 0):d},{m.get('local_factor_mean',0.0):.6f},"
                f"{m.get('r_efficiency_avg',0.0):.6f},{m.get('r_throughput_avg',0.0):.6f},{m.get('r_coordination_avg',0.0):.6f},{m.get('r_congestion_avg',0.0):.6f},{m.get('r_delay_avg',0.0):.6f},{m.get('r_queue_overflow_avg',0.0):.6f},"
                f"{m.get('r_stops_avg',0.0):.6f},{m.get('r_fuel_avg',0.0):.6f},{m.get('r_smoothness_avg',0.0):.6f},{m.get('r_demand_robust_avg',0.0):.6f},{m.get('r_signal_compliance_avg',0.0):.6f},{m.get('r_green_pass_avg',0.0):.6f},"
                f"{float(getattr(agent,'metrics',{}).get('td_error_mean',0.0)):.6f},{float(getattr(agent,'metrics',{}).get('actor_loss',0.0)):.6f}\n"
            )

        # 基础版：不保存环境绑定的完整检查点，统一使用智能体检查点

        # 保存最佳模型（按 avg20）
        if avg20 > best_avg20:
            best_avg20 = avg20
            best_name = "sac.pt"
            best_ckpt = os.path.join(save_dir, best_name)
            agent.save(best_ckpt)
            compat_best_name = "best_sac.pt"
            agent.save(os.path.join(save_dir, compat_best_name))
            print(f"[save] new best checkpoint: {best_ckpt} (also wrote {compat_best_name}) | avg20={best_avg20:.2f}")

        # 周期备份：每 N 回合保存 *_ep{ep}.pt
        if save_every > 0 and (ep % save_every == 0):
            ep_name = f"sac_ep{ep}.pt"
            ep_ckpt = os.path.join(save_dir, ep_name)
            agent.save(ep_ckpt)
            print(f"[backup] wrote periodic checkpoint: {ep_ckpt}")

    env.close()


if __name__ == "__main__":
    main()
