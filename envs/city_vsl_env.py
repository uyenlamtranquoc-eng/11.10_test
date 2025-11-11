# type: ignore  # Ignore all type checking for traci
import os
import sys
import importlib
import datetime
from collections import deque
from typing import Dict, List, Tuple, Optional
import xml.etree.ElementTree as ET

import numpy as np

import gymnasium as gym
from gymnasium import spaces

# 延迟选择后端：优先在 _init_sumo 中按 use_gui/可用性决定
try:
    import traci as _traci_mod  # type: ignore
except Exception:
    _traci_mod = None
traci = _traci_mod  # type: ignore


class CityVSLEnv(gym.Env):
    """
    城市走廊（单向，3个信号）可变限速的RL环境
    - 分段限速：upstream / mid / down 三段
    - 只控制 CAV：typeID == "CAV" 的车
    - 每 decision_interval 秒做一次决策
    - 状态包含：当前运行量 + 信号相位 + CAV渗透率 + 简单预测量（对应论文的state extension）
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        sumo_cfg_path: str,
        use_gui: bool = False,
        backend: str = "auto",  # auto/libsumo/traci
        lane_groups: Optional[Dict[str, List[str]]] = None,
        lane_groups_wb: Optional[Dict[str, List[str]]] = None,
        tls_ids: Optional[List[str]] = None,
        discrete_speeds_kph: Optional[List[int]] = None,
        decision_interval: int = 5,
        cav_type_id: str = "CAV",
        max_sim_seconds: int = 3600,
        max_delta_kph: float = 10.0,  # 限速变化硬约束，论文式
        reward_speed_weight: float = 1.0,
        reward_congestion_weight: float = 1.0,
        reward_delay_weight: float = 1.0,
        reward_queue_overflow_weight: float = 1.0,
        reward_stops_weight: float = 0.5,
        reward_fuel_weight: float = 0.1,
        reward_smoothness_weight: float = 0.2,
        reward_coordination_weight: float = 0.5,
        reward_throughput_weight: float = 0.3,
        reward_demand_robust_weight: float = 0.3,
        reward_change_penalty: float = 0.1,
        # 新增：可配置的信号遵从与绿窗通过权重（保持向后兼容）
        reward_signal_compliance_weight: float = 0.2,
        reward_green_pass_weight: float = 0.3,
        warmup_steps: int = 5,
        reward_clip_min: Optional[float] = None,
        reward_clip_max: Optional[float] = None,
        reward_queue_overflow_threshold: float = 0.8,
        reward_delay_max_ratio_extra: float = 3.0,
        reward_use_real_fuel: bool = False,
        norm_cfg: Optional[Dict[str, object]] = None,
    ):
        super().__init__()

        self.sumo_cfg_path = sumo_cfg_path
        self.use_gui = use_gui
        self.backend_pref = (backend or "auto").lower()
        self.decision_interval = int(decision_interval)
        self.cav_type_id = cav_type_id
        self.max_sim_seconds = max_sim_seconds
        self.max_delta_kph = max_delta_kph
        self.warmup_steps = int(warmup_steps)

        # 评估指标累计器（按仿真步聚合）
        self.metrics = {
            "queue_sum": 0.0,
            "queue_steps": 0,
            "arrived_total": 0,
            "sim_seconds": 0,
            # 归一指标累计（用于 episode 汇总）
            "delay_sum_norm": 0.0,
            "delay_steps": 0,
            "stops_sum_norm": 0.0,
            "stops_steps": 0,
            "smooth_sum_norm": 0.0,
            "smooth_steps": 0,
        }

        # 三段路（东向）
        self.lane_groups = lane_groups or {
            "upstream": [],
            "mid": [],
            "down": [],
        }
        self.group_keys = ["upstream", "mid", "down"]

        # 三段路（西向，可选）：与 group_keys 对齐键名
        self.wb_lane_groups = lane_groups_wb or {
            "upstream": [],
            "mid": [],
            "down": [],
        }

        # 三个信号
        self.tls_ids = tls_ids or ["J0", "J1", "J2"]

        # 离散车速
        self.discrete_speeds_kph = discrete_speeds_kph or [30, 35, 40, 45, 50, 55]
        self.speed_levels_mps = [v / 3.6 for v in self.discrete_speeds_kph]

        # 构建动作空间：6 个独立离散动作（每段一档）
        n_per_segment = len(self.speed_levels_mps) if self.speed_levels_mps else 1
        self.action_space = spaces.MultiDiscrete(np.array([n_per_segment] * 6, dtype=np.int64))

        # 观测空间（增强版 + 双趋势 + 全局CAV + 方向化变幅因子）：
        # 东向3段 + 西向3段：每段 occ + speed + halts_norm = 3 * 6 = 18
        # 每信号: phase_norm + remain_norm + next_green_start_norm + next_green_duration_norm = 4 * 3 = 12
        # 需求趋势（60s滑动）= 1；慢趋势（5min滑动）= 1
        # 全局 CAV 占比（episode 级 sanity check）= 1
        # 每段 CAV 占比 = 6
        # 每段速度波动（与上一步的差值，归一化）= 6
        # 方向化变幅因子（红绿切换导致的临时收紧/放宽）= 2（EB/WB）
        self.obs_dim = (3 * len(self.group_keys) * 2) + (4 * len(self.tls_ids)) + 2 + 1 + (len(self.group_keys) * 2) + (len(self.group_keys) * 2) + 2
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32
        )

        # reward 权重：对应论文的多目标
        self.rw = {
            "efficiency": float(reward_speed_weight),
            "congestion": float(reward_congestion_weight),
            "delay": float(reward_delay_weight),
            "queue_overflow": float(reward_queue_overflow_weight),
            "stops": float(reward_stops_weight),
            "fuel": float(reward_fuel_weight),
            "smoothness": float(reward_smoothness_weight),
            "coordination": float(reward_coordination_weight),
            "throughput": float(reward_throughput_weight),
            "demand_robust": float(reward_demand_robust_weight),
            "action_change": float(reward_change_penalty),
            # 新增：信号遵从惩罚与绿窗通过奖励（可配置，默认与之前一致）
            "signal_compliance": float(reward_signal_compliance_weight),
            "green_pass": float(reward_green_pass_weight),
        }

        self._reward_clip_min = reward_clip_min
        self._reward_clip_max = reward_clip_max

        # 奖励归一与阈值
        self.queue_overflow_threshold = float(reward_queue_overflow_threshold)
        self.delay_max_ratio_extra = float(reward_delay_max_ratio_extra)
        self.use_real_fuel = bool(reward_use_real_fuel)

        # 归一化参考：速度按离散档位最大值（至少 20 m/s）；信号按实际程序配置
        self.max_ref_speed_mps = max(20.0, max(self.speed_levels_mps)) if self.speed_levels_mps else 20.0
        self._tls_phase_meta: Dict[str, Dict[str, float]] = {}
        # 周期级信号缓存：每相位时长、是否含绿、总周期长度等
        self._tls_cycle_meta: Dict[str, Dict[str, object]] = {}
        self._green_phase_cache: Dict[str, set] = {}
        # 路段↔信号映射与信号状态缓存（用于动作耦合）
        self.segment_tls_map: Dict[str, Dict[str, str]] = {"eb": {}, "wb": {}}
        self._prev_tls_green_state: Dict[str, bool] = {tid: False for tid in (self.tls_ids or [])}
        self._prev_segment_speeds_norm: Dict[str, Dict[str, float]] = {"eb": {}, "wb": {}}
        self._prev_meas_segment_speeds_norm: Dict[str, Dict[str, float]] = {"eb": {}, "wb": {}}
        # 最近一次耦合变幅因子（方向/分段），用于观测提示策略当步能实现的最大变幅
        self.last_local_factor: Dict[str, Dict[str, float]] = {
            "eb": {k: 1.0 for k in self.group_keys},
            "wb": {k: 1.0 for k in self.group_keys},
        }

        # 新增：方向化局部因子缩放与速度波动尺度（从 norm_cfg 读取，提供默认值）
        _norm = norm_cfg or {}
        self.local_factor_dir_scale: Dict[str, float] = {
            "eb": float(_norm.get("local_factor_dir_scale_eb", 1.0)),
            "wb": float(_norm.get("local_factor_dir_scale_wb", 1.0)),
        }
        # smoothness_penalty_scale 用于放大速度波动归一指标（不改变原始测量），仅影响奖励与日志
        self.smoothness_penalty_scale: float = float(_norm.get("smoothness_penalty_scale", 1.0))
        self._fuel_hist: deque = deque(maxlen=int(max(3, round(600.0 / float(max(self.decision_interval, 1))))))
        # 提前初始化归一策略配置，避免在首次使用前引用为空
        self.norm_cfg = norm_cfg or {}
        # 新增：近绿窗窗口、黄灯容差、耦合钳制事件去抖与冷却、绿窗通过占有率权重幂
        self.near_green_window_seconds: float = float(self.norm_cfg.get("near_green_window_seconds", 2.0 * float(self.decision_interval)))
        self.yellow_phase_tolerance_seconds: float = float(self.norm_cfg.get("yellow_phase_tolerance_seconds", 2.0))
        self.clamp_event_debounce_steps: int = int(self.norm_cfg.get("coupling_clamp_debounce_steps", 1))
        self.clamp_event_cooldown_steps: int = int(self.norm_cfg.get("coupling_clamp_cooldown_steps", 3))
        self.green_pass_occ_weight_pow: float = float(self.norm_cfg.get("green_pass_occ_weight_pow", 1.0))
        # 钳制事件状态：分段方向去抖累计与冷却计数
        self._clamp_event_pending: Dict[Tuple[str, str], int] = {}
        self._clamp_event_cooldowns: Dict[Tuple[str, str], int] = {}
        # 需求趋势：最近数步到达流率（veh/s）的滑动窗口（快与慢两个时间尺度，可配置）
        _trend_fast_sec = float(self.norm_cfg.get("arrival_trend_window_seconds_fast", 60.0))
        _trend_slow_sec = float(self.norm_cfg.get("arrival_trend_window_seconds_slow", 300.0))
        self._trend_window_steps: int = int(max(3, round(_trend_fast_sec / float(max(self.decision_interval, 1)))))
        self._arrival_rate_hist: deque = deque(maxlen=self._trend_window_steps)
        # 方向化到达率窗口（EB/WB）：与全局窗口同长度
        self._arrival_rate_hist_eb: deque = deque(maxlen=self._trend_window_steps)
        self._arrival_rate_hist_wb: deque = deque(maxlen=self._trend_window_steps)
        self._trend_window_steps_slow: int = int(max(3, round(_trend_slow_sec / float(max(self.decision_interval, 1)))))  # 慢趋势窗口
        self._arrival_rate_hist_slow: deque = deque(maxlen=self._trend_window_steps_slow)
        self._arrival_rate_hist_slow_eb: deque = deque(maxlen=self._trend_window_steps_slow)
        self._arrival_rate_hist_slow_wb: deque = deque(maxlen=self._trend_window_steps_slow)
        # 实时兜底权重：将瞬时到达率混入趋势归一，降低窗口滞后
        self._trend_inst_fallback_w: float = float(self.norm_cfg.get("trend_inst_fallback_weight", 0.0))
        # 归一化参考通行能力（veh/h）：按下游段车道数估计
        down_lanes = len(self.lane_groups.get("down", [])) + len(self.wb_lane_groups.get("down", []))
        down_lanes = int(max(down_lanes, 1))
        self._ref_throughput_vph: float = float(1800 * down_lanes)

        # 仿真时间（秒，按 SUMO 实时累计）
        self.sim_time = 0.0
        self._prev_time_s = 0.0
        self._last_action_index = -1
        # 初始动作速度：取第0档（若存在），否则 20 m/s 兜底
        init_speed = (self.speed_levels_mps[0] if self.speed_levels_mps else 20.0)
        self._last_action_speeds = (init_speed,) * 6  # m/s（东向3段 + 西向3段）

        # 控制集追踪与默认速度缓存（用于避免“限速残留”）
        self._controlled_veh_ids: set = set()
        self._veh_type_max_speed: Dict[str, float] = {}

        # 决策周期语义：按秒；运行时按 step-length 折算为仿真步数
        self._decision_interval_s: float = float(self.decision_interval)

        # 每步指标缓存：减少 reward 与 _observe 重复的 TraCI 读取
        # 结构示例：
        # {
        #   "lane_metrics": {
        #       "eb": {"upstream": {"speed": 10.0, "halts": 0.5, "occ": 0.2}, ...},
        #       "wb": {"upstream": {"speed": 9.5,  "halts": 0.7, "occ": 0.25}, ...}
        #   }
        # }
        self._step_cache: Dict[str, Dict] = {}

        # 每次运行使用独立时间戳子目录：outputs/YYYYMMDD_HHMMSS
        self.run_output_dir = self._make_run_output_dir()
        self._run_additional_path = None

        # 走廊出口集合（用于“仅走廊范围”的需求趋势）
        self._corridor_exit_edges: List[str] = []
        self._corridor_exit_lanes: List[str] = []
        self._corridor_exit_seen: set = set()
        # 方向化出口集合与车辆追踪
        self._corridor_exit_lanes_eb: List[str] = []
        self._corridor_exit_lanes_wb: List[str] = []
        self._corridor_exit_seen_eb: set = set()
        self._corridor_exit_seen_wb: set = set()

        # —— 归一策略（场景适配）——
        # halts 归一上限策略
        self.halts_upper_strategy: str = str(self.norm_cfg.get("halts_upper_strategy", "by_lane_length_and_p95")).lower()
        self.halts_fixed_upper: float = float(self.norm_cfg.get("halts_fixed_upper", 5.0))
        self.jam_spacing_m: float = float(self.norm_cfg.get("jam_spacing_m", 7.5))
        self._halts_window_steps: int = int(max(3, round(float(self.norm_cfg.get("halts_window_seconds", 600)) / float(max(self.decision_interval, 1)))))
        # rolling 历史：分方向、分段
        self._halts_hist_eb: Dict[str, deque] = {g: deque(maxlen=self._halts_window_steps) for g in self.group_keys}
        self._halts_hist_wb: Dict[str, deque] = {g: deque(maxlen=self._halts_window_steps) for g in self.group_keys}
        # 车道长度缓存与分段物理上限（按长度估算）
        self._lane_length_cache: Dict[str, float] = {}
        self._halts_upper_eb: Dict[str, float] = {g: float(self.halts_fixed_upper) for g in self.group_keys}
        self._halts_upper_wb: Dict[str, float] = {g: float(self.halts_fixed_upper) for g in self.group_keys}

        # 吞吐参考策略
        self.throughput_ref_strategy: str = str(self.norm_cfg.get("throughput_ref_strategy", "rolling_p95")).lower()
        self.saturation_flow_per_lane_vphg: float = float(self.norm_cfg.get("saturation_flow_per_lane_vphg", 1800.0))
        self.progression_factor: float = float(self.norm_cfg.get("progression_factor", 0.9))
        self._throughput_window_steps: int = int(max(3, round(float(self.norm_cfg.get("throughput_window_seconds", 600)) / float(max(self.decision_interval, 1)))))
        self._throughput_ref_hist: deque = deque(maxlen=self._throughput_window_steps)
        try:
            self._init_corridor_exits()
        except Exception:
            self._corridor_exit_edges = []
            self._corridor_exit_lanes = []
            self._corridor_exit_seen = set()
            self._corridor_exit_lanes_eb = []
            self._corridor_exit_lanes_wb = []
            self._corridor_exit_seen_eb = set()
            self._corridor_exit_seen_wb = set()

        # 起 SUMO
        self._init_sumo()

    # --------------------------------------------------------
    def _init_sumo(self):
        if "SUMO_HOME" not in os.environ or not os.environ.get("SUMO_HOME"):
            raise RuntimeError("Please set SUMO_HOME before running the environment.")
        # 确保 tools 在 sys.path 以支持 traci 导入
        tools_path = os.path.join(os.environ["SUMO_HOME"], "tools")
        if tools_path not in sys.path:
            sys.path.append(tools_path)

        # 选择后端：GUI 强制使用 TraCI+sumo-gui；非 GUI 优先 libsumo（失败回退 TraCI）
        global traci
        backend_used = None
        # 使用原始 sumocfg；为本次运行生成 additional 覆盖文件，将 e2 输出重定向到运行目录
        sumocfg_to_use = self.sumo_cfg_path
        try:
            self._run_additional_path = self._write_run_additional_file()
        except Exception:
            self._run_additional_path = None
        # 日志文件：将 SUMO 控制台消息写入运行目录
        message_log_path = os.path.join(self.run_output_dir, "sumo_messages.log")

        if self.use_gui:
            traci = importlib.import_module("traci")
            sumo_cmd = ["sumo-gui", "-c", sumocfg_to_use, "--start"]
            # 写入 SUMO 日志到运行目录并静音控制台
            sumo_cmd += [
                "--message-log", message_log_path,
                "--log", message_log_path,
                "--error-log", message_log_path,
                "--no-warnings", "true",
                "--no-step-log", "true",
            ]
            # 覆盖 additional 文件（若生成成功）
            if self._run_additional_path:
                sumo_cmd += ["--additional-files", self._run_additional_path]
            traci.start(sumo_cmd)
            backend_used = "traci-gui"
        else:
            pref = self.backend_pref
            if pref in ("auto", "libsumo"):
                try:
                    traci = importlib.import_module("libsumo")
                    # libsumo 使用 load(...) 载入仿真，无需指定外部进程名
                    load_args = ["-c", sumocfg_to_use, "--start"]
                    load_args += [
                        "--message-log", message_log_path,
                        "--log", message_log_path,
                        "--error-log", message_log_path,
                        "--no-warnings", "true",
                        "--no-step-log", "true",
                    ]
                    if self._run_additional_path:
                        load_args += ["--additional-files", self._run_additional_path]
                    if hasattr(traci, "isLoaded") and traci.isLoaded():
                        traci.close()
                    traci.load(load_args)
                    backend_used = "libsumo"
                except Exception:
                    traci = importlib.import_module("traci")
                    sumo_cmd = ["sumo", "-c", sumocfg_to_use, "--start"]
                    sumo_cmd += [
                        "--message-log", message_log_path,
                        "--log", message_log_path,
                        "--error-log", message_log_path,
                        "--no-warnings", "true",
                        "--no-step-log", "true",
                    ]
                    if self._run_additional_path:
                        sumo_cmd += ["--additional-files", self._run_additional_path]
                    traci.start(sumo_cmd)
                    backend_used = "traci"
            elif pref == "traci":
                traci = importlib.import_module("traci")
                sumo_cmd = ["sumo", "-c", sumocfg_to_use, "--start"]
                sumo_cmd += [
                    "--message-log", message_log_path,
                    "--log", message_log_path,
                    "--error-log", message_log_path,
                    "--no-warnings", "true",
                    "--no-step-log", "true",
                ]
                if self._run_additional_path:
                    sumo_cmd += ["--additional-files", self._run_additional_path]
                traci.start(sumo_cmd)
                backend_used = "traci"
            else:
                # 未识别：回到 auto
                try:
                    traci = importlib.import_module("libsumo")
                    # 与上面保持一致，libsumo 使用 load(...) 而不是 start(...)
                    load_args = ["-c", sumocfg_to_use, "--start"]
                    load_args += [
                        "--message-log", message_log_path,
                        "--log", message_log_path,
                        "--error-log", message_log_path,
                        "--no-warnings", "true",
                        "--no-step-log", "true",
                    ]
                    if self._run_additional_path:
                        load_args += ["--additional-files", self._run_additional_path]
                    if hasattr(traci, "isLoaded") and traci.isLoaded():
                        traci.close()
                    traci.load(load_args)
                    backend_used = "libsumo"
                except Exception:
                    traci = importlib.import_module("traci")
                    traci.start([
                        "sumo", "-c", sumocfg_to_use, "--start",
                        "--message-log", message_log_path,
                        "--log", message_log_path,
                        "--error-log", message_log_path,
                        "--no-warnings", "true",
                        "--no-step-log", "true",
                    ] + (["--additional-files", self._run_additional_path] if self._run_additional_path else []))
                    backend_used = "traci"
        # 不在控制台打印后端信息；如需调试，可写入消息日志
        # 缓存信号相位元数据用于归一化
        self._compute_tls_phase_meta()
        # 构建默认的“段→下游信号”映射（启发式：段索引+1 → 信号索引）
        self._build_segment_tls_map()

    # --------------------------------------------------------
    def _get_net_file_path_from_sumocfg(self) -> Optional[str]:
        """从 sumo_cfg_path 解析 net.xml 的路径。"""
        try:
            tree = ET.parse(self.sumo_cfg_path)
            root = tree.getroot()
            net_file_val = None
            for elem in root.iter():
                if elem.tag.endswith('net-file'):
                    net_file_val = elem.attrib.get('value')
                    if net_file_val:
                        break
            if not net_file_val:
                return None
            base = os.path.dirname(self.sumo_cfg_path)
            return os.path.join(base, net_file_val)
        except Exception:
            return None

    # --------------------------------------------------------
    def _init_corridor_exits(self) -> None:
        """识别走廊下游出口边与车道（方向化：EB→E；WB→W）。"""
        net_path = self._get_net_file_path_from_sumocfg()
        if not net_path or not os.path.exists(net_path):
            return

        def _edge_id_from_lane(lid: str) -> Optional[str]:
            parts = lid.split('_')
            if len(parts) < 3:
                return None
            return '_'.join(parts[:-1])

        def _to_node_from_edge(eid: str) -> Optional[str]:
            parts = eid.split('_')
            if len(parts) < 2:
                return None
            return parts[-1]

        eb_end_node = 'J2'
        wb_end_node = 'J0'
        eb_down_lanes = self.lane_groups.get('down', [])
        wb_down_lanes = self.wb_lane_groups.get('down', [])
        if eb_down_lanes:
            eb_eid = _edge_id_from_lane(eb_down_lanes[0])
            maybe = _to_node_from_edge(eb_eid) if eb_eid else None
            if maybe:
                eb_end_node = maybe
        if wb_down_lanes:
            wb_eid = _edge_id_from_lane(wb_down_lanes[0])
            maybe = _to_node_from_edge(wb_eid) if wb_eid else None
            if maybe:
                wb_end_node = maybe

        eb_boundary_to = 'E'
        wb_boundary_to = 'W'

        try:
            tree = ET.parse(net_path)
            root = tree.getroot()
            exit_edges_eb: List[str] = []
            exit_edges_wb: List[str] = []
            exit_lanes_eb: List[str] = []
            exit_lanes_wb: List[str] = []
            for edge in root.findall('edge'):
                eid = edge.attrib.get('id', '')
                frm = edge.attrib.get('from', '')
                to = edge.attrib.get('to', '')
                if frm == eb_end_node and to == eb_boundary_to:
                    exit_edges_eb.append(eid)
                    for lane in edge.findall('lane'):
                        lid = lane.attrib.get('id')
                        if lid:
                            exit_lanes_eb.append(lid)
                elif frm == wb_end_node and to == wb_boundary_to:
                    exit_edges_wb.append(eid)
                    for lane in edge.findall('lane'):
                        lid = lane.attrib.get('id')
                        if lid:
                            exit_lanes_wb.append(lid)
            # 方向化集合
            self._corridor_exit_edges = exit_edges_eb + exit_edges_wb
            self._corridor_exit_lanes_eb = exit_lanes_eb
            self._corridor_exit_lanes_wb = exit_lanes_wb
            # 兼容原有字段（并集）
            self._corridor_exit_lanes = self._corridor_exit_lanes_eb + self._corridor_exit_lanes_wb
        except Exception:
            self._corridor_exit_edges = []
            self._corridor_exit_lanes = []
            self._corridor_exit_lanes_eb = []
            self._corridor_exit_lanes_wb = []

    def _compute_tls_phase_meta(self) -> None:
        """解析当前程序的信号相位，并构建周期级缓存用于后续特征计算。"""
        self._tls_phase_meta = {}
        self._tls_cycle_meta = {}
        for tid in self.tls_ids:
            try:
                # 选择当前程序对应的逻辑
                try:
                    curr_prog = traci.trafficlight.getProgram(tid)
                except Exception:
                    curr_prog = None
                try:
                    logics = traci.trafficlight.getAllProgramLogics(tid)
                except Exception:
                    logics = []

                logic = None
                if logics:
                    if curr_prog is not None:
                        for L in logics:
                            pid = L.getProgramID() if hasattr(L, "getProgramID") else getattr(L, "programID", None)
                            if pid == curr_prog:
                                logic = L
                                break
                    if logic is None:
                        logic = logics[0]

                phase_count = 0
                max_dur = 60.0
                green_phases = set()
                durations: List[float] = []
                greens: List[bool] = []
                cum_durs: List[float] = []
                cycle_len = 0.0
                if logic is not None:
                    phases = logic.getPhases() if hasattr(logic, "getPhases") else getattr(logic, "phases", [])
                    phase_count = len(phases) if phases else 0
                    if phases:
                        for idx, p in enumerate(phases):
                            try:
                                d = float(getattr(p, "duration", None) or getattr(p, "maxDur", 0.0) or 0.0)
                            except Exception:
                                d = 0.0
                            d = max(d, 0.0)
                            durations.append(d)
                            try:
                                state = getattr(p, "state", "") if hasattr(p, "state") else str(p)
                            except Exception:
                                state = ""
                            # 仅将主绿（'G'）视为对应相位的“绿窗”，避免任何相位都含有次绿('g')导致绿窗始终为真
                            # 说明：SUMO相位字符串中，'G' 表示主绿放行，'g' 常表示次级/许可绿；
                            # 这里采用保守策略，仅以 'G' 判定绿窗，避免协调奖励退化为常数。
                            isg = ('G' in state)
                            greens.append(isg)
                            if isg:
                                green_phases.add(idx)
                        cycle_len = float(sum(durations)) if durations else 0.0
                        try:
                            max_dur = max(durations) if durations else max_dur
                        except Exception:
                            pass
                        # 累积时长
                        s = 0.0
                        for d in durations:
                            s += d
                            cum_durs.append(s)
                if phase_count <= 0:
                    # 退化：仅用当前相位与剩余时间近似
                    try:
                        phase_count = max(1, int(traci.trafficlight.getPhase(tid)) + 1)
                        remain = traci.trafficlight.getNextSwitch(tid) - traci.simulation.getTime()
                        max_dur = max(1.0, float(remain))
                    except Exception:
                        phase_count = 8
                        max_dur = 60.0
                    durations = [max_dur]
                    greens = [False]
                    cycle_len = float(max_dur)
                    green_phases = set()
                    cum_durs = [cycle_len]

                # 写入缓存
                self._tls_phase_meta[tid] = {
                    "phase_count": float(phase_count),
                    "max_phase_duration": float(max_dur),
                }
                self._tls_cycle_meta[tid] = {
                    "durations": durations,
                    "greens": greens,
                    "cycle_len": float(max(cycle_len, 1e-6)),
                    "cum_durations": cum_durs,
                }
                self._green_phase_cache[tid] = green_phases
            except traci.TraCIException:
                self._tls_phase_meta[tid] = {"phase_count": 8.0, "max_phase_duration": 60.0}
                self._tls_cycle_meta[tid] = {
                    "durations": [60.0],
                    "greens": [False],
                    "cycle_len": 60.0,
                    "cum_durations": [60.0],
                }
                self._green_phase_cache[tid] = set()
            except Exception:
                self._tls_phase_meta[tid] = {"phase_count": 8.0, "max_phase_duration": 60.0}
                self._tls_cycle_meta[tid] = {
                    "durations": [60.0],
                    "greens": [False],
                    "cycle_len": 60.0,
                    "cum_durations": [60.0],
                }
                self._green_phase_cache[tid] = set()
    
    def _is_green_phase(self, tls_id: str, phase_idx: int) -> bool:
        """判断指定信号的指定相位是否为绿灯。"""
        return phase_idx in self._green_phase_cache.get(tls_id, set())

    # --------------------------------------------------------
    def _build_segment_tls_map(self) -> None:
        """建立每段对应的下游信号映射。

        修正为方向化与地理序一致的映射：
        - 东向（EB）：上游→tls_ids[0]，中游→tls_ids[1]，下游→tls_ids[last]
        - 西向（WB）：上游→tls_ids[last]，中游→tls_ids[last-1]，下游→tls_ids[0]
        若信号数量不足则使用边界索引兜底。
        """
        ids = list(self.tls_ids or [])
        if not ids:
            return
        last_idx = len(ids) - 1
        for i, g in enumerate(self.group_keys):
            # EB 按自然顺序映射
            tls_id_eb = ids[min(i, last_idx)]
            # WB 反向映射（走廊方向相反）
            tls_id_wb = ids[max(last_idx - i, 0)]
            self.segment_tls_map.setdefault("eb", {})[g] = tls_id_eb
            self.segment_tls_map.setdefault("wb", {})[g] = tls_id_wb

    # --------------------------------------------------------
    def _tick_coupling_cooldowns(self) -> None:
        """每个决策步调用一次：钳制事件的冷却计数器递减。"""
        try:
            for k in list(self._clamp_event_cooldowns.keys()):
                v = int(self._clamp_event_cooldowns.get(k, 0) or 0)
                if v > 0:
                    self._clamp_event_cooldowns[k] = v - 1
                else:
                    # 清零后删除，避免膨胀
                    self._clamp_event_cooldowns.pop(k, None)
        except Exception:
            pass

    def _register_coupling_clamp_attempt(self, direction: str, segment: str) -> None:
        """记录一次钳制尝试，应用去抖与冷却：

        - 在冷却期内不计事件（pending 清零）。
        - 连续≥ debounce_steps 次尝试后计 1 次事件，并进入冷却期。
        """
        try:
            key = (str(direction), str(segment))
            cool = int(self._clamp_event_cooldowns.get(key, 0) or 0)
            if cool > 0:
                # 冷却期：不累计，不计事件
                self._clamp_event_pending[key] = 0
                return
            cnt = int(self._clamp_event_pending.get(key, 0) or 0) + 1
            self._clamp_event_pending[key] = cnt
            if cnt >= int(self.clamp_event_debounce_steps):
                # 计一次事件并启动冷却
                self.metrics["coupling_clamp_events"] = int(self.metrics.get("coupling_clamp_events", 0) or 0) + 1
                self._clamp_event_pending[key] = 0
                self._clamp_event_cooldowns[key] = int(self.clamp_event_cooldown_steps)
        except Exception:
            # 安全失败：若状态异常，不影响主流程
            pass

    def _apply_signal_speed_coupling(self, target_speeds: Tuple[float, ...]) -> Tuple[float, ...]:
        """在车辆施加前，依据信号绿窗、排队与需求，对每段目标速度进行耦合约束。"""
        # 确保有最新的分段指标（含占有率）
        lm = self._step_cache.get("lane_metrics")
        if lm is None:
            try:
                self._collect_lane_group_metrics(need_occ=True)
            except Exception:
                pass
            lm = self._step_cache.get("lane_metrics", {"eb": {}, "wb": {}})

        # 初始化本步的信号窗口标志缓存
        try:
            signal_flags = self._step_cache.setdefault("signal_window_flags", {"eb": {}, "wb": {}})
        except Exception:
            signal_flags = {"eb": {}, "wb": {}}

        prev = np.array(self._last_action_speeds, dtype=float)
        ts = np.array(list(target_speeds), dtype=float)
        orig_ts = np.array(list(target_speeds), dtype=float)
        base_delta = float(self.max_delta_kph) / 3.6
        # 全局需求作为兜底
        demand_global = float(self.metrics.get("demand_trend_norm", 0.0))
        # 动态慢速阈值：取动作空间下限与参考速度比例的更大者
        try:
            slow_candidates = []
            if self.speed_levels_mps:
                slow_candidates.append(float(min(self.speed_levels_mps)))
            slow_candidates.append(0.4 * float(self.max_ref_speed_mps))
            slow_mps = float(max(slow_candidates)) if slow_candidates else 12.0
        except Exception:
            slow_mps = 12.0
        occ_thr = 0.30
        h_thr = float(self.queue_overflow_threshold)

        for direction, offset in (("eb", 0), ("wb", 3)):
            # 计算下游拥堵信号用于级联抑制（溢出/占有率）
            try:
                down_vals = lm.get(direction, {}).get("down", {})
                avg_halts_down = float(down_vals.get("halts", 0.0))
                upper_halts_down = self._get_halts_upper_for_segment(direction, "down")
                halts_norm_down = max(0.0, min(avg_halts_down / float(max(upper_halts_down, 1e-6)), 1.0))
                occ_down = float(down_vals.get("occ", 0.0))
                down_congested = bool((halts_norm_down > h_thr) or (occ_down > occ_thr))
            except Exception:
                down_congested = False

            for seg_idx, seg in enumerate(self.group_keys):
                i = offset + seg_idx
                tls_id = self.segment_tls_map.get(direction, {}).get(seg)
                if not tls_id:
                    continue
                # 实时信号状态
                try:
                    t_now = float(traci.simulation.getTime())
                except Exception:
                    t_now = 0.0
                try:
                    phase_idx = int(traci.trafficlight.getPhase(tls_id))
                    next_switch = float(traci.trafficlight.getNextSwitch(tls_id))
                    remain_curr = max(0.0, next_switch - t_now)
                    is_now_green = bool(self._is_green_phase(tls_id, phase_idx))
                except Exception:
                    remain_curr = 0.0
                    is_now_green = False

                # 绿灯窗口（归一值 → 真实秒）
                try:
                    is_green_now, start_norm, dur_norm = self._get_tls_green_window_features(tls_id)
                except Exception:
                    is_green_now, start_norm, dur_norm = (0.0, 1.0, 0.0)
                meta = self._tls_cycle_meta.get(tls_id, {})
                cycle_len = float(meta.get("cycle_len", 60.0))
                start_sec = float(start_norm) * cycle_len
                dur_sec = float(dur_norm) * cycle_len

                # 记录信号窗口标志（is_green_now/near_green/start_sec/dur_sec/remain_curr）
                try:
                    near_green = bool(start_sec <= float(self.near_green_window_seconds))
                    signal_flags.setdefault(direction, {})[seg] = {
                        "is_green_now": bool(is_now_green),
                        "near_green": bool(near_green),
                        "start_sec": float(start_sec),
                        "dur_sec": float(dur_sec),
                        "remain_curr": float(remain_curr),
                    }
                except Exception:
                    pass

                # 段拥挤与溢出
                vals = lm.get(direction, {}).get(seg, {})
                avg_halts = float(vals.get("halts", 0.0))
                upper_halts = self._get_halts_upper_for_segment(direction, seg)
                halts_norm = max(0.0, min(avg_halts / float(max(upper_halts, 1e-6)), 1.0))
                occ = float(vals.get("occ", 0.0))

                # 红灯慢行与追赶策略（按方向化需求）
                if is_green_now < 1.0:
                    demand_dir = float(self.metrics.get("demand_trend_norm_eb" if direction == "eb" else "demand_trend_norm_wb", demand_global))
                    slow_cap = slow_mps * (1.0 - 0.25 * min(max(demand_dir, 0.0), 1.0))
                    near_green = (start_sec <= float(self.near_green_window_seconds))
                    before_ts = ts[i]
                    if near_green:
                        denom = float(max(self.near_green_window_seconds, 1e-6))
                        alpha = max(0.0, min(1.0 - (start_sec / denom), 1.0))
                        cap = slow_cap + alpha * (self.max_ref_speed_mps - slow_cap)
                        ts[i] = min(ts[i], cap)
                    else:
                        ts[i] = min(ts[i], slow_cap)
                    # 记录一次“耦合限制”事件（若降低了目标速度）
                    try:
                        if ts[i] < before_ts - 1e-9:
                            self._register_coupling_clamp_attempt(direction, seg)
                        
                    except Exception:
                        pass

                # 溢出抑制：占有率或 halts_norm 超阈，对上游更严格限速
                if (halts_norm > h_thr) or (occ > occ_thr):
                    before_ts = ts[i]
                    ts[i] = min(ts[i], max(prev[i], slow_mps))
                    local_delta = base_delta * 0.5
                    diff = float(ts[i] - prev[i])
                    ts[i] = prev[i] + np.sign(diff) * min(abs(diff), local_delta)
                    # 拥挤/占有率过阈：同时收紧本段变幅系数
                    try:
                        # 在后续统一应用前，先标记需要收紧
                        signal_flags.setdefault(direction, {}).setdefault(seg, {})["_lf_reduce_due_to_congestion"] = True
                    except Exception:
                        pass
                    # 记录一次“耦合限制”事件（若降低了目标速度）
                    try:
                        if ts[i] < before_ts - 1e-9:
                            self._register_coupling_clamp_attempt(direction, seg)
                    except Exception:
                        pass
                # 面向诊断的拥挤/占有率附加标志
                try:
                    sf = signal_flags.setdefault(direction, {}).setdefault(seg, {})
                    sf["halts_norm"] = float(halts_norm)
                    sf["occ"] = float(occ)
                except Exception:
                    pass

                # 级联抑制：若下游拥堵，收紧上游与中游的目标速度与变幅
                local_factor = 1.0
                if down_congested and seg != "down":
                    before_ts_cascade = ts[i]
                    ts[i] = min(ts[i], max(prev[i], slow_mps))
                    # 记录一次“耦合限制”事件（若降低了目标速度）
                    try:
                        if ts[i] < before_ts_cascade - 1e-9:
                            self._register_coupling_clamp_attempt(direction, seg)
                    except Exception:
                        pass
                    # 同时收紧变幅系数
                    local_factor *= 0.5

                # 若本段出现拥挤/占有率超阈，适度收紧变幅系数（诊断标记由上方溢出抑制处置逻辑设置）
                try:
                    if signal_flags.get(direction, {}).get(seg, {}).get("_lf_reduce_due_to_congestion", False):
                        local_factor *= 0.7
                except Exception:
                    pass

                # 绿窗不足则平滑下降：避免中段突然停车
                def _seg_len_avg(direction: str, segment: str) -> float:
                    lane_ids = (self.lane_groups if direction == "eb" else self.wb_lane_groups).get(segment, [])
                    if not lane_ids:
                        return 0.0
                    lengths: List[float] = []
                    for lid in lane_ids:
                        try:
                            L = float(self._lane_length_cache.get(lid)) if lid in self._lane_length_cache else float(traci.lane.getLength(lid))
                            self._lane_length_cache[lid] = L
                            lengths.append(L)
                        except Exception:
                            pass
                    return float(np.mean(lengths)) if lengths else 0.0

                if is_now_green and dur_sec > 0.0:
                    L = _seg_len_avg(direction, seg)
                    if L > 0.0 and ts[i] > 0.1:
                        pass_time = L / float(max(ts[i], 1e-3))
                        if dur_sec < pass_time:
                            before_ts = ts[i]
                            ts[i] = max(slow_mps, min(ts[i], (slow_mps + ts[i]) * 0.5))
                            # 记录一次“耦合限制”事件（若降低了目标速度）
                            try:
                                if ts[i] < before_ts - 1e-9:
                                    self._register_coupling_clamp_attempt(direction, seg)
                            except Exception:
                                pass

                # 信号切换下的变幅协同
                prev_green = bool(self._prev_tls_green_state.get(tls_id, False))
                # 若前面因级联收紧已调整 local_factor，则在此基础上再叠加切换策略
                # 默认 local_factor=1.0（已在上方初始化/可能被下游拥堵收紧）
                if (not prev_green) and is_now_green:
                    # 红→绿：临时放宽变幅以尽快恢复
                    local_factor = max(local_factor, 2.0)
                elif prev_green and (not is_now_green) and (remain_curr <= float(self.decision_interval)):
                    # 绿→红且剩余很短：收紧变幅，避免突升
                    local_factor = min(local_factor, 0.5)

                # 临近绿窗（红灯阶段且下一绿窗即将到来）：轻微放宽变幅以提前平滑过渡
                try:
                    if signal_flags.get(direction, {}).get(seg, {}).get("near_green", False) and (is_green_now < 1.0):
                        local_factor = max(local_factor, 1.25)
                except Exception:
                    pass

                # 应用方向化局部因子缩放
                dir_scale = float(self.local_factor_dir_scale.get(direction, 1.0))
                local_factor_applied = float(local_factor * dir_scale)
                diff = float(ts[i] - prev[i])
                ts[i] = prev[i] + np.sign(diff) * min(abs(diff), base_delta * local_factor_applied)
                # 累计记录局部因子（用于 episode 级均值评估）
                try:
                    self.metrics["local_factor_sum"] = float(self.metrics.get("local_factor_sum", 0.0) or 0.0) + float(local_factor_applied)
                    self.metrics["local_factor_steps"] = int(self.metrics.get("local_factor_steps", 0) or 0) + 1
                except Exception:
                    pass
                # 记录最近一次方向/分段的局部因子以供观测
                try:
                    self.last_local_factor.setdefault(direction, {})[seg] = float(local_factor_applied)
                except Exception:
                    pass
                self._prev_tls_green_state[tls_id] = bool(is_now_green == 1.0)

        return tuple(float(x) for x in ts.tolist())

    # --------------------------------------------------------
    def _compute_halts_upper_bounds(self) -> None:
        """按车道长度与拥挤间距估算各分段的 halts 物理上限（每车道队列容量的分段平均）。"""
        # 东向
        for gname in self.group_keys:
            lanes = self.lane_groups.get(gname, [])
            capacities: List[float] = []
            for lid in lanes:
                try:
                    L = float(self._lane_length_cache.get(lid)) if lid in self._lane_length_cache else float(traci.lane.getLength(lid))
                    self._lane_length_cache[lid] = L
                    if L > 0.0 and self.jam_spacing_m > 0.0:
                        capacities.append(float(L) / float(self.jam_spacing_m))
                except Exception:
                    pass
            upper = float(np.mean(capacities)) if capacities else float(self.halts_fixed_upper)
            self._halts_upper_eb[gname] = float(max(upper, 1.0))
        # 西向
        for gname in self.group_keys:
            lanes = self.wb_lane_groups.get(gname, [])
            capacities: List[float] = []
            for lid in lanes:
                try:
                    L = float(self._lane_length_cache.get(lid)) if lid in self._lane_length_cache else float(traci.lane.getLength(lid))
                    self._lane_length_cache[lid] = L
                    if L > 0.0 and self.jam_spacing_m > 0.0:
                        capacities.append(float(L) / float(self.jam_spacing_m))
                except Exception:
                    pass
            upper = float(np.mean(capacities)) if capacities else float(self.halts_fixed_upper)
            self._halts_upper_wb[gname] = float(max(upper, 1.0))

    def _get_halts_upper_for_segment(self, direction: str, segment: str) -> float:
        """根据策略返回指定方向/分段的 halts 归一上限。"""
        base_len_upper = None
        try:
            if direction == "eb":
                base_len_upper = self._halts_upper_eb.get(segment, None)
            else:
                base_len_upper = self._halts_upper_wb.get(segment, None)
        except Exception:
            base_len_upper = None
        base_len_upper = float(base_len_upper) if base_len_upper is not None else float(self.halts_fixed_upper)

        # rolling p95
        try:
            hist = (self._halts_hist_eb if direction == "eb" else self._halts_hist_wb).get(segment, deque(maxlen=self._halts_window_steps))
            vals = list(hist)
            p95 = float(np.percentile(vals, 95.0)) if vals else None
        except Exception:
            p95 = None

        strat = (self.halts_upper_strategy or "by_lane_length_and_p95").lower()
        if strat == "fixed":
            upper = float(self.halts_fixed_upper)
        elif strat == "by_lane_length":
            upper = float(base_len_upper)
        elif strat == "rolling_p95":
            upper = float(p95) if p95 is not None else float(self.halts_fixed_upper)
        else:  # by_lane_length_and_p95
            if p95 is not None:
                upper = float(max(base_len_upper, float(p95)))
            else:
                upper = float(base_len_upper)
        return float(max(upper, 1e-6))

    def _get_ref_throughput_vph(self) -> float:
        """根据策略返回走廊吞吐归一参考（veh/h）。默认为 Rolling P95，自适应当前信号与到达。"""
        # 优先使用自动识别的出口车道数，否则用下游分段车道数
        try:
            exit_lane_count = int(len(self._corridor_exit_lanes))
        except Exception:
            exit_lane_count = 0
        if exit_lane_count <= 0:
            try:
                exit_lane_count = int(max(len(self.lane_groups.get("down", [])) + len(self.wb_lane_groups.get("down", [])), 1))
            except Exception:
                exit_lane_count = 1

        strat = (self.throughput_ref_strategy or "rolling_p95").lower()
        if strat == "fixed_per_lane_1800":
            ref_vph = float(self.saturation_flow_per_lane_vphg) * float(exit_lane_count) * float(max(min(self.progression_factor, 1.0), 0.0))
            return float(max(ref_vph, 1e-6))

        # rolling_p95：用过去窗口的到达率（veh/s）95分位
        try:
            vals = list(self._throughput_ref_hist)
            p95_per_s = float(np.percentile(vals, 95.0)) if vals else None
            if p95_per_s is not None:
                ref_vph = float(p95_per_s * 3600.0)
            else:
                # 兜底：固定每车道1800并乘出口车道数
                ref_vph = float(self.saturation_flow_per_lane_vphg) * float(exit_lane_count)
        except Exception:
            ref_vph = float(self.saturation_flow_per_lane_vphg) * float(exit_lane_count)
        return float(max(ref_vph, 1e-6))

    # --------------------------------------------------------
    def _make_run_output_dir(self) -> str:
        """创建并返回运行时输出目录：<repo_root>/outputs/YYYYMMDD_HHMMSS"""
        proj_root = os.path.abspath(os.path.join(os.path.dirname(self.sumo_cfg_path), ".."))
        out_root = os.path.join(proj_root, "outputs")
        os.makedirs(out_root, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(out_root, stamp)
        os.makedirs(run_dir, exist_ok=True)
        return run_dir

    def _get_sumocfg_additional_files_value(self) -> Optional[str]:
        try:
            with open(self.sumo_cfg_path, "r", encoding="utf-8") as f:
                for line in f:
                    if "<additional-files" in line and "value=" in line:
                        q1 = line.find('"')
                        q2 = line.find('"', q1 + 1)
                        if q1 != -1 and q2 != -1:
                            return line[q1 + 1:q2]
        except Exception:
            return None
        return None

    def _write_run_additional_file(self) -> Optional[str]:
        """生成 additional.generated.xml，覆盖所有 laneAreaDetector 的 file 指向运行目录。"""
        add_rel = self._get_sumocfg_additional_files_value()
        if not add_rel:
            return None
        base_dir = os.path.dirname(self.sumo_cfg_path)
        src_path = os.path.join(base_dir, add_rel)
        try:
            with open(src_path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception:
            return None
        e2_out_path = os.path.join(self.run_output_dir, "grid1x3_e2.out.xml")
        import re
        content_new = re.sub(r'file="[^\"]+"', f'file="{e2_out_path}"', content)
        out_path = os.path.join(self.run_output_dir, "additional.generated.xml")
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(content_new)
            return out_path
        except Exception:
            return None

    # --------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if traci.isLoaded():
            traci.close()
        self._init_sumo()

        # 重置仿真时间累计（使用 SUMO 的真实时间）
        self.sim_time = 0.0
        self._prev_time_s = 0.0
        self._last_action_index = -1
        init_speed = (self.speed_levels_mps[0] if self.speed_levels_mps else 20.0)
        self._last_action_speeds = (init_speed,) * 6
        self._prev_segment_speeds_norm = {"eb": {}, "wb": {}}
        self._arrival_rate_hist.clear()
        # 方向化到达率窗口清空
        try:
            self._arrival_rate_hist_eb.clear()
            self._arrival_rate_hist_wb.clear()
        except Exception:
            pass
        try:
            self._arrival_rate_hist_slow.clear()
        except Exception:
            pass
        try:
            self._arrival_rate_hist_slow_eb.clear()
            self._arrival_rate_hist_slow_wb.clear()
        except Exception:
            pass
        # 清空走廊出口追踪集合
        try:
            self._corridor_exit_seen.clear()
        except Exception:
            self._corridor_exit_seen = set()
        # 方向化集合清空
        try:
            self._corridor_exit_seen_eb.clear()
            self._corridor_exit_seen_wb.clear()
        except Exception:
            self._corridor_exit_seen_eb = set()
            self._corridor_exit_seen_wb = set()

        # 清空 rolling 历史并更新分段物理上限
        try:
            self._throughput_ref_hist.clear()
        except Exception:
            pass
        try:
            for g in self.group_keys:
                try:
                    self._halts_hist_eb.get(g, deque()).clear()
                    self._halts_hist_wb.get(g, deque()).clear()
                except Exception:
                    pass
        except Exception:
            pass
        # 依赖 TraCI 的车道长度：需在 SUMO 初始化后计算
        try:
            self._compute_halts_upper_bounds()
        except Exception as e:
            print(f"[warn] 计算 halts 物理上限失败：{e}")

        # 清空评估累计（保持与 __init__ 中的指标键一致，并新增方向化与耦合统计键）
        self.metrics = {
            # 基础累计
            "queue_sum": 0.0,
            "queue_steps": 0,
            "arrived_total": 0,
            "arrived_total_eb": 0,
            "arrived_total_wb": 0,
            "sim_seconds": 0,
            # 归一化指标累计（用于 episode 汇总）
            "delay_sum_norm": 0.0,
            "delay_steps": 0,
            "stops_sum_norm": 0.0,
            "stops_steps": 0,
            "smooth_sum_norm": 0.0,
            "smooth_steps": 0,
            # 需求趋势（快/慢）
            "arrival_rate_vph_ma": 0.0,
            "arrival_rate_vph_ma_eb": 0.0,
            "arrival_rate_vph_ma_wb": 0.0,
            "demand_trend_norm": 0.0,
            "demand_trend_norm_eb": 0.0,
            "demand_trend_norm_wb": 0.0,
            "arrival_rate_vph_ma_5min": 0.0,
            "arrival_rate_vph_ma_5min_eb": 0.0,
            "arrival_rate_vph_ma_5min_wb": 0.0,
            "demand_trend_slow_norm": 0.0,
            "demand_trend_slow_norm_eb": 0.0,
            "demand_trend_slow_norm_wb": 0.0,
            # 耦合统计
            "coupling_clamp_events": 0,
            "local_factor_sum": 0.0,
            "local_factor_steps": 0,
            # 奖励分量累计（用于episode均值）
            "reward_component_steps": 0,
            "r_efficiency_sum": 0.0,
            "r_throughput_sum": 0.0,
            "r_coordination_sum": 0.0,
            "r_congestion_sum": 0.0,
            "r_delay_sum": 0.0,
            "r_queue_overflow_sum": 0.0,
            "r_stops_sum": 0.0,
            "r_fuel_sum": 0.0,
            "r_smoothness_sum": 0.0,
            "r_demand_robust_sum": 0.0,
            "r_signal_compliance_sum": 0.0,
            "r_green_pass_sum": 0.0,
        }
        self._controlled_veh_ids.clear()
        self._veh_type_max_speed.clear()
        # 清空钳制事件去抖与冷却状态
        try:
            self._clamp_event_pending.clear()
        except Exception:
            self._clamp_event_pending = {}
        try:
            self._clamp_event_cooldowns.clear()
        except Exception:
            self._clamp_event_cooldowns = {}

        # 暖机步数：推进仿真但不计入指标
        for _ in range(self.warmup_steps):
            traci.simulationStep()
        # 暖机完成后，以 SUMO 当前时间初始化累计
        try:
            self.sim_time = float(traci.simulation.getTime())
        except Exception:
            self.sim_time = float(self.warmup_steps)
        self._prev_time_s = self.sim_time

        obs = self._observe()
        return obs, {}

    # --------------------------------------------------------
    def step(self, action):
        """
        标准step方法，包含基本的错误处理和内存管理
        
        主要改进：
        1. 基本TraCI异常处理
        2. 内存泄漏防护和性能监控
        3. 动作历史记录（有限长度）
        """
        # 0. 决策步开始：更新钳制事件冷却计数器（每决策步一次）
        try:
            self._tick_coupling_cooldowns()
        except Exception:
            pass

        # 1. MultiDiscrete 动作解析与越界保护：期望长度为 6 的整数向量
        try:
            indices = np.array(action, dtype=np.int64).reshape(-1)
        except Exception as e:
            print(f"[错误] 动作解析失败: {e}")
            indices = np.zeros(6, dtype=np.int64)
        if indices.size != 6:
            indices = np.pad(indices, (0, max(0, 6 - indices.size)), mode="constant")[:6]
        n_per_segment = len(self.speed_levels_mps) if self.speed_levels_mps else 1
        indices = np.clip(indices, 0, max(0, n_per_segment - 1))

        # 2. 取目标速度（东向3段 + 西向3段），按档位映射到 m/s
        #    indices[0:3] -> EB 三段；indices[3:6] -> WB 三段
        try:
            target_speeds = [self.speed_levels_mps[int(i)] for i in indices.tolist()]  # m/s
        except Exception as e:
            print(f"[错误] 速度映射失败: {e}")
            target_speeds = [20.0] * 6  # 默认速度

        # 3. 做论文式"限速变化≤Δv"的硬约束（按段，6 段）
        max_delta_mps = self.max_delta_kph / 3.6
        prev_speeds = self._last_action_speeds
        clipped_speeds = []
        for prev_v, new_v in zip(prev_speeds, target_speeds):
            if abs(new_v - prev_v) > max_delta_mps:
                if new_v > prev_v:
                    new_v = prev_v + max_delta_mps
                else:
                    new_v = prev_v - max_delta_mps
            clipped_speeds.append(new_v)
        target_speeds = tuple(clipped_speeds)

        # 3.1 信号耦合：在车辆施加前，根据红绿灯窗口与拥挤、需求，对目标速度做约束
        try:
            target_speeds = self._apply_signal_speed_coupling(target_speeds)
        except Exception as e:
            print(f"[警告] 信号限速耦合失败: {e}")

        ep_reward = 0.0

        # 重置本决策周期的每步缓存
        self._step_cache = {}

        # 4. 决策周期内：以目标时间 self.sim_time + decision_interval 为准，逐步推进至该目标
        #    这样避免依赖 deltaT 的不确定性（某些平台返回异常值导致步数过大）
        target_end_time = float(self.sim_time) + float(self._decision_interval_s)
        safety_iter = 0
        while True:
            if float(self.sim_time) >= target_end_time:
                break
            safety_iter += 1
            if safety_iter > 10000:
                # 安全退出：防止异常 deltaT 或仿真停滞导致死循环
                print(f"[警告] 仿真步数超限: {safety_iter}, 提前结束决策周期")
                break
            
            # 4.1 下控制：只控 CAV；东向与西向各用自己的3段速度
            eb_speeds = target_speeds[0:3]
            wb_speeds = target_speeds[3:6]
            curr_controlled = set()
            
            # 东向：分段应用限速，同时记录进入控制集的车辆并缓存其类型默认最大速度
            controlled_count_eb = 0
            for gname, target_speed in zip(self.group_keys, eb_speeds):
                for lid in self.lane_groups.get(gname, []):
                    try:
                        vehs = traci.lane.getLastStepVehicleIDs(lid)
                    except Exception:
                        vehs = []
                    
                    for vid in vehs:
                        try:
                            if traci.vehicle.getTypeID(vid) == self.cav_type_id:
                                curr_controlled.add(vid)
                                controlled_count_eb += 1
                                if vid not in self._veh_type_max_speed:
                                    try:
                                        vtype = traci.vehicle.getTypeID(vid)
                                        tmax = float(traci.vehicletype.getMaxSpeed(vtype))
                                    except Exception:
                                        try:
                                            tmax = float(traci.vehicle.getMaxSpeed(vid))
                                        except Exception:
                                            tmax = float(target_speed)
                                    self._veh_type_max_speed[vid] = tmax
                                traci.vehicle.setMaxSpeed(vid, float(target_speed))
                        except Exception:
                            pass  # 简化错误处理，忽略单个车辆设置失败
            
            # 西向：同理
            controlled_count_wb = 0
            for gname, target_speed in zip(self.group_keys, wb_speeds):
                for lid in self.wb_lane_groups.get(gname, []):
                    try:
                        vehs = traci.lane.getLastStepVehicleIDs(lid)
                    except Exception:
                        vehs = []
                    
                    for vid in vehs:
                        try:
                            if traci.vehicle.getTypeID(vid) == self.cav_type_id:
                                curr_controlled.add(vid)
                                controlled_count_wb += 1
                                if vid not in self._veh_type_max_speed:
                                    try:
                                        vtype = traci.vehicle.getTypeID(vid)
                                        tmax = float(traci.vehicletype.getMaxSpeed(vtype))
                                    except Exception:
                                        try:
                                            tmax = float(traci.vehicle.getMaxSpeed(vid))
                                        except Exception:
                                            tmax = float(target_speed)
                                    self._veh_type_max_speed[vid] = tmax
                                traci.vehicle.setMaxSpeed(vid, float(target_speed))
                        except Exception:
                            pass  # 简化错误处理，忽略单个车辆设置失败
            
            # 调试输出：每100步打印一次控制信息
            # 4.1.1 恢复：离开控制区的车辆恢复默认类型最大速度，避免"限速残留"
            exited = self._controlled_veh_ids - curr_controlled
            for vid in list(exited):
                try:
                    default_v = float(self._veh_type_max_speed.pop(vid, 27.78))
                    traci.vehicle.setMaxSpeed(vid, default_v)
                except traci.TraCIException as e:
                    print(f"[TraCI异常] 恢复车辆速度失败(vid={vid}): {e}")
                except Exception as e:
                    print(f"[异常] 恢复车辆速度失败(vid={vid}): {e}")
                finally:
                    self._controlled_veh_ids.discard(vid)
            # 更新当前控制集
            self._controlled_veh_ids = curr_controlled

            # 推进 1 个仿真步（基本错误处理）
            try:
                traci.simulationStep()
            except Exception as e:
                print(f"[警告] 仿真步失败: {e}")
                # 简单错误处理，不尝试重启
                break
            
            # 更新时间与步长（秒）
            try:
                t_now = float(traci.simulation.getTime())
            except Exception:
                t_now = self.sim_time + 1.0  # 使用估计时间
            
            dt_s = max(0.0, t_now - self._prev_time_s)
            self.sim_time = t_now
            self._prev_time_s = t_now

            # —— 评估累计：队列、能耗、距离、到达、时间损失 ——
            # 1) 队列：各段车道的"停止车辆数"（双向合并）
            queue_count = 0.0
            for gname in self.group_keys:
                lane_ids = self.lane_groups.get(gname, []) + self.wb_lane_groups.get(gname, [])
                for lid in lane_ids:
                    try:
                        queue_count += float(traci.lane.getLastStepHaltingNumber(lid))
                    except Exception:
                        pass  # 简化错误处理，忽略单个检测器失败
            self.metrics["queue_sum"] += queue_count
            self.metrics["queue_steps"] += 1

            # 2) 车辆能耗 / 距离 / 时间损失（删除，不参与奖励计算）

            # 3) 本步到达车辆：仅计数
            try:
                arrived_ids = traci.simulation.getArrivedIDList()
            except Exception:
                arrived_ids = []
                
            for vid in arrived_ids:
                self.metrics["arrived_total"] += 1

            # 3.1) 走廊到达：匹配上一步在走廊出口车道上的车辆
            try:
                arrived_corridor = [vid for vid in arrived_ids if vid in self._corridor_exit_seen]
            except Exception:
                arrived_corridor = []
            # 清理已到达的车辆标记，避免集合膨胀
            for vid in arrived_corridor:
                try:
                    self._corridor_exit_seen.discard(vid)
                except Exception:
                    pass

            # 3.1.1) 方向化到达：EB/WB 分别统计并清理方向集合
            try:
                arrived_corridor_eb = [vid for vid in arrived_ids if vid in self._corridor_exit_seen_eb]
            except Exception:
                arrived_corridor_eb = []
            try:
                arrived_corridor_wb = [vid for vid in arrived_ids if vid in self._corridor_exit_seen_wb]
            except Exception:
                arrived_corridor_wb = []
            for vid in arrived_corridor_eb:
                try:
                    self._corridor_exit_seen_eb.discard(vid)
                except Exception:
                    pass
            for vid in arrived_corridor_wb:
                try:
                    self._corridor_exit_seen_wb.discard(vid)
                except Exception:
                    pass

            # 4) 仿真时长累计（秒）
            self.metrics["sim_seconds"] += dt_s

            # 需求趋势（仅走廊范围）：到达流率（veh/s）滑动均值（归一为 veh/h / ref_vph）
            try:
                inst_rate_per_s = float(len(arrived_corridor)) / float(max(dt_s, 1e-6))
            except Exception:
                inst_rate_per_s = 0.0
            self._arrival_rate_hist.append(inst_rate_per_s)
            self._arrival_rate_hist_slow.append(inst_rate_per_s)
            try:
                self._throughput_ref_hist.append(inst_rate_per_s)
            except Exception:
                pass
            # —— 方向化需求趋势（EB/WB）——
            try:
                inst_rate_per_s_eb = float(len(arrived_corridor_eb)) / float(max(dt_s, 1e-6))
            except Exception:
                inst_rate_per_s_eb = 0.0
            try:
                inst_rate_per_s_wb = float(len(arrived_corridor_wb)) / float(max(dt_s, 1e-6))
            except Exception:
                inst_rate_per_s_wb = 0.0
            try:
                self._arrival_rate_hist_eb.append(inst_rate_per_s_eb)
                self._arrival_rate_hist_wb.append(inst_rate_per_s_wb)
                self._arrival_rate_hist_slow_eb.append(inst_rate_per_s_eb)
                self._arrival_rate_hist_slow_wb.append(inst_rate_per_s_wb)
            except Exception:
                pass
            try:
                avg_rate_per_s = float(np.mean(list(self._arrival_rate_hist))) if self._arrival_rate_hist else 0.0
            except Exception:
                avg_rate_per_s = 0.0
            avg_rate_vph = avg_rate_per_s * 3600.0
            ref_throughput_vph = self._get_ref_throughput_vph()
            avg_norm = max(0.0, min(avg_rate_vph / float(max(ref_throughput_vph, 1e-6)), 1.0))
            inst_norm = max(0.0, min((inst_rate_per_s * 3600.0) / float(max(ref_throughput_vph, 1e-6)), 1.0))
            w = float(max(0.0, min(self._trend_inst_fallback_w, 1.0)))
            trend_norm = max(0.0, min(((1.0 - w) * avg_norm + w * inst_norm), 1.0))
            self.metrics["arrival_rate_vph_ma"] = avg_rate_vph
            self.metrics["demand_trend_norm"] = trend_norm
            self.metrics["ref_throughput_vph"] = ref_throughput_vph

            # 方向化快趋势（EB/WB）：按出口车道占比分配参考通行能力
            try:
                avg_rate_per_s_eb = float(np.mean(list(self._arrival_rate_hist_eb))) if self._arrival_rate_hist_eb else 0.0
            except Exception:
                avg_rate_per_s_eb = 0.0
            try:
                avg_rate_per_s_wb = float(np.mean(list(self._arrival_rate_hist_wb))) if self._arrival_rate_hist_wb else 0.0
            except Exception:
                avg_rate_per_s_wb = 0.0
            avg_rate_vph_eb = avg_rate_per_s_eb * 3600.0
            avg_rate_vph_wb = avg_rate_per_s_wb * 3600.0
            try:
                total_exit = float(max(len(self._corridor_exit_lanes), 1))
                eb_exit = float(len(self._corridor_exit_lanes_eb))
                wb_exit = float(len(self._corridor_exit_lanes_wb))
                ref_total_vph = float(ref_throughput_vph)
                ref_vph_eb = ref_total_vph * (eb_exit / total_exit)
                ref_vph_wb = ref_total_vph * (wb_exit / total_exit)
            except Exception:
                ref_vph_eb = float(ref_throughput_vph)
                ref_vph_wb = float(ref_throughput_vph)
            avg_norm_eb = max(0.0, min(avg_rate_vph_eb / float(max(ref_vph_eb, 1e-6)), 1.0))
            avg_norm_wb = max(0.0, min(avg_rate_vph_wb / float(max(ref_vph_wb, 1e-6)), 1.0))
            inst_norm_eb = max(0.0, min((inst_rate_per_s_eb * 3600.0) / float(max(ref_vph_eb, 1e-6)), 1.0))
            inst_norm_wb = max(0.0, min((inst_rate_per_s_wb * 3600.0) / float(max(ref_vph_wb, 1e-6)), 1.0))
            wdir = float(max(0.0, min(self._trend_inst_fallback_w, 1.0)))
            trend_norm_eb = max(0.0, min(((1.0 - wdir) * avg_norm_eb + wdir * inst_norm_eb), 1.0))
            trend_norm_wb = max(0.0, min(((1.0 - wdir) * avg_norm_wb + wdir * inst_norm_wb), 1.0))
            self.metrics["arrival_rate_vph_ma_eb"] = avg_rate_vph_eb
            self.metrics["arrival_rate_vph_ma_wb"] = avg_rate_vph_wb
            self.metrics["demand_trend_norm_eb"] = trend_norm_eb
            self.metrics["demand_trend_norm_wb"] = trend_norm_wb
            # 方向化慢趋势（5分钟）
            try:
                avg_rate_per_s_slow_eb = float(np.mean(list(self._arrival_rate_hist_slow_eb))) if self._arrival_rate_hist_slow_eb else 0.0
            except Exception:
                avg_rate_per_s_slow_eb = 0.0
            try:
                avg_rate_per_s_slow_wb = float(np.mean(list(self._arrival_rate_hist_slow_wb))) if self._arrival_rate_hist_slow_wb else 0.0
            except Exception:
                avg_rate_per_s_slow_wb = 0.0
            avg_rate_vph_slow_eb = avg_rate_per_s_slow_eb * 3600.0
            avg_rate_vph_slow_wb = avg_rate_per_s_slow_wb * 3600.0
            self.metrics["arrival_rate_vph_ma_5min_eb"] = avg_rate_vph_slow_eb
            self.metrics["arrival_rate_vph_ma_5min_wb"] = avg_rate_vph_slow_wb
            slow_sum_vph = max(avg_rate_vph_slow_eb + avg_rate_vph_slow_wb, 1e-6)
            self.metrics["demand_trend_slow_norm_eb"] = avg_rate_vph_slow_eb / slow_sum_vph
            self.metrics["demand_trend_slow_norm_wb"] = avg_rate_vph_slow_wb / slow_sum_vph
            # 累计方向化到达总数（仅走廊）
            try:
                self.metrics["arrived_total_eb"] += int(len(arrived_corridor_eb))
                self.metrics["arrived_total_wb"] += int(len(arrived_corridor_wb))
            except Exception:
                pass

            # 慢趋势（5分钟滑动窗口）
            try:
                avg_rate_per_s_slow = float(np.mean(list(self._arrival_rate_hist_slow))) if self._arrival_rate_hist_slow else 0.0
            except Exception:
                avg_rate_per_s_slow = 0.0
            avg_rate_vph_slow = avg_rate_per_s_slow * 3600.0
            trend_slow_norm = max(0.0, min(avg_rate_vph_slow / float(max(ref_throughput_vph, 1e-6)), 1.0))
            self.metrics["arrival_rate_vph_ma_5min"] = avg_rate_vph_slow
            self.metrics["demand_trend_slow_norm"] = trend_slow_norm

            # 3.2) 更新走廊出口车道上的车辆集合，用于下一步匹配到达
            try:
                for lid in self._corridor_exit_lanes_eb or []:
                    try:
                        vids_on_exit = traci.lane.getLastStepVehicleIDs(lid)
                    except Exception:
                        vids_on_exit = []
                    for vid in vids_on_exit:
                        try:
                            self._corridor_exit_seen.add(vid)
                            self._corridor_exit_seen_eb.add(vid)
                        except Exception:
                            pass
                for lid in self._corridor_exit_lanes_wb or []:
                    try:
                        vids_on_exit = traci.lane.getLastStepVehicleIDs(lid)
                    except Exception:
                        vids_on_exit = []
                    for vid in vids_on_exit:
                        try:
                            self._corridor_exit_seen.add(vid)
                            self._corridor_exit_seen_wb.add(vid)
                        except Exception:
                            pass
            except Exception:
                pass

            # 计算奖励（基本错误处理）
            try:
                step_reward = self._compute_reward_per_step()
                ep_reward += step_reward
            except Exception:
                ep_reward += 0.0  # 使用默认奖励

        # 5. 决策结束：加一次动作变化惩罚
        try:
            penalty = self._compute_action_change_penalty(tuple(target_speeds))
            ep_reward += penalty
        except Exception:
            pass  # 忽略惩罚计算失败

        # 6. 更新last
        self._last_action_index = 0  # 仅用于"首次不惩罚"的语义
        self._last_action_speeds = tuple(target_speeds)

        # 7. 观测 & done
        try:
            # 在生成观测前补齐占有率到缓存，避免 _observe 重复读取
            try:
                self._collect_lane_group_metrics(need_occ=True)
            except Exception:
                pass
            obs = self._observe()
        except Exception:
            obs = np.zeros(self.obs_dim, dtype=np.float32)  # 返回默认观测
            
        # 以"真实秒"为终止标准
        terminated = float(self.sim_time) >= float(self.max_sim_seconds)
        truncated = False

        # 计算“已执行离散档位”索引（基于最终目标速度与离散速度表的就近匹配）
        exec_levels: List[int] = []
        try:
            if self.speed_levels_mps is not None:
                _levels = np.asarray(self.speed_levels_mps, dtype=np.float32)
                _speeds = np.asarray(tuple(target_speeds), dtype=np.float32)
                _diffs = np.abs(_speeds[:, None] - _levels[None, :])
                exec_levels = np.argmin(_diffs, axis=1).astype(np.int32).tolist()
        except Exception:
            exec_levels = []

        info = {
            "sim_time": self.sim_time,
            "action_speeds_mps": tuple(target_speeds),
            "raw_action_levels": indices.tolist(),
            "executed_action_levels": exec_levels,
            # 暴露奖励分量与权重，便于算法侧按组件或按段进行信用分配
            "reward_components": {
                "efficiency": float(self.metrics.get("r_efficiency", 0.0)),
                "throughput": float(self.metrics.get("r_throughput", 0.0)),
                "coordination": float(self.metrics.get("r_coordination", 0.0)),
                "congestion": float(self.metrics.get("r_congestion", 0.0)),
                "delay": float(self.metrics.get("r_delay", 0.0)),
                "queue_overflow": float(self.metrics.get("r_queue_overflow", 0.0)),
                "stops": float(self.metrics.get("r_stops", 0.0)),
                "fuel": float(self.metrics.get("r_fuel", 0.0)),
                "smoothness": float(self.metrics.get("r_smoothness", 0.0)),
                "demand_robust": float(self.metrics.get("r_demand_robust", 0.0)),
                "signal_compliance": float(self.metrics.get("r_signal_compliance", 0.0)),
                "green_pass": float(self.metrics.get("r_green_pass", 0.0)),
                "total": float(self.metrics.get("r_total", 0.0)),
            },
            "reward_weights": {
                k: float(v) for k, v in (self.rw or {}).items()
            },
            # 当前决策周期各方向/分段的信号窗口标志
            "signal_window_flags": self._step_cache.get("signal_window_flags", {"eb": {}, "wb": {}}),
        }

        return obs, ep_reward, terminated, truncated, info

    # --------------------------------------------------------
    def get_metrics(self) -> Dict[str, float]:
        """
        返回当前累计指标的汇总：
        - avg_queue_veh: 平均排队车辆数（辆）= 每步停止车辆数平均
        - throughput_veh_per_hour: 通行流量（辆/小时）= 到达数 / 时长 * 3600
        """
        q_steps = self.metrics.get("queue_steps", 0)
        sim_sec = self.metrics.get("sim_seconds", 0)
        arrived = self.metrics.get("arrived_total", 0)

        avg_queue = (self.metrics.get("queue_sum", 0.0) / q_steps) if q_steps > 0 else 0.0
        throughput = (arrived / sim_sec * 3600.0) if sim_sec > 0 else 0.0

        # 旅行时间/延误（归一）、停车延迟（归一）、速度波动（归一）
        d_steps = int(self.metrics.get("delay_steps", 0) or 0)
        s_steps = int(self.metrics.get("stops_steps", 0) or 0)
        sm_steps = int(self.metrics.get("smooth_steps", 0) or 0)
        avg_delay_norm = (float(self.metrics.get("delay_sum_norm", 0.0)) / d_steps) if d_steps > 0 else 0.0
        avg_stops_norm = (float(self.metrics.get("stops_sum_norm", 0.0)) / s_steps) if s_steps > 0 else 0.0
        avg_speed_fluct_norm = (float(self.metrics.get("smooth_sum_norm", 0.0)) / sm_steps) if sm_steps > 0 else 0.0

        # 方向化与趋势、耦合统计
        arrived_eb = int(self.metrics.get("arrived_total_eb", 0) or 0)
        arrived_wb = int(self.metrics.get("arrived_total_wb", 0) or 0)
        clamp_events = int(self.metrics.get("coupling_clamp_events", 0) or 0)
        lf_sum = float(self.metrics.get("local_factor_sum", 0.0) or 0.0)
        lf_steps = int(self.metrics.get("local_factor_steps", 0) or 0)
        lf_mean = (lf_sum / float(max(lf_steps, 1))) if lf_steps > 0 else 0.0

        out = {
            "avg_queue_veh": avg_queue,
            "throughput_veh_per_hour": throughput,
            "arrived_total": arrived,
            "arrived_total_eb": arrived_eb,
            "arrived_total_wb": arrived_wb,
            "sim_seconds": sim_sec,
            "avg_delay_norm": avg_delay_norm,
            "avg_stops_norm": avg_stops_norm,
            "avg_speed_fluct_norm": avg_speed_fluct_norm,
            # 需求趋势（全局与方向化）
            "arrival_rate_vph_ma": float(self.metrics.get("arrival_rate_vph_ma", 0.0) or 0.0),
            "demand_trend_norm": float(self.metrics.get("demand_trend_norm", 0.0) or 0.0),
            "arrival_rate_vph_ma_eb": float(self.metrics.get("arrival_rate_vph_ma_eb", 0.0) or 0.0),
            "arrival_rate_vph_ma_wb": float(self.metrics.get("arrival_rate_vph_ma_wb", 0.0) or 0.0),
            "demand_trend_norm_eb": float(self.metrics.get("demand_trend_norm_eb", 0.0) or 0.0),
            "demand_trend_norm_wb": float(self.metrics.get("demand_trend_norm_wb", 0.0) or 0.0),
            "arrival_rate_vph_ma_5min": float(self.metrics.get("arrival_rate_vph_ma_5min", 0.0) or 0.0),
            "arrival_rate_vph_ma_5min_eb": float(self.metrics.get("arrival_rate_vph_ma_5min_eb", 0.0) or 0.0),
            "arrival_rate_vph_ma_5min_wb": float(self.metrics.get("arrival_rate_vph_ma_5min_wb", 0.0) or 0.0),
            "demand_trend_slow_norm": float(self.metrics.get("demand_trend_slow_norm", 0.0) or 0.0),
            "demand_trend_slow_norm_eb": float(self.metrics.get("demand_trend_slow_norm_eb", 0.0) or 0.0),
            "demand_trend_slow_norm_wb": float(self.metrics.get("demand_trend_slow_norm_wb", 0.0) or 0.0),
            # 耦合统计
            "coupling_clamp_events": clamp_events,
            "local_factor_mean": lf_mean,
            # 参考吞吐量（veh/h）
            "ref_throughput_vph": float(self.metrics.get("ref_throughput_vph", 0.0) or 0.0),
        }

        # 奖励分量episode均值
        r_steps = int(self.metrics.get("reward_component_steps", 0) or 0)
        for k in [
            "r_efficiency","r_throughput","r_coordination","r_congestion","r_delay","r_queue_overflow",
            "r_stops","r_fuel","r_smoothness","r_demand_robust","r_signal_compliance","r_green_pass"
        ]:
            s = float(self.metrics.get(f"{k}_sum", 0.0) or 0.0)
            out[f"{k}_avg"] = (s / float(max(r_steps, 1))) if r_steps > 0 else 0.0
        out["reward_component_steps"] = r_steps

        return out

    # --------------------------------------------------------
    def _collect_lane_group_metrics(self, need_occ: bool = True) -> None:
        """采集并缓存每个车道组（东/西向）的平均速度、停止车辆数、占有率。

        参数：
        - need_occ: 是否采集占有率；奖励仅需速度/停止车辆数时可以为 False。
        """
        lm = self._step_cache.setdefault("lane_metrics", {"eb": {}, "wb": {}})

        for g in self.group_keys:
            # 东向
            eb_lanes = self.lane_groups.get(g, [])
            eb_speeds, eb_halts, eb_occs = [], [], []
            for lid in eb_lanes:
                try:
                    eb_speeds.append(traci.lane.getLastStepMeanSpeed(lid))
                except Exception:
                    pass
                try:
                    eb_halts.append(traci.lane.getLastStepHaltingNumber(lid))
                except Exception:
                    pass
                if need_occ:
                    try:
                        eb_occs.append(traci.lane.getLastStepOccupancy(lid))
                    except Exception:
                        pass
            eb_speed_avg = float(np.mean(eb_speeds)) if eb_speeds else 0.0
            eb_halts_avg = float(np.mean(eb_halts)) if eb_halts else 0.0
            data_eb = {"speed": eb_speed_avg, "halts": eb_halts_avg}
            if need_occ:
                data_eb["occ"] = float(np.mean(eb_occs)) if eb_occs else 0.0
            lm["eb"][g] = data_eb

            # 西向
            wb_lanes = self.wb_lane_groups.get(g, [])
            wb_speeds, wb_halts, wb_occs = [], [], []
            for lid in wb_lanes:
                try:
                    wb_speeds.append(traci.lane.getLastStepMeanSpeed(lid))
                except Exception:
                    pass
                try:
                    wb_halts.append(traci.lane.getLastStepHaltingNumber(lid))
                except Exception:
                    pass
                if need_occ:
                    try:
                        wb_occs.append(traci.lane.getLastStepOccupancy(lid))
                    except Exception:
                        pass
            wb_speed_avg = float(np.mean(wb_speeds)) if wb_speeds else 0.0
            wb_halts_avg = float(np.mean(wb_halts)) if wb_halts else 0.0
            data_wb = {"speed": wb_speed_avg, "halts": wb_halts_avg}
            if need_occ:
                data_wb["occ"] = float(np.mean(wb_occs)) if wb_occs else 0.0
            lm["wb"][g] = data_wb

    # --------------------------------------------------------
    def _get_cav_ratio_on_lanes(self, lane_ids: List[str]) -> float:
        total = 0
        cav = 0
        for lid in lane_ids or []:
            try:
                vids = traci.lane.getLastStepVehicleIDs(lid)
            except Exception:
                vids = []
            total += len(vids)
            for vid in vids:
                try:
                    if traci.vehicle.getTypeID(vid) == self.cav_type_id:
                        cav += 1
                except Exception:
                    pass
        if total <= 0:
            return 0.0
        return cav / float(total)

    # --------------------------------------------------------
    def _get_tls_green_window_features(self, tls_id: str) -> Tuple[float, float, float]:
        """返回 (is_green_now, next_green_start_norm, next_green_duration_norm)。

        使用预解析的 _tls_cycle_meta（相位 durations / greens / cycle_len），
        归一化基于一个相位周期（durations 之和）。
        若当前为绿灯，则起始偏移为 0，持续为当前绿窗剩余时间 + 后续连续绿相位持续；
        若当前非绿，则起始偏移为直到下一绿窗的时间，持续为该绿窗的连续绿时长。
        """
        # 当前相位与剩余时间（实时）
        try:
            t_now = float(traci.simulation.getTime())
        except Exception:
            t_now = 0.0
        try:
            phase_idx = int(traci.trafficlight.getPhase(tls_id))
            next_switch = float(traci.trafficlight.getNextSwitch(tls_id))
            remain_curr = max(0.0, next_switch - t_now)
        except Exception:
            phase_idx = 0
            remain_curr = 0.0

        meta = self._tls_cycle_meta.get(tls_id, None)
        if not meta:
            # 回退：无缓存时使用当前相位时长近似
            cycle_len = max(remain_curr, 1.0)
            is_green_now = 1.0 if self._is_green_phase(tls_id, phase_idx) else 0.0
            start_norm = 0.0 if is_green_now == 1.0 else min(remain_curr / cycle_len, 1.0)
            dur_norm = min(remain_curr / cycle_len, 1.0)
            return (is_green_now, start_norm, dur_norm)

        durations: List[float] = list(meta.get("durations", []))
        greens: List[bool] = list(meta.get("greens", []))
        cycle_len: float = float(meta.get("cycle_len", 1.0))
        n = len(durations)
        cycle_len = max(cycle_len, 1e-6)

        # 当前是否绿灯
        is_now_green = bool(greens and 0 <= phase_idx < n and greens[phase_idx])
        is_green_now = 1.0 if is_now_green else 0.0

        if is_now_green:
            # 当前绿窗剩余 + 后续连续绿相位持续
            dur = float(remain_curr)
            if n > 0:
                i = (phase_idx + 1) % n
                while greens[i] and i != phase_idx:
                    dur += float(durations[i])
                    i = (i + 1) % n
            start_norm = 0.0
            dur_norm = max(0.0, min(dur / cycle_len, 1.0))
            return (1.0, start_norm, dur_norm)
        else:
            # 非绿：起始偏移为当前剩余 + 后续直到遇到下一绿的时间；持续为该绿窗连续绿相位之和
            offset = float(remain_curr)
            dur = 0.0
            if n > 0:
                i = (phase_idx + 1) % n
                # 累积到下一个绿相位的等待
                while (not greens[i]) and i != phase_idx:
                    offset += float(durations[i])
                    i = (i + 1) % n
                # 从下一个绿相位开始累积连续绿窗持续
                j = i
                while greens[j] and j != phase_idx:
                    dur += float(durations[j])
                    j = (j + 1) % n
            start_norm = max(0.0, min(offset / cycle_len, 1.0))
            dur_norm = max(0.0, min(dur / cycle_len, 1.0))
            return (0.0, start_norm, dur_norm)

    # --------------------------------------------------------
    def _observe(self) -> np.ndarray:
        feats: List[float] = []

        # 简化版观测：东向3段 + 西向3段，每段仅提供 occ + speed = 2个特征
        curr_occ_eb = {}
        curr_speed_eb = {}
        curr_occ_wb = {}
        curr_speed_wb = {}

        # 东向
        for gname in self.group_keys:
            lane_ids = self.lane_groups.get(gname, [])
            if not lane_ids:
                feats.extend([0.0, 0.0, 0.0])  # occ, speed, halts_norm
                curr_occ_eb[gname] = 0.0
                curr_speed_eb[gname] = 0.0
                continue
            # 使用缓存或直接读取
            lm_local = self._step_cache.get("lane_metrics", {"eb": {}, "wb": {}})
            cached_eb = lm_local.get("eb", {}).get(gname, {})
            avg_occ = cached_eb.get("occ", None)
            avg_speed = cached_eb.get("speed", None)
            avg_halts = cached_eb.get("halts", None)
            if avg_occ is None:
                occs: List[float] = []
                for lid in lane_ids:
                    try:
                        occs.append(traci.lane.getLastStepOccupancy(lid))
                    except Exception:
                        pass
                avg_occ = float(np.mean(occs)) if occs else 0.0
                lm_local.setdefault("eb", {}).setdefault(gname, {})["occ"] = avg_occ
            if avg_speed is None:
                speeds: List[float] = []
                for lid in lane_ids:
                    try:
                        speeds.append(traci.lane.getLastStepMeanSpeed(lid))
                    except Exception:
                        pass
                avg_speed = float(np.mean(speeds)) if speeds else 0.0
                lm_local.setdefault("eb", {}).setdefault(gname, {})["speed"] = avg_speed
            if avg_halts is None:
                halts: List[float] = []
                for lid in lane_ids:
                    try:
                        halts.append(traci.lane.getLastStepHaltingNumber(lid))
                    except Exception:
                        pass
                avg_halts = float(np.mean(halts)) if halts else 0.0
                lm_local.setdefault("eb", {}).setdefault(gname, {})["halts"] = avg_halts
            avg_speed_norm = min(avg_speed / self.max_ref_speed_mps, 1.0)
            try:
                self._halts_hist_eb.get(gname, deque(maxlen=self._halts_window_steps)).append(avg_halts)
            except Exception:
                pass
            upper_halts = self._get_halts_upper_for_segment("eb", gname)
            halts_norm = max(0.0, min(avg_halts / float(max(upper_halts, 1e-6)), 1.0))
            
            # 输出当前占有率、速度与停止车归一
            feats.append(avg_occ)
            feats.append(avg_speed_norm)
            feats.append(halts_norm)
            curr_occ_eb[gname] = avg_occ
            curr_speed_eb[gname] = avg_speed_norm

        # 西向
        for gname in self.group_keys:
            lane_ids = self.wb_lane_groups.get(gname, [])
            if not lane_ids:
                feats.extend([0.0, 0.0, 0.0])
                curr_occ_wb[gname] = 0.0
                curr_speed_wb[gname] = 0.0
                continue
            lm_local = self._step_cache.get("lane_metrics", {"eb": {}, "wb": {}})
            cached_wb = lm_local.get("wb", {}).get(gname, {})
            avg_occ = cached_wb.get("occ", None)
            avg_speed = cached_wb.get("speed", None)
            avg_halts = cached_wb.get("halts", None)
            if avg_occ is None:
                occs = []
                for lid in lane_ids:
                    try:
                        occs.append(traci.lane.getLastStepOccupancy(lid))
                    except Exception:
                        pass
                avg_occ = float(np.mean(occs)) if occs else 0.0
                lm_local.setdefault("wb", {}).setdefault(gname, {})["occ"] = avg_occ
            if avg_speed is None:
                speeds = []
                for lid in lane_ids:
                    try:
                        speeds.append(traci.lane.getLastStepMeanSpeed(lid))
                    except Exception:
                        pass
                avg_speed = float(np.mean(speeds)) if speeds else 0.0
                lm_local.setdefault("wb", {}).setdefault(gname, {})["speed"] = avg_speed
            if avg_halts is None:
                halts = []
                for lid in lane_ids:
                    try:
                        halts.append(traci.lane.getLastStepHaltingNumber(lid))
                    except Exception:
                        pass
                avg_halts = float(np.mean(halts)) if halts else 0.0
                lm_local.setdefault("wb", {}).setdefault(gname, {})["halts"] = avg_halts
            avg_speed_norm = min(avg_speed / self.max_ref_speed_mps, 1.0)
            try:
                self._halts_hist_wb.get(gname, deque(maxlen=self._halts_window_steps)).append(avg_halts)
            except Exception:
                pass
            upper_halts = self._get_halts_upper_for_segment("wb", gname)
            halts_norm = max(0.0, min(avg_halts / float(max(upper_halts, 1e-6)), 1.0))
            
            feats.append(avg_occ)
            feats.append(avg_speed_norm)
            feats.append(halts_norm)
            curr_occ_wb[gname] = avg_occ
            curr_speed_wb[gname] = avg_speed_norm

        # 2) 信号：phase + remaining + 绿灯窗口
        for tid in self.tls_ids:
            try:
                phase_index = traci.trafficlight.getPhase(tid)
                remain = traci.trafficlight.getNextSwitch(tid) - traci.simulation.getTime()
            except traci.TraCIException:
                phase_index = 0
                remain = 0.0
            meta = self._tls_phase_meta.get(tid, {"phase_count": 8.0, "max_phase_duration": 60.0})
            phase_den = float(max(int(meta["phase_count"]) - 1, 1))
            phase_norm = min(float(phase_index) / phase_den, 1.0)
            remain = max(remain, 0.0)
            remain_norm = min(float(remain) / float(max(meta["max_phase_duration"], 1.0)), 1.0)
            
            # 绿灯窗口：下一绿灯窗口的起始偏移与持续时间（归一到一个周期）
            _, next_green_start_norm, next_green_dur_norm = self._get_tls_green_window_features(tid)
            feats.extend([phase_norm, remain_norm, next_green_start_norm, next_green_dur_norm])

        # 3) 需求趋势（60s滑动，归一）
        feats.append(float(self.metrics.get("demand_trend_norm", 0.0)))
        # 4) 慢趋势（5分钟滑动，归一）
        feats.append(float(self.metrics.get("demand_trend_slow_norm", 0.0)))

        # 5) 全局 CAV 占比（episode 级 sanity check）
        try:
            global_cav_ratio = self._get_cav_ratio()
        except Exception:
            global_cav_ratio = 0.0
        feats.append(global_cav_ratio)

        # 6) 每段 CAV 占比（东/西向）
        for gname in self.group_keys:
            eb_ratio = self._get_cav_ratio_on_lanes(self.lane_groups.get(gname, []))
            feats.append(eb_ratio)
        for gname in self.group_keys:
            wb_ratio = self._get_cav_ratio_on_lanes(self.wb_lane_groups.get(gname, []))
            feats.append(wb_ratio)

        # 7) 每段速度波动（与上一步的差值，归一化）
        for gname in self.group_keys:
            prev = float(self._prev_segment_speeds_norm.get("eb", {}).get(gname, 0.0))
            curr = float(curr_speed_eb.get(gname, 0.0))
            delta = max(0.0, min(abs(curr - prev), 1.0))
            feats.append(delta)
            self._prev_segment_speeds_norm.setdefault("eb", {})[gname] = curr
        for gname in self.group_keys:
            prev = float(self._prev_segment_speeds_norm.get("wb", {}).get(gname, 0.0))
            curr = float(curr_speed_wb.get(gname, 0.0))
            delta = max(0.0, min(abs(curr - prev), 1.0))
            feats.append(delta)
            self._prev_segment_speeds_norm.setdefault("wb", {})[gname] = curr

        # 8) 方向化变幅因子（红/绿切换导致的临时收紧/放宽，归一到[0,1]；1≈常规、>1≈放宽、<1≈收紧）
        try:
            lf_eb_vals = list(self.last_local_factor.get("eb", {}).values())
            lf_wb_vals = list(self.last_local_factor.get("wb", {}).values())
            lf_eb = float(np.mean(lf_eb_vals)) if lf_eb_vals else 1.0
            lf_wb = float(np.mean(lf_wb_vals)) if lf_wb_vals else 1.0
        except Exception:
            lf_eb, lf_wb = 1.0, 1.0
        # 经验范围约在 [0.5, 2.0]，保守归一到 [0,1]
        lf_eb_norm = max(0.0, min(lf_eb / 2.0, 1.0))
        lf_wb_norm = max(0.0, min(lf_wb / 2.0, 1.0))
        feats.extend([lf_eb_norm, lf_wb_norm])

        return np.array(feats, dtype=np.float32)

    # --------------------------------------------------------
    def _compute_reward_per_step(self) -> float:
        """
        多目标奖励（覆盖旅行时间/延误、排队溢出、停车延迟、能耗代理、速度平滑、绿波协调、吞吐与需求稳健）。

        统一将各子项归一到 [0,1] 后按权重组合：
        total = +w_coord*coord + w_eff*speed_norm + w_thru*throughput
                -w_delay*delay - w_qov*queue_overflow - w_stops*stops
                -w_fuel*fuel - w_smooth*smooth - w_robust*robust
        """
        try:
            # 1) 读取缓存/回退采集
            lm = self._step_cache.get("lane_metrics")
            if lm is None:
                # 需要占有率以支持拥堵项
                self._collect_lane_group_metrics(need_occ=True)
                lm = self._step_cache.get("lane_metrics", {"eb": {}, "wb": {}})

            # 2) 基础量：速度（归一）、停止车（归一：动态上限）
            speeds_norm: List[float] = []
            halts_norms: List[float] = []

            def _seg_len_avg(direction: str, segment: str) -> float:
                lane_ids = (self.lane_groups if direction == "eb" else self.wb_lane_groups).get(segment, [])
                if not lane_ids:
                    return 0.0
                lengths: List[float] = []
                for lid in lane_ids:
                    try:
                        L = float(self._lane_length_cache.get(lid)) if lid in self._lane_length_cache else float(traci.lane.getLength(lid))
                        self._lane_length_cache[lid] = L
                        lengths.append(L)
                    except Exception:
                        pass
                return float(np.mean(lengths)) if lengths else 0.0

            delay_norms: List[float] = []
            smooth_deltas: List[float] = []

            for direction in ("eb", "wb"):
                for g in self.group_keys:
                    vals = lm.get(direction, {}).get(g, {})
                    v_mps = float(vals.get("speed", 0.0))
                    v_norm = max(0.0, min(v_mps / max(self.max_ref_speed_mps, 1e-6), 1.0))
                    speeds_norm.append(v_norm)

                    avg_halts = float(vals.get("halts", 0.0))
                    upper_halts = self._get_halts_upper_for_segment(direction, g)
                    hn = max(0.0, min(avg_halts / float(max(upper_halts, 1e-6)), 1.0))
                    halts_norms.append(hn)

                    # 旅行时间/延误（基于段均长度与平均速度近似）
                    L = _seg_len_avg(direction, g)
                    if L > 0.0:
                        t_free = L / float(max(self.max_ref_speed_mps, 1e-6))
                        t_act = L / float(max(v_mps, 1e-3))
                        extra_ratio = max(0.0, (t_act / float(max(t_free, 1e-6))) - 1.0)
                        d_norm = max(0.0, min(extra_ratio / float(max(self.delay_max_ratio_extra, 1e-6)), 1.0))
                        delay_norms.append(d_norm)

                    # 速度波动（与上一步测得的差异，归一）
                    prev = float(self._prev_meas_segment_speeds_norm.get(direction, {}).get(g, 0.0))
                    curr = v_norm
                    delta = max(0.0, min(abs(curr - prev), 1.0))
                    smooth_deltas.append(delta)
                    self._prev_meas_segment_speeds_norm.setdefault(direction, {})[g] = curr

            speed_norm = float(np.mean(speeds_norm)) if speeds_norm else 0.0
            halting_norm = float(np.mean(halts_norms)) if halts_norms else 0.0

            # 2.5) 拥堵惩罚（占有率平均，0..1）
            occ_vals: List[float] = []
            try:
                for direction in ("eb", "wb"):
                    for g in self.group_keys:
                        v = float(lm.get(direction, {}).get(g, {}).get("occ", 0.0))
                        occ_vals.append(v)
            except Exception:
                occ_vals = occ_vals
            congestion_pen = float(np.mean(occ_vals)) if occ_vals else 0.0

            # 3) 队列溢出惩罚归一化：将“超阈值部分”按 (1 - 阈值) 缩放到 [0,1]
            if halts_norms:
                _excess = [max(h - self.queue_overflow_threshold, 0.0) for h in halts_norms]
                _raw_overflow = float(np.mean(_excess))
                _denom = float(max(1.0 - self.queue_overflow_threshold, 1e-6))
                queue_overflow = max(0.0, min(_raw_overflow / _denom, 1.0))
            else:
                queue_overflow = 0.0

            # 4) 停车惩罚（基于 halts_norm）
            stops_pen = halting_norm

            # 5) 延误惩罚（平均）
            delay_pen = float(np.mean(delay_norms)) if delay_norms else 0.0

            # 6) 能耗代理或真实燃油
            fuel_norm = 0.0
            if self.use_real_fuel:
                try:
                    corr_lanes = set(sum([self.lane_groups.get(k, []) for k in self.group_keys], [])) | set(sum([self.wb_lane_groups.get(k, []) for k in self.group_keys], []))
                    total_fc = 0.0
                    vids = []
                    try:
                        vids = traci.vehicle.getIDList()
                    except Exception:
                        vids = []
                    for vid in vids:
                        try:
                            lid = traci.vehicle.getLaneID(vid)
                            if lid in corr_lanes:
                                total_fc += float(traci.vehicle.getFuelConsumption(vid))
                        except Exception:
                            pass
                    self._fuel_hist.append(total_fc)
                    # 归一：相对窗口 P95（兜底用均值）
                    ref = 0.0
                    try:
                        arr = np.array(list(self._fuel_hist), dtype=np.float32)
                        if arr.size > 0:
                            ref = float(np.percentile(arr, 95)) if arr.size >= 5 else float(np.mean(arr))
                    except Exception:
                        ref = 0.0
                    if ref > 0.0:
                        fuel_norm = max(0.0, min(total_fc / ref, 1.0))
                    else:
                        fuel_norm = 0.0
                except Exception:
                    fuel_norm = 0.0
            else:
                # 代理：以怠速（halts）与速度波动联合近似
                fuel_norm = max(0.0, min(0.6 * halting_norm + 0.4 * (float(np.mean(smooth_deltas)) if smooth_deltas else 0.0), 1.0))

            # 7) 速度平滑惩罚（平均速度差），可按尺度放大用于提高可分辨性
            _smooth_raw = float(np.mean(smooth_deltas)) if smooth_deltas else 0.0
            smooth_pen = max(0.0, min(_smooth_raw * float(self.smoothness_penalty_scale), 1.0))

            # 8) 绿波/协调奖励（信号就绪度）
            coord_vals: List[float] = []
            for tid in (self.tls_ids or []):
                try:
                    is_green_now, start_norm, dur_norm = self._get_tls_green_window_features(tid)
                except Exception:
                    is_green_now, start_norm, dur_norm = (0.0, 1.0, 0.0)
                # 简化就绪度：当前绿=1；否则以“持续-起始偏移”衡量下一绿窗可达性
                ready = 1.0 if is_green_now == 1.0 else max(0.0, dur_norm - start_norm)
                coord_vals.append(max(0.0, min(ready, 1.0)))
            coordination = float(np.mean(coord_vals)) if coord_vals else 0.0

            # 8.1) 信号遵从惩罚与绿窗通过激励（改用CAV子集速度，动态红灯安全阈值）
            try:
                safe_candidates = []
                if self.speed_levels_mps:
                    safe_candidates.append(float(min(self.speed_levels_mps)))
                safe_candidates.append(0.5 * float(self.max_ref_speed_mps))
                safe_mps_red = float(max(safe_candidates)) if safe_candidates else 12.0
            except Exception:
                safe_mps_red = 12.0
            signal_pen_vals: List[float] = []
            green_pass_items: List[tuple] = []  # 每项为 (score, weight)

            def _avg_cav_speed_on_lanes(lane_ids: List[str]) -> Optional[float]:
                if not lane_ids:
                    return None
                speeds: List[float] = []
                seen = set()
                for lid in lane_ids:
                    try:
                        vids = traci.lane.getLastStepVehicleIDs(lid)
                    except Exception:
                        vids = []
                    for vid in vids:
                        if vid in seen:
                            continue
                        seen.add(vid)
                        try:
                            if str(traci.vehicle.getTypeID(vid)) == str(self.cav_type_id):
                                speeds.append(float(traci.vehicle.getSpeed(vid)))
                        except Exception:
                            pass
                if speeds:
                    return float(np.mean(speeds))
                return None

            for direction in ("eb", "wb"):
                for g in self.group_keys:
                    tls_id = self.segment_tls_map.get(direction, {}).get(g)
                    if not tls_id:
                        continue
                    vals = lm.get(direction, {}).get(g, {})
                    v_mps = float(vals.get("speed", 0.0))
                    # 车道集合，用于CAV速度与长度估计
                    lane_ids = (self.lane_groups if direction == "eb" else self.wb_lane_groups).get(g, [])
                    v_cav = _avg_cav_speed_on_lanes(lane_ids)
                    v_used = float(v_cav) if v_cav is not None else float(v_mps)
                    try:
                        is_green_now, start_norm, dur_norm = self._get_tls_green_window_features(tls_id)
                    except Exception:
                        is_green_now, start_norm, dur_norm = (0.0, 1.0, 0.0)
                    meta = self._tls_cycle_meta.get(tls_id, {})
                    cycle_len = float(meta.get("cycle_len", 60.0))
                    start_sec = float(start_norm) * cycle_len
                    dur_sec = float(dur_norm) * cycle_len
                    # 红灯期间仍高速的惩罚（相对参考速度归一，黄灯容差豁免）
                    if is_green_now < 1.0:
                        in_yellow_tolerance = bool(start_sec <= float(self.yellow_phase_tolerance_seconds))
                        if (not in_yellow_tolerance) and v_used > safe_mps_red:
                            signal_pen_vals.append(max(0.0, min((v_used - safe_mps_red) / float(max(self.max_ref_speed_mps, 1e-6)), 1.0)))
                    # 绿窗通过奖励：预测到达时间位于（当前/下一）绿窗附近，使用三角核软匹配；按占有率加权
                    # 段平均长度与预计通过时间
                    L = 0.0
                    if lane_ids:
                        lens: List[float] = []
                        for lid in lane_ids:
                            try:
                                L_i = float(self._lane_length_cache.get(lid)) if lid in self._lane_length_cache else float(traci.lane.getLength(lid))
                                self._lane_length_cache[lid] = L_i
                                lens.append(L_i)
                            except Exception:
                                pass
                        L = float(np.mean(lens)) if lens else 0.0
                    if L > 0.0 and v_used > 0.1 and dur_sec > 0.0:
                        eta = L / float(max(v_used, 1e-3))
                        # 使用三角核：以绿窗中心为峰，半宽取 max(dur/2, decision_interval)
                        half = max(dur_sec * 0.5, float(self.decision_interval))
                        center = start_sec + 0.5 * dur_sec
                        score = 1.0 - (abs(eta - center) / float(max(half, 1e-6)))
                        score = max(0.0, min(score, 1.0))
                        occ = float(vals.get("occ", 0.0))
                        w = float(max(occ, 0.0)) ** float(self.green_pass_occ_weight_pow)
                        green_pass_items.append((score, w))

            signal_compliance_pen = float(np.mean(signal_pen_vals)) if signal_pen_vals else 0.0
            if green_pass_items:
                total_w = float(sum(w for _, w in green_pass_items))
                if total_w > 1e-9:
                    green_pass_reward = float(sum(s * w for s, w in green_pass_items)) / total_w
                else:
                    green_pass_reward = float(np.mean([s for s, _ in green_pass_items]))
            else:
                green_pass_reward = 0.0

            # 9) 吞吐奖励（使用已计算的趋势归一）
            throughput = float(self.metrics.get("demand_trend_norm", 0.0))

            # 10) 需求趋势稳健惩罚（在需求上升且仍高速且队列小的情形施加惩罚）
            trend_fast = float(self.metrics.get("demand_trend_norm", 0.0))
            robust_pen = 0.0
            if trend_fast > 0.6 and speed_norm > 0.6 and halting_norm < 0.2:
                robust_pen = max(0.0, min((trend_fast - 0.6) / 0.4, 1.0))

            # 11) 组合奖励
            total = 0.0
            total += self.rw.get("coordination", 0.0) * coordination
            total += self.rw.get("efficiency", 0.0) * speed_norm
            total += self.rw.get("throughput", 0.0) * throughput
            total -= self.rw.get("delay", 0.0) * delay_pen
            total -= self.rw.get("queue_overflow", 0.0) * queue_overflow
            # 新增：拥堵惩罚（占有率）
            total -= self.rw.get("congestion", 0.0) * congestion_pen
            total -= self.rw.get("stops", 0.0) * stops_pen
            total -= self.rw.get("fuel", 0.0) * fuel_norm
            total -= self.rw.get("smoothness", 0.0) * smooth_pen
            total -= self.rw.get("demand_robust", 0.0) * robust_pen
            # 信号遵从（负）与绿窗通过（正）
            total -= self.rw.get("signal_compliance", 0.0) * signal_compliance_pen
            total += self.rw.get("green_pass", 0.0) * green_pass_reward

            # 可选裁剪
            if (self._reward_clip_min is not None) or (self._reward_clip_max is not None):
                clip_min = self._reward_clip_min if self._reward_clip_min is not None else -float("inf")
                clip_max = self._reward_clip_max if self._reward_clip_max is not None else float("inf")
                total = float(np.clip(total, clip_min, clip_max))

            # 记录奖励分量（用于分析/可视化）
            try:
                self.metrics["r_efficiency"] = self.rw.get("efficiency", 0.0) * speed_norm
                self.metrics["r_throughput"] = self.rw.get("throughput", 0.0) * throughput
                self.metrics["r_coordination"] = self.rw.get("coordination", 0.0) * coordination
                self.metrics["r_congestion"] = -self.rw.get("congestion", 0.0) * congestion_pen
                self.metrics["r_delay"] = -self.rw.get("delay", 0.0) * delay_pen
                self.metrics["r_queue_overflow"] = -self.rw.get("queue_overflow", 0.0) * queue_overflow
                self.metrics["r_stops"] = -self.rw.get("stops", 0.0) * stops_pen
                self.metrics["r_fuel"] = -self.rw.get("fuel", 0.0) * fuel_norm
                self.metrics["r_smoothness"] = -self.rw.get("smoothness", 0.0) * smooth_pen
                self.metrics["r_demand_robust"] = -self.rw.get("demand_robust", 0.0) * robust_pen
                self.metrics["r_signal_compliance"] = -self.rw.get("signal_compliance", 0.0) * signal_compliance_pen
                self.metrics["r_green_pass"] = self.rw.get("green_pass", 0.0) * green_pass_reward
                self.metrics["r_total"] = total
                # 便于直观监控的原始量
                self.metrics["r_speed_mean_mps"] = float(np.mean([float(lm.get("eb", {}).get(g, {}).get("speed", 0.0)) for g in self.group_keys] + [float(lm.get("wb", {}).get(g, {}).get("speed", 0.0)) for g in self.group_keys])) if lm else 0.0
                self.metrics["r_halting_avg"] = halting_norm

                # 累计归一指标（供 episode 结束平均）
                self.metrics["delay_sum_norm"] += float(delay_pen)
                self.metrics["delay_steps"] += 1
                self.metrics["stops_sum_norm"] += float(stops_pen)
                self.metrics["stops_steps"] += 1
                self.metrics["smooth_sum_norm"] += float(smooth_pen)
                self.metrics["smooth_steps"] += 1
                # 奖励分量累计与步数计数（episode均值）
                try:
                    self.metrics["reward_component_steps"] = int(self.metrics.get("reward_component_steps", 0) or 0) + 1
                    for k in [
                        "r_efficiency","r_throughput","r_coordination","r_congestion","r_delay","r_queue_overflow",
                        "r_stops","r_fuel","r_smoothness","r_demand_robust","r_signal_compliance","r_green_pass"
                    ]:
                        self.metrics[f"{k}_sum"] = float(self.metrics.get(f"{k}_sum", 0.0) or 0.0) + float(self.metrics.get(k, 0.0) or 0.0)
                except Exception:
                    pass
            except Exception:
                pass

            return total
        except Exception as e:
            print(f"[奖励计算] 多目标奖励计算失败: {e}")
            return 0.0
    # --------------------------------------------------------

    # --------------------------------------------------------
    

    # --------------------------------------------------------
    def _compute_action_change_penalty(self, new_speeds_mps: Tuple[float, float, float, float, float, float]) -> float:
        # 每个决策周期算一次
        if self._last_action_index < 0:
            return 0.0
        prev = np.array(self._last_action_speeds)
        curr = np.array(new_speeds_mps)
        diff_kph = np.abs(curr - prev) * 3.6
        # 归一化到 [0,1]：按每段最大允许变化 self.max_delta_kph 聚合（6 段）
        denom = float(max(len(self.group_keys) * 2 * max(self.max_delta_kph, 1e-6), 1.0))
        norm = float(np.sum(diff_kph)) / denom
        norm = max(0.0, min(norm, 1.0))
        return -self.rw["action_change"] * norm

    # --------------------------------------------------------
    def _get_cav_ratio(self) -> float:
        try:
            veh_ids = traci.vehicle.getIDList()
        except traci.TraCIException:
            return 0.0
        if not veh_ids:
            return 0.0
        cav_num = sum(1 for vid in veh_ids if traci.vehicle.getTypeID(vid) == self.cav_type_id)
        return cav_num / len(veh_ids)

    # 简化的错误处理：不再实现检查点保存和加载功能
    
    def close(self):
        # 先恢复所有被控车辆的默认类型最大速度，避免"限速残留"影响后续回合
        try:
            for vid in list(self._controlled_veh_ids):
                try:
                    default_v = float(self._veh_type_max_speed.pop(vid, self.max_ref_speed_mps))
                    traci.vehicle.setMaxSpeed(vid, default_v)
                except Exception:
                    pass
                finally:
                    self._controlled_veh_ids.discard(vid)
        except Exception:
            pass
        if traci.isLoaded():
            try:
                traci.close()
            except Exception:
                pass
