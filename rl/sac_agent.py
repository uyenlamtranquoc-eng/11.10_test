from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadPolicy(nn.Module):
    """
    策略网络：共享两层 MLP 干路 + 每头一个线性输出层。
    支持：
    - 可选 LayerNorm 规范化干路输出，稳定训练；
    - 可选“联合动作约束”门控：在 6 段（3+3）场景下，为 EB/WB 两个方向施加统一门控系数，约束各段 logits 的全局尺度。
    """

    def __init__(
        self,
        obs_dim: int,
        nvec: List[int],
        hidden_size: int = 256,
        use_layer_norm: bool = False,
        use_joint_constraint: bool = False,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.nvec = list(map(int, nvec))
        self.use_layer_norm = bool(use_layer_norm)
        self.use_joint_constraint = bool(use_joint_constraint)

        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        if self.use_layer_norm:
            self.layer_norm = nn.LayerNorm(hidden_size)
        else:
            self.layer_norm = None

        self.heads = nn.ModuleList([nn.Linear(hidden_size, n) for n in self.nvec])

        # 当 len(nvec)==6 时，默认将 [0,1,2] 作为 EB，[3,4,5] 作为 WB，做简单门控约束
        if self.use_joint_constraint and len(self.nvec) == 6:
            self._eb_idx = [0, 1, 2]
            self._wb_idx = [3, 4, 5]
            self.dir_gate_eb = nn.Linear(hidden_size, 1)
            self.dir_gate_wb = nn.Linear(hidden_size, 1)
        else:
            self._eb_idx = []
            self._wb_idx = []
            self.dir_gate_eb = None
            self.dir_gate_wb = None
        # 运行时门控均值（用于外部监控），在 forward 中更新
        self.last_gate_mean_eb: Optional[float] = None
        self.last_gate_mean_wb: Optional[float] = None

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        h = self.trunk(x)
        if self.layer_norm is not None:
            h = self.layer_norm(h)
        logits = [head(h) for head in self.heads]

        if self.use_joint_constraint and self.dir_gate_eb is not None and self.dir_gate_wb is not None:
            # 为两个方向生成门控系数（Sigmoid 到 0..1），统一缩放该方向的 logits 幅度
            g_eb = torch.sigmoid(self.dir_gate_eb(h))  # [B,1]
            g_wb = torch.sigmoid(self.dir_gate_wb(h))  # [B,1]
            # 记录门控均值（跨 batch 平均），便于训练期间观测门控约束的实际生效程度
            try:
                self.last_gate_mean_eb = float(g_eb.mean().item())
                self.last_gate_mean_wb = float(g_wb.mean().item())
            except Exception:
                self.last_gate_mean_eb = None
                self.last_gate_mean_wb = None
            out: List[torch.Tensor] = []
            for i in range(len(logits)):
                li = logits[i]
                if i in self._eb_idx:
                    li = li * g_eb
                elif i in self._wb_idx:
                    li = li * g_wb
                out.append(li)
            return out
        return logits


class MultiHeadQ(nn.Module):
    """
    Q 网络：共享两层 MLP 干路 + 每头一个线性输出层。
    支持“联合动作上下文”：在干路输入端拼接所有头的离散动作嵌入，使每头 Q 感知其他段动作，从而改善 credit assignment。
    输出为每个头的 Q 值列表（shape: [B, n_i]）。
    """

    def __init__(
        self,
        obs_dim: int,
        nvec: List[int],
        hidden_size: int = 256,
        use_action_context: bool = False,
        ctx_embed_dim: int = 16,
    ):
        super().__init__()
        self.nvec = list(map(int, nvec))
        self.use_action_context = bool(use_action_context)
        self.ctx_embed_dim = int(ctx_embed_dim)

        if self.use_action_context:
            self.ctx_embeds = nn.ModuleList([nn.Embedding(n, self.ctx_embed_dim) for n in self.nvec])
            ctx_total = self.ctx_embed_dim * len(self.nvec)
            in_dim = int(obs_dim + ctx_total)
        else:
            self.ctx_embeds = None
            in_dim = int(obs_dim)

        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.heads = nn.ModuleList([nn.Linear(hidden_size, n) for n in self.nvec])

    def forward(self, x: torch.Tensor, action_ctx: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
        if self.use_action_context and (self.ctx_embeds is not None) and (action_ctx is not None):
            # action_ctx: [B, H]，每头一个离散索引
            pieces: List[torch.Tensor] = []
            for i, emb in enumerate(self.ctx_embeds):
                ai = action_ctx[:, i].long().clamp(min=0, max=self.nvec[i] - 1)
                pieces.append(emb(ai))  # [B, D]
            ctx = torch.cat(pieces, dim=1)  # [B, H*D]
            x_in = torch.cat([x, ctx], dim=1)
        else:
            x_in = x
        h = self.trunk(x_in)
        return [head(h) for head in self.heads]


class MultiHeadSACAgent:
    """
    最简版离散多头 SAC（MultiDiscrete）
    - 无自适应温度（alpha 固定常数）
    - 无 multiscale/multi-objective/temporal/coordination 等增强模块
    - 仅保留双 Q + 策略更新 + 目标网络软更新
    """

    def __init__(
        self,
        obs_dim: int,
        nvec: List[int],
        lr: float = 3e-4,
        actor_lr: Optional[float] = None,
        critic_lr: Optional[float] = None,
        gamma: float = 0.99,
        target_update_interval: int = 1,
        device: str = "cpu",
        tau: float = 0.005,
        alpha: float = 0.2,
        hidden_size: int = 256,
        max_grad_norm: Optional[float] = None,
        auto_alpha: bool = False,
        alpha_lr: Optional[float] = None,
        target_entropy_scale: float = 1.0,
        use_joint_action_constraint: bool = False,
        actor_use_layer_norm: bool = False,
        actor_context_samples: int = 1,
        target_context_samples: int = 1,
    ) -> None:
        self.device = torch.device(device)
        self.obs_dim = int(obs_dim)
        self.nvec = list(map(int, nvec))
        self.num_heads = len(self.nvec)

        self.gamma = float(gamma)
        self.tau = float(tau)
        self.target_update_interval = int(target_update_interval)
        self.alpha = float(alpha)
        self.auto_alpha = bool(auto_alpha)
        self.target_entropy_scale = float(target_entropy_scale)
        self._train_steps = 0
        self.max_grad_norm = max_grad_norm
        # 多样本上下文采样数（降低 Monte Carlo 方差）
        self.actor_ctx_K = max(1, int(actor_context_samples))
        self.target_ctx_K = max(1, int(target_context_samples))

        # 网络
        self.policy = MultiHeadPolicy(
            self.obs_dim,
            self.nvec,
            hidden_size,
            use_layer_norm=bool(actor_use_layer_norm),
            use_joint_constraint=bool(use_joint_action_constraint),
        ).to(self.device)
        # 让 Q 感知联合动作上下文
        self.q1 = MultiHeadQ(self.obs_dim, self.nvec, hidden_size, use_action_context=True).to(self.device)
        self.q2 = MultiHeadQ(self.obs_dim, self.nvec, hidden_size, use_action_context=True).to(self.device)
        self.q1_target = MultiHeadQ(self.obs_dim, self.nvec, hidden_size, use_action_context=True).to(self.device)
        self.q2_target = MultiHeadQ(self.obs_dim, self.nvec, hidden_size, use_action_context=True).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        self.q1_target.eval()
        self.q2_target.eval()

        # 优化器
        actor_lr = float(actor_lr) if actor_lr is not None else float(lr)
        critic_lr = float(critic_lr) if critic_lr is not None else float(lr)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=actor_lr)
        self.q1_optimizer = torch.optim.Adam(self.q1.parameters(), lr=critic_lr)
        self.q2_optimizer = torch.optim.Adam(self.q2.parameters(), lr=critic_lr)

        # 自适应温度 alpha
        if self.auto_alpha:
            # 每头独立 alpha（log 空间参数化）
            init_log = float(np.log(max(1e-6, self.alpha)))
            self.log_alpha_vec = nn.Parameter(torch.full((self.num_heads,), init_log, device=self.device))
            _alpha_lr = float(alpha_lr) if alpha_lr is not None else actor_lr
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha_vec], lr=_alpha_lr)
            # 目标熵：离散最大熵为 +log|A_i|，乘以缩放系数
            self.target_entropy_heads = [float(np.log(max(1, n)) * self.target_entropy_scale) for n in self.nvec]
            self.target_entropy_total = float(sum(self.target_entropy_heads))
        else:
            self.log_alpha_vec = None
            self.alpha_optimizer = None
            self.target_entropy_heads = None
            self.target_entropy_total = None

        # 运行指标
        self.metrics: Dict[str, Any] = {}

    # ---------------------- 基本工具 ----------------------
    @property
    def temperature(self) -> float:
        """返回温度的整体代表值（均值）。"""
        if self.auto_alpha and self.log_alpha_vec is not None:
            return float(self.log_alpha_vec.detach().exp().mean().item())
        return float(self.alpha)

    def temperature_heads(self) -> List[float]:
        if self.auto_alpha and self.log_alpha_vec is not None:
            return [float(v) for v in self.log_alpha_vec.detach().exp().cpu().numpy().tolist()]
        return [float(self.alpha)] * self.num_heads

    def set_train(self) -> None:
        self.policy.train()
        self.q1.train()
        self.q2.train()
        self.q1_target.eval()
        self.q2_target.eval()

    def set_eval(self) -> None:
        self.policy.eval()
        self.q1.eval()
        self.q2.eval()
        self.q1_target.eval()
        self.q2_target.eval()

    # ---------------------- 动作选择 ----------------------
    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        多头离散动作选择：
        - deterministic=True 时，每头取 argmax(logits)
        - 否则按 Categorical 分布采样
        """
        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            logits_list = self.policy(x)
            actions = []
            for logits in logits_list:
                if deterministic:
                    a = torch.argmax(logits, dim=-1)
                else:
                    dist = torch.distributions.Categorical(logits=logits)
                    a = dist.sample()
                actions.append(int(a.item()))
        return np.array(actions, dtype=np.int64)

    # ---------------------- 训练更新 ----------------------

    def update(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> Dict[str, Any]:
        """
        执行一次最小 SAC 更新。
        输入 batch: (states, actions, rewards, next_states, dones)
        - states: [B, obs_dim]
        - actions: [B, num_heads] (int64)
        - rewards: [B]
        - next_states: [B, obs_dim]
        - dones: [B] (bool or 0/1)
        """
        states, actions, rewards, next_states, dones = batch
        states = states.to(self.device)
        next_states = next_states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device).view(-1)
        dones = dones.to(self.device).view(-1)

        B = states.shape[0]

        # ---------- 1) 计算每头的目标值（对 a_-i 进行 K 次上下文采样并平均，降低方差） ----------
        with torch.no_grad():
            next_logits = self.policy(next_states)
            next_logp_list = [F.log_softmax(l, dim=-1) for l in next_logits]
            next_pi_list = [torch.exp(lp) for lp in next_logp_list]
            alpha_heads = (
                self.log_alpha_vec.detach().exp() if (self.auto_alpha and self.log_alpha_vec is not None) else torch.full((self.num_heads,), self.alpha, device=self.device)
            )
            v_next_heads_accum: List[torch.Tensor] = [torch.zeros(B, device=self.device) for _ in range(self.num_heads)]
            for _k in range(self.target_ctx_K):
                # 其他头的联合上下文基底：一次采样，用于固定 a_-i
                next_ctx_base = torch.stack([torch.distributions.Categorical(logits=l).sample() for l in next_logits], dim=1)  # [B,H]
                for i in range(self.num_heads):
                    v_i = torch.zeros(B, device=self.device)
                    for ai in range(self.nvec[i]):
                        ctx_i = next_ctx_base.clone()
                        ctx_i[:, i] = int(ai)
                        q1_i_vec = self.q1_target(next_states, action_ctx=ctx_i)[i]
                        q2_i_vec = self.q2_target(next_states, action_ctx=ctx_i)[i]
                        min_q_ai = torch.minimum(q1_i_vec[:, ai], q2_i_vec[:, ai])  # [B]
                        v_i = v_i + next_pi_list[i][:, ai] * (min_q_ai - alpha_heads[i] * next_logp_list[i][:, ai])
                    v_next_heads_accum[i] = v_next_heads_accum[i] + v_i
            v_next_heads = [v / float(self.target_ctx_K) for v in v_next_heads_accum]
            target_q_per_head = torch.stack([
                rewards + (1.0 - dones) * self.gamma * v_next_heads[i]
                for i in range(self.num_heads)
            ], dim=1)

        # ---------- 2) 更新 Q1/Q2 ----------
        # Q 前向时传入当前联合动作作为上下文
        q1_values_list = self.q1(states, action_ctx=actions)
        q2_values_list = self.q2(states, action_ctx=actions)
        # 注：actor 步骤需在不同上下文下评估候选动作的 Q；当前 q1/q2 前向不可直接复用

        # gather 当前动作的 Q 值，得到 [B, H]
        q1_a = []
        q2_a = []
        for i in range(self.num_heads):
            ai = actions[:, i].long()
            q1_sel = q1_values_list[i].gather(1, ai.view(-1, 1)).view(-1)
            q2_sel = q2_values_list[i].gather(1, ai.view(-1, 1)).view(-1)
            q1_a.append(q1_sel)
            q2_a.append(q2_sel)
        q1_a = torch.stack(q1_a, dim=1)  # [B, H]
        q2_a = torch.stack(q2_a, dim=1)  # [B, H]

        # 每样本 TD 误差（头维平均），用于内部监控（不返回给外部）
        td_errors_per_head = torch.abs(target_q_per_head - torch.minimum(q1_a, q2_a))  # [B, H]
        td_error_mean = td_errors_per_head.mean().item()
        # 记录诊断指标（供训练脚本写入CSV）
        try:
            self.metrics['td_error_mean'] = float(td_error_mean)
        except Exception:
            pass

        # MSE 损失（也可替换为 Huber）；计算每样本、每头的误差并聚合
        q1_td = q1_a - target_q_per_head  # [B, H]
        q2_td = q2_a - target_q_per_head  # [B, H]
        q1_loss = torch.mean(q1_td ** 2)
        q2_loss = torch.mean(q2_td ** 2)

        self.q1_optimizer.zero_grad(set_to_none=True)
        q1_loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.q1.parameters(), self.max_grad_norm)
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad(set_to_none=True)
        q2_loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.q2.parameters(), self.max_grad_norm)
        self.q2_optimizer.step()

        # ---------- 3) 更新策略（最小化 J_pi = E[alpha*logpi - minQ]） ----------
        logits_list = self.policy(states)
        logp_list = [F.log_softmax(l, dim=-1) for l in logits_list]
        pi_list = [torch.exp(lp) for lp in logp_list]

        alpha_heads_actor = (
            self.log_alpha_vec.detach().exp() if (self.auto_alpha and self.log_alpha_vec is not None) else torch.full((self.num_heads,), self.alpha, device=self.device)
        )
        actor_loss_terms: List[torch.Tensor] = []
        for i in range(self.num_heads):
            # 在 K 次不同联合上下文下，估计第 i 头候选动作的 minQ_i(a_i, a_-i) 的期望
            min_q_accum = torch.zeros(B, self.nvec[i], device=self.device)
            for _k in range(self.actor_ctx_K):
                # 为 actor 期望构造一次联合上下文采样（作为 a_-i 基底），并为每个候选 a_i 重建上下文
                curr_ctx_base = torch.stack([torch.distributions.Categorical(logits=l).sample() for l in logits_list], dim=1)
                with torch.no_grad():
                    min_q_per_action: List[torch.Tensor] = []
                    for ai in range(self.nvec[i]):
                        ctx_i = curr_ctx_base.clone()
                        ctx_i[:, i] = int(ai)
                        q1_i_vec = self.q1(states, action_ctx=ctx_i)[i]
                        q2_i_vec = self.q2(states, action_ctx=ctx_i)[i]
                        min_q_ai = torch.minimum(q1_i_vec[:, ai], q2_i_vec[:, ai])  # [B]
                        min_q_per_action.append(min_q_ai)
                    min_q_tensor_k = torch.stack(min_q_per_action, dim=1)  # [B, n_i]
                min_q_accum = min_q_accum + min_q_tensor_k
            min_q_tensor = min_q_accum / float(self.actor_ctx_K)
            # E_{a_i~pi_i}[ alpha*logpi_i - minQ_i ]
            term_i = (pi_list[i] * (alpha_heads_actor[i] * logp_list[i] - min_q_tensor)).sum(dim=-1).mean()
            actor_loss_terms.append(term_i)
        # 将各头损失取平均，避免随头数线性放大（使用stack.mean确保返回Tensor）
        actor_loss = torch.stack(actor_loss_terms).mean()
        # 记录actor损失（标量），便于外部CSV日志采集
        try:
            self.metrics['actor_loss'] = float(actor_loss.detach().item())
        except Exception:
            pass

        self.policy_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.policy_optimizer.step()

        # ---------- 4) 软更新目标网络 ----------
        self._train_steps += 1
        if self._train_steps % self.target_update_interval == 0:
            self._soft_update(self.q1, self.q1_target, self.tau)
            self._soft_update(self.q2, self.q2_target, self.tau)

        # ---------- 5) 自适应 alpha 更新 ----------
        if self.auto_alpha and self.alpha_optimizer is not None and self.log_alpha_vec is not None and self.target_entropy_heads is not None:
            # 每头独立的离散熵（不对策略反传梯度）
            entropy_heads = [-(pi_list[i] * logp_list[i]).sum(dim=-1) for i in range(self.num_heads)]  # each [B]
            alpha_vec = self.log_alpha_vec.exp()
            # J(alpha_i) = alpha_i * (H_i - target_i)
            alpha_losses = [alpha_vec[i] * (entropy_heads[i].detach() - float(self.target_entropy_heads[i])) for i in range(self.num_heads)]
            alpha_loss = torch.stack([v.mean() for v in alpha_losses]).mean()
            self.alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_optimizer.step()

        # 记录指标：损失、温度、TD 误差统计、每头平均熵
        head_entropies = [
            float((-(pi_list[i] * logp_list[i]).sum(dim=-1)).mean().item())
            for i in range(self.num_heads)
        ]
        # 门控均值（若启用联合动作约束，则在最近一次前向中更新）
        g_eb_mean = 0.0
        g_wb_mean = 0.0
        try:
            v_eb = getattr(self.policy, 'last_gate_mean_eb', None)
            v_wb = getattr(self.policy, 'last_gate_mean_wb', None)
            g_eb_mean = float(v_eb) if v_eb is not None else 0.0
            g_wb_mean = float(v_wb) if v_wb is not None else 0.0
        except Exception:
            pass

        self.metrics = {
            'q1_loss': float(q1_loss.item()),
            'q2_loss': float(q2_loss.item()),
            'actor_loss': float(actor_loss.item()),
            'alpha': float(self.temperature),
            'alpha_heads': self.temperature_heads(),
            'train_steps': int(self._train_steps),
            'td_error_mean': float(td_error_mean),
            'td_error_max': float(td_errors_per_head.max().item()),
            'td_error_std': float(td_errors_per_head.std().item()),
            'gate_mean_eb': float(g_eb_mean),
            'gate_mean_wb': float(g_wb_mean),
        }
        for i, h in enumerate(head_entropies):
            self.metrics[f'entropy_head_{i}'] = h

        
        out: Dict[str, Any] = dict(self.metrics)
        out['total_loss'] = self.metrics['q1_loss'] + self.metrics['q2_loss'] + self.metrics['actor_loss']
        return out

    @staticmethod
    def _soft_update(source: nn.Module, target: nn.Module, tau: float) -> None:
        with torch.no_grad():
            for p_t, p_s in zip(target.parameters(), source.parameters()):
                p_t.data.mul_(1.0 - tau).add_(tau * p_s.data)

    # ---------------------- Checkpoint ----------------------
    def save(self, path: str) -> None:
        ckpt = {
            'obs_dim': self.obs_dim,
            'nvec': self.nvec,
            'gamma': self.gamma,
            'tau': self.tau,
            'target_update_interval': self.target_update_interval,
            'alpha': self.alpha,
            'auto_alpha': bool(self.auto_alpha),
            'target_entropy_scale': float(self.target_entropy_scale),
            'policy_state': self.policy.state_dict(),
            'q1_state': self.q1.state_dict(),
            'q2_state': self.q2.state_dict(),
            'q1_target_state': self.q1_target.state_dict(),
            'q2_target_state': self.q2_target.state_dict(),
            'policy_opt': self.policy_optimizer.state_dict(),
            'q1_opt': self.q1_optimizer.state_dict(),
            'q2_opt': self.q2_optimizer.state_dict(),
            'train_steps': self._train_steps,
        }
        if self.auto_alpha and self.log_alpha_vec is not None and self.alpha_optimizer is not None:
            ckpt['log_alpha_vec'] = [float(v) for v in self.log_alpha_vec.detach().cpu().numpy().tolist()]
            ckpt['alpha_opt'] = self.alpha_optimizer.state_dict()
            if self.target_entropy_heads is not None:
                ckpt['target_entropy_heads'] = list(self.target_entropy_heads)
        torch.save(ckpt, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)

        def pick(d: Dict[str, Any], primary: str, alts: List[str]) -> Optional[Any]:
            if primary in d:
                return d[primary]
            for k in alts:
                if k in d:
                    return d[k]
            for nest in ['networks', 'models', 'state', 'agent', 'agents']:
                v = d.get(nest)
                if isinstance(v, dict):
                    if primary in v:
                        return v[primary]
                    for k in alts:
                        if k in v:
                            return v[k]
            return None

        pol_sd = pick(ckpt, 'policy_state', ['policy', 'actor_state', 'policy_state_dict', 'pi_state'])
        q1_sd = pick(ckpt, 'q1_state', ['q1', 'critic1_state', 'critic_1_state'])
        q2_sd = pick(ckpt, 'q2_state', ['q2', 'critic2_state', 'critic_2_state'])
        q1_t_sd = pick(ckpt, 'q1_target_state', ['q1_target', 'target_q1_state'])
        q2_t_sd = pick(ckpt, 'q2_target_state', ['q2_target', 'target_q2_state'])
        pol_opt_sd = pick(ckpt, 'policy_opt', ['policy_optimizer', 'actor_opt', 'policy_optim_state'])
        q1_opt_sd = pick(ckpt, 'q1_opt', ['critic1_opt', 'q1_optimizer'])
        q2_opt_sd = pick(ckpt, 'q2_opt', ['critic2_opt', 'q2_optimizer'])

        if pol_sd is not None:
            self.policy.load_state_dict(pol_sd)
        if q1_sd is not None:
            self.q1.load_state_dict(q1_sd)
        if q2_sd is not None:
            self.q2.load_state_dict(q2_sd)
        if q1_t_sd is not None:
            self.q1_target.load_state_dict(q1_t_sd)
        else:
            self.q1_target.load_state_dict(self.q1.state_dict())
        if q2_t_sd is not None:
            self.q2_target.load_state_dict(q2_t_sd)
        else:
            self.q2_target.load_state_dict(self.q2.state_dict())

        if pol_opt_sd is not None:
            self.policy_optimizer.load_state_dict(pol_opt_sd)
        if q1_opt_sd is not None:
            self.q1_optimizer.load_state_dict(q1_opt_sd)
        if q2_opt_sd is not None:
            self.q2_optimizer.load_state_dict(q2_opt_sd)

        self.alpha = float(ckpt.get('alpha', self.alpha))
        self.auto_alpha = bool(ckpt.get('auto_alpha', self.auto_alpha))
        self.target_entropy_scale = float(ckpt.get('target_entropy_scale', self.target_entropy_scale))
        if self.auto_alpha:
            # 若无存储向量，则用当前 alpha 初始化均值
            vec = ckpt.get('log_alpha_vec', None)
            if vec is None:
                init_log = float(np.log(max(1e-6, self.alpha)))
                self.log_alpha_vec = nn.Parameter(torch.full((self.num_heads,), init_log, device=self.device))
            else:
                arr = torch.tensor([float(v) for v in vec], device=self.device)
                if arr.numel() != self.num_heads:
                    # 维度不匹配时回退到均值
                    init_log = float(np.log(max(1e-6, self.alpha)))
                    arr = torch.full((self.num_heads,), init_log, device=self.device)
                self.log_alpha_vec = nn.Parameter(arr)
            # 重建 alpha 优化器（默认与 actor 同学习率）
            pol_lr = 0.0
            for g in self.policy_optimizer.param_groups:
                pol_lr = float(g.get('lr', 3e-4))
                break
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha_vec], lr=pol_lr)
            if 'alpha_opt' in ckpt and ckpt['alpha_opt'] is not None:
                try:
                    self.alpha_optimizer.load_state_dict(ckpt['alpha_opt'])
                except Exception:
                    pass
            # 目标熵恢复或重建（修正为正号）
            te_heads = ckpt.get('target_entropy_heads', None)
            if te_heads is None:
                self.target_entropy_heads = [float(np.log(max(1, n)) * self.target_entropy_scale) for n in self.nvec]
            else:
                self.target_entropy_heads = [float(v) for v in te_heads]
            self.target_entropy_total = float(sum(self.target_entropy_heads))
        self._train_steps = int(ckpt.get('train_steps', ckpt.get('steps', 0)))

    # 兼容 run_train 的完整检查点保存/恢复
    def get_state(self) -> Dict[str, Any]:
        state = {
            'obs_dim': self.obs_dim,
            'nvec': self.nvec,
            'gamma': self.gamma,
            'tau': self.tau,
            'target_update_interval': self.target_update_interval,
            'alpha': self.alpha,
            'auto_alpha': bool(self.auto_alpha),
            'target_entropy_scale': float(self.target_entropy_scale),
            'policy_state': self.policy.state_dict(),
            'q1_state': self.q1.state_dict(),
            'q2_state': self.q2.state_dict(),
            'q1_target_state': self.q1_target.state_dict(),
            'q2_target_state': self.q2_target.state_dict(),
            'policy_opt': self.policy_optimizer.state_dict(),
            'q1_opt': self.q1_optimizer.state_dict(),
            'q2_opt': self.q2_optimizer.state_dict(),
            'train_steps': self._train_steps,
        }
        if self.auto_alpha and self.log_alpha_vec is not None and self.alpha_optimizer is not None:
            state['log_alpha_vec'] = [float(v) for v in self.log_alpha_vec.detach().cpu().numpy().tolist()]
            state['alpha_opt'] = self.alpha_optimizer.state_dict()
            if self.target_entropy_heads is not None:
                state['target_entropy_heads'] = list(self.target_entropy_heads)
        return state

    def load_state(self, state: Dict[str, Any]) -> None:
        def pick(d: Dict[str, Any], primary: str, alts: List[str]) -> Optional[Any]:
            if primary in d:
                return d[primary]
            for k in alts:
                if k in d:
                    return d[k]
            for nest in ['networks', 'models', 'state', 'agent', 'agents']:
                v = d.get(nest)
                if isinstance(v, dict):
                    if primary in v:
                        return v[primary]
                    for k in alts:
                        if k in v:
                            return v[k]
            return None

        pol_sd = pick(state, 'policy_state', ['policy', 'actor_state', 'policy_state_dict', 'pi_state'])
        q1_sd = pick(state, 'q1_state', ['q1', 'critic1_state', 'critic_1_state'])
        q2_sd = pick(state, 'q2_state', ['q2', 'critic2_state', 'critic_2_state'])
        q1_t_sd = pick(state, 'q1_target_state', ['q1_target', 'target_q1_state'])
        q2_t_sd = pick(state, 'q2_target_state', ['q2_target', 'target_q2_state'])
        pol_opt_sd = pick(state, 'policy_opt', ['policy_optimizer', 'actor_opt', 'policy_optim_state'])
        q1_opt_sd = pick(state, 'q1_opt', ['critic1_opt', 'q1_optimizer'])
        q2_opt_sd = pick(state, 'q2_opt', ['critic2_opt', 'q2_optimizer'])

        if pol_sd is not None:
            self.policy.load_state_dict(pol_sd)
        if q1_sd is not None:
            self.q1.load_state_dict(q1_sd)
        if q2_sd is not None:
            self.q2.load_state_dict(q2_sd)
        if q1_t_sd is not None:
            self.q1_target.load_state_dict(q1_t_sd)
        else:
            self.q1_target.load_state_dict(self.q1.state_dict())
        if q2_t_sd is not None:
            self.q2_target.load_state_dict(q2_t_sd)
        else:
            self.q2_target.load_state_dict(self.q2.state_dict())

        if pol_opt_sd is not None:
            self.policy_optimizer.load_state_dict(pol_opt_sd)
        if q1_opt_sd is not None:
            self.q1_optimizer.load_state_dict(q1_opt_sd)
        if q2_opt_sd is not None:
            self.q2_optimizer.load_state_dict(q2_opt_sd)

        self.alpha = float(state.get('alpha', self.alpha))
        self.auto_alpha = bool(state.get('auto_alpha', self.auto_alpha))
        self.target_entropy_scale = float(state.get('target_entropy_scale', self.target_entropy_scale))
        if self.auto_alpha:
            vec = state.get('log_alpha_vec', None)
            if vec is None:
                init_log = float(np.log(max(1e-6, self.alpha)))
                self.log_alpha_vec = nn.Parameter(torch.full((self.num_heads,), init_log, device=self.device))
            else:
                arr = torch.tensor([float(v) for v in vec], device=self.device)
                if arr.numel() != self.num_heads:
                    init_log = float(np.log(max(1e-6, self.alpha)))
                    arr = torch.full((self.num_heads,), init_log, device=self.device)
                self.log_alpha_vec = nn.Parameter(arr)
            pol_lr = 0.0
            for g in self.policy_optimizer.param_groups:
                pol_lr = float(g.get('lr', 3e-4))
                break
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha_vec], lr=pol_lr)
            aopt = state.get('alpha_opt', None)
            if aopt is not None:
                try:
                    self.alpha_optimizer.load_state_dict(aopt)
                except Exception:
                    pass
            te_heads = state.get('target_entropy_heads', None)
            if te_heads is None:
                self.target_entropy_heads = [float(np.log(max(1, n)) * self.target_entropy_scale) for n in self.nvec]
            else:
                self.target_entropy_heads = [float(v) for v in te_heads]
            self.target_entropy_total = float(sum(self.target_entropy_heads))
        self._train_steps = int(state.get('train_steps', state.get('steps', 0)))