import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from utils import soft_update, hard_update
from mt_sac.mt_model import GaussianPolicy, QNetwork
from mt_sac.debug_model import DebugGaussianPolicy, DebugQNetwork
from sac.sac import SAC

class MT_SAC(SAC):
    def __init__(self, num_inputs, action_space, writer, args, num_tasks=10, num_experts=3):
        super(MT_SAC, self).__init__(num_inputs, action_space, writer, args)

        self.critic = DebugQNetwork(num_inputs, action_space.shape[0], args.hidden_size,  writer=writer, num_experts=num_experts).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)

        self.critic_target = DebugQNetwork(num_inputs, action_space.shape[0], args.hidden_size, num_experts=num_experts).to(self.device)
        hard_update(self.critic_target, self.critic)

        self.policy = DebugGaussianPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space, writer=writer, num_experts=num_experts).to(self.device)
        self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)
        
    def log_embeddings(self, t, names):
        actor_queries = self.policy.moe.task_queries.detach().cpu()
        critic_queries_1 = self.critic.moe_1.task_queries.detach().cpu()
        critic_queries_2 = self.critic.moe_2.task_queries.detach().cpu()

        self.writer.add_embedding(actor_queries, metadata=names, tag='actor_queries', global_step=t)
        self.writer.add_embedding(critic_queries_1, metadata=names, tag='critic_queries_1', global_step=t)
        self.writer.add_embedding(critic_queries_2, metadata=names, tag='critic_queries_2', global_step=t)

    def update_parameters(self, memory, batch_size, updates):
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target, _ = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
        qf1, qf2, reg_loss_critic = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss + reg_loss_critic.mean() # add on moe regularisation loss
        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        pi, log_pi, _, reg_loss_pi = self.policy.sample(state_batch)

        qf1_pi, qf2_pi, _ = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi + reg_loss_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs


        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item(), reg_loss_critic.mean().item(), reg_loss_pi.mean().item()
