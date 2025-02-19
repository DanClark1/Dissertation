import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from utils import soft_update, hard_update
from equal_expert_mt_sac.mt_model import GaussianPolicy, QNetwork
from mt_sac.mt_sac import MT_SAC

class EE_MT_SAC(MT_SAC):
    def __init__(self, num_inputs, action_space, args, num_tasks=10, num_experts=3):

        super(EE_MT_SAC, self).__init__(num_inputs, action_space, args, num_tasks, num_experts)

        self.critic = QNetwork(num_inputs, action_space.shape[0], args.hidden_size, num_experts=num_experts).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)

        self.critic_target = QNetwork(num_inputs, action_space.shape[0], args.hidden_size, num_experts=num_experts).to(self.device)
        hard_update(self.critic_target, self.critic)

        self.policy = GaussianPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space, num_experts=num_experts).to(self.device)
        self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)
    
    def log_embeddings(self, writer, t, names):
        # no embeddings to log
        pass


