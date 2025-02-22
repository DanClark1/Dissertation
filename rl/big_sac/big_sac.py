import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from utils import soft_update, hard_update
from big_sac.big_model import GaussianPolicy, QNetwork, DeterministicPolicy
from sac.sac import SAC

class BIG_SAC(SAC):
    '''
    SAC but there's more parameters in the model
    Used to compare similar numbers of parameters in the models
    '''
    def __init__(self, num_inputs, action_space, writer, args):
        super(BIG_SAC, self).__init__(num_inputs, action_space, writer, args)
        self.critic = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)

        self.critic_target = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)

        self.policy = GaussianPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
        self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)
