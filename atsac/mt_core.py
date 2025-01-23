import numpy as np
import scipy.signal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


LOG_STD_MAX = 2
LOG_STD_MIN = -20

class SquashedGaussianMoEActor(nn.Module):

    def __init__(self, obs_dim, act_dim, backbone_hidden_sizes, expert_hidden_sizes, num_experts, num_tasks, activation, act_limit):
        super().__init__()
        self.backbone = mlp([obs_dim] + list(backbone_hidden_sizes), activation, activation)

        # expert networks
        self.experts = nn.ModuleList([
            mlp([backbone_hidden_sizes[-1]] + list(expert_hidden_sizes), activation=activation) for _ in range(num_experts)
        ])

        # for attention to work these dimensions need to match
        task_queries_dim = expert_hidden_sizes[-1]
        self.task_queries = nn.Parameter(torch.randn(num_tasks, task_queries_dim))

        # matricies for computing values and keys from the experts
        self.key_matricies = nn.Parameter(torch.randn(num_experts,task_queries_dim, expert_hidden_sizes[-1]))
        self.value_matricies = nn.Parameter(torch.randn(num_experts, task_queries_dim, expert_hidden_sizes[-1]))

        self.mu_layer = mlp([expert_hidden_sizes[-1]] + [backbone_hidden_sizes[0]] + [act_dim], activation=activation)
        self.log_std_layer = mlp([expert_hidden_sizes[-1]] + [backbone_hidden_sizes[0]] + [act_dim], activation=activation)
        self.act_limit = act_limit

    def forward(self, obs, task, deterministic=False, with_logprob=True):
    
        # if obs is a single observation, add a batch dimension
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
            task = torch.tensor([task])

        backbone_output = self.backbone(obs)

        # Define a function to forward pass through each expert
        def expert_forward(expert, input_tensor):
            return expert(input_tensor)

        # Parallelize the expert outputs
        futures = [torch.jit.fork(expert_forward, expert, backbone_output) for expert in self.experts]

        # Gather the results
        expert_outputs = torch.stack([torch.jit.wait(future) for future in futures], dim=1)

        # compute values and keys
        expert_values = torch.einsum('kli,lij->klj', expert_outputs, self.value_matricies)
        expert_keys = torch.einsum('kli,lij->klj', expert_outputs, self.key_matricies)

        # compute attention weights
        attention_scores = torch.einsum('kni,ki->kn', expert_keys, self.task_queries[task])

        attention_weights = torch.softmax(attention_scores, dim=-1)

        # summing together each expert scaled by attention weights
        tower_input = torch.einsum('kn,kni->ki', attention_weights, expert_values)


        mu = self.mu_layer(tower_input)
        log_std = self.log_std_layer(tower_input)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding 
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290) 
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        return pi_action, logp_pi


class MoEQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, backbone_hidden_sizes, expert_hidden_sizes, num_experts, num_tasks, activation):
        super().__init__()
        # for attention to work these dimensions need to match
        task_queries_dim = expert_hidden_sizes[-1]

        # backbone network - shared features
        self.backbone = mlp([obs_dim + act_dim] + list(backbone_hidden_sizes), activation)

        # expert networks
        self.experts = nn.ModuleList([
            mlp([backbone_hidden_sizes[-1]] + list(expert_hidden_sizes), activation=activation) for _ in range(num_experts)
        ])

        # task queries
        self.task_queries = nn.Parameter(torch.randn(num_tasks, task_queries_dim))

        # matricies for computing values and keys from the experts
        self.key_matricies = nn.Parameter(torch.randn(num_experts,task_queries_dim, expert_hidden_sizes[-1]))
        self.value_matricies = nn.Parameter(torch.randn(num_experts, task_queries_dim, expert_hidden_sizes[-1]))

        # tower network (just assuming its the same dimensions as the backbone network)
        self.tower = mlp([expert_hidden_sizes[-1]] + list(backbone_hidden_sizes) + [1], activation=activation)


    def forward(self, obs, task, act):

        backbone_output = self.backbone(torch.cat([obs, act], dim=-1))

        # Define a function to forward pass through each expert
        def expert_forward(expert, input_tensor):
            return expert(input_tensor)

        # Parallelize the expert outputs
        futures = [torch.jit.fork(expert_forward, expert, backbone_output) for expert in self.experts]

        # Gather the results
        expert_outputs = torch.stack([torch.jit.wait(future) for future in futures], dim=1)

        # compute values and keys
        expert_values = torch.einsum('kli,lij->klj', expert_outputs, self.value_matricies)
        expert_keys = torch.einsum('kli,lij->klj', expert_outputs, self.key_matricies)

        # compute attention weights
        attention_scores = torch.einsum('kni,ki->kn', expert_keys, self.task_queries[task])

        attention_weights = torch.softmax(attention_scores, dim=-1)

        # summing together each expert scaled by attention weights
        tower_input = torch.einsum('kn,kni->ki', attention_weights, expert_values)

        # pass through tower network
        q = self.tower(tower_input)
        
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.

class MoEActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, num_tasks, num_experts, backbone_hidden_sizes=(256,256), 
                 actor_hidden_sizes=(256, 256),
                 expert_hidden_sizes=(400,400),
                 activation=nn.ReLU):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        self.pi = SquashedGaussianMoEActor(obs_dim, act_dim, actor_hidden_sizes, expert_hidden_sizes, num_experts, num_tasks, activation, act_limit)
        self.q1 = MoEQFunction(obs_dim, act_dim, backbone_hidden_sizes, expert_hidden_sizes, num_experts, num_tasks, activation)
        self.q2 = MoEQFunction(obs_dim, act_dim, backbone_hidden_sizes, expert_hidden_sizes, num_experts, num_tasks, activation)

    def act(self, obs, task, deterministic=False):
        with torch.no_grad():
            a, _ = self.pi(obs, task, deterministic, False)
            return a.numpy()