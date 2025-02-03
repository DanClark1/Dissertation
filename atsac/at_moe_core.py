import numpy as np
import scipy.signal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal




def format_obs(o, num_tasks):
        '''
        Extracts the one-hot encoding from the observation vector,
        identifies the task and returns the observation vector
        without the one-hot encoding

        :param o: observation vector
        :param num_tasks: number of tasks
        :return: observation vector without the one-hot encoding
        '''
        task = torch.argmax(o[...,-num_tasks:], axis=-1)

        return o[..., :-num_tasks], task

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


class MoELayer(nn.Module):
    def __init__(self, input_dim, expert_hidden_sizes, num_experts, num_tasks, activation, mu):
        super().__init__()
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        self.mu = mu

        # Create expert networks (each expert is an MLP)
        self.experts = nn.ModuleList([
            mlp([input_dim] + list(expert_hidden_sizes), activation=activation)
            for _ in range(num_experts)
        ])

        # The task queries need to have a dimension matching the experts’ output.
        task_queries_dim = expert_hidden_sizes[-1]
        self.task_queries = nn.Parameter(torch.randn(num_tasks, task_queries_dim))
        # Matrices to compute keys and values from the expert outputs.
        self.key_matricies = nn.Parameter(torch.randn(num_experts, task_queries_dim, expert_hidden_sizes[-1]))
        self.value_matricies = nn.Parameter(torch.randn(num_experts, task_queries_dim, expert_hidden_sizes[-1]))

    def forward(self, backbone_output, task):
        # Compute expert outputs in parallel.
        def expert_forward(expert, inp):
            return expert(inp)
        futures = [torch.jit.fork(expert_forward, expert, backbone_output)
                   for expert in self.experts]
        expert_outputs = torch.stack([torch.jit.wait(f) for f in futures], dim=1)

        # Compute keys and values using einsum.
        expert_keys = torch.einsum('kli,lij->klj', expert_outputs, self.key_matricies)
        expert_values = torch.einsum('kli,lij->klj', expert_outputs, self.value_matricies)

        # Use the task query (indexed by the task) to compute attention scores.
        # Make sure to adjust dimensions if your task variable isn’t batch–wise.
        attention_scores = torch.einsum('kni,ki->kn', expert_keys, self.task_queries[task])
        attention_weights = torch.softmax(attention_scores, dim=-1)

        # Aggregate expert outputs.
        tower_input = torch.einsum('kn,kni->ki', attention_weights, expert_values)

        # Optionally compute a regularization term.
        eps = torch.ones_like(attention_weights) / (1e6)
        reg_loss_term = - (1 / self.num_experts) * self.mu * (torch.sum(attention_weights + eps))
        return tower_input, reg_loss_term


class SquashedGaussianMoEActor(nn.Module):

    def __init__(self, obs_dim, act_dim, backbone_hidden_sizes, expert_hidden_sizes, num_experts, num_tasks, activation, act_limit, mu=0.01):
        super().__init__()
        self.mu = mu
        self.num_tasks = num_tasks

        self.num_experts = num_experts
        obs_dim = obs_dim - num_tasks # removing one hot vector

        self.backbone = mlp([obs_dim] + list(backbone_hidden_sizes), activation, activation)

        self.moe_layer = MoELayer(backbone_hidden_sizes[-1], expert_hidden_sizes,
                                  num_experts, num_tasks, activation, mu)

        self.mu_layer = mlp([expert_hidden_sizes[-1]] + [backbone_hidden_sizes[0]] + [act_dim], activation=activation)
        self.log_std_layer = mlp([expert_hidden_sizes[-1]] + [backbone_hidden_sizes[0]] + [act_dim], activation=activation)
        self.act_limit = act_limit

    def forward(self, obs, deterministic=False, with_logprob=True):

        obs, task = format_obs(obs, self.num_tasks)
    
        # if obs is a single observation, add a batch dimension
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
            task = torch.tensor([task])

        backbone_output = self.backbone(obs)

        tower_input, reg_loss_term = self.moe_layer(backbone_output, task)

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

        return pi_action, logp_pi, reg_loss_term


class MoEQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, backbone_hidden_sizes, expert_hidden_sizes, num_experts, num_tasks, activation, mu=0.01, writer=None):
        super().__init__()

        self.mu = mu
        self.num_tasks = num_tasks
        self.num_experts = num_experts

        obs_dim = obs_dim - num_tasks # removing one hot vector from obs space

        # backbone network - shared features
        self.backbone = mlp([obs_dim + act_dim] + list(backbone_hidden_sizes), activation)

        self.moe_layer = MoELayer(backbone_hidden_sizes[-1], expert_hidden_sizes,
                                  num_experts, num_tasks, activation, mu)

        # tower network (just assuming its the same dimensions as the backbone network)
        self.tower = mlp([expert_hidden_sizes[-1]] + list(backbone_hidden_sizes) + [1], activation=activation)


    def forward(self, obs, act):
         
        obs, task = format_obs(obs, self.num_tasks)

        backbone_output = self.backbone(torch.cat([obs, act], dim=-1))

        tower_input, reg_loss_term = self.moe_layer(backbone_output, task)

        # pass through tower network
        q = self.tower(tower_input)
        
        return torch.squeeze(q, -1), reg_loss_term # Critical to ensure q has right shape.

class MoEActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, num_tasks, num_experts, backbone_hidden_sizes=(256,256), 
                 actor_hidden_sizes=(256, 256),
                 expert_hidden_sizes=(400,400),
                 activation=nn.ReLU, writer=None):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # build policy and value functions
        self.pi = SquashedGaussianMoEActor(obs_dim, act_dim, actor_hidden_sizes, expert_hidden_sizes, num_experts, num_tasks, activation, act_limit)
        self.q1 = MoEQFunction(obs_dim, act_dim, backbone_hidden_sizes, expert_hidden_sizes, num_experts, num_tasks, activation, writer=writer)
        self.q2 = MoEQFunction(obs_dim, act_dim, backbone_hidden_sizes, expert_hidden_sizes, num_experts, num_tasks, activation, writer=writer)
        self.pi.to(self.device)
        self.q1.to(self.device)
        self.q2.to(self.device)

    def act(self, obs, task, deterministic=False):
        with torch.no_grad():
            a, *_ = self.pi(obs, deterministic, False)
            return a