import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)




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
        reg_loss_term = - (1 / self.num_experts) * self.mu * (torch.sum(attention_weights + eps, axis=-1))
        return tower_input, reg_loss_term

class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, expert_hidden_dim=400):
        super(QNetwork, self).__init__()

        # Q1 architecture
        self.linear1_1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2_1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3_1 = nn.Linear(hidden_dim, 1)
        self.moe_1 = MoELayer(hidden_dim, [expert_hidden_dim], 2, 10, nn.ReLU(), 0.01)

        # Q2 architecture
        self.linear1_2 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2_2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3_2 = nn.Linear(hidden_dim, 1)

        self.moe_2 = MoELayer(hidden_dim, [expert_hidden_dim], 2, 10, nn.ReLU(), 0.01)

        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)
        
        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2


class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(GaussianPolicy, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)


class DeterministicPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(DeterministicPolicy, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, num_actions)
        self.noise = torch.Tensor(num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = 1.
            self.action_bias = 0.
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
        return mean

    def sample(self, state):
        mean = self.forward(state)
        noise = self.noise.normal_(0., std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.), mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.noise = self.noise.to(device)
        return super(DeterministicPolicy, self).to(device)
