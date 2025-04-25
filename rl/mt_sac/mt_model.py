import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import utils
import cProfile
import pstats
import math
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

def project_to_unique_subspaces(
    U: torch.Tensor,
    A: torch.Tensor
) -> torch.Tensor:
    """
    Args:
      U: (batch, K, dim)                — MoE outputs
      A: (dim, dim)                     — unconstrained parameter
    Returns:
      V: (batch, K, dim)                — each expert in its own orthogonal subspace
    """
    batch, K, dim = U.shape
    dsub = dim // K

    # 1) build Cayley orthogonal matrix
    S = A - A.t()                                # skew-symmetric
    I = torch.eye(dim, device=A.device, dtype=A.dtype)
    # solve (I - S) X = (I + S)
    Q = torch.linalg.solve(I - S, I + S)         # (dim, dim), orthogonal

    # 2) slice into K sub-bases
    #    Q[:, i*dsub:(i+1)*dsub] is the basis for expert i
    V = torch.zeros_like(U)
    for i in range(K):
        Bi = Q[:, i*dsub:(i+1)*dsub]             # (dim, dsub)
        ui = U[:, i]                             # (batch, dim)
        coords = ui @ Bi                         # (batch, dsub)
        V[:, i]  = coords @ Bi.t()               # back to (batch, dim)

    return V



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
    def __init__(self, input_dim, hidden_size, num_tasks, num_experts=3, activation=F.relu, mu=0.01, phi=0.1, task_embeddings_dim=100):
        super().__init__()
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        self.mu = mu
        self.phi = phi
        self.representation_store_limit = 200
        self.task_representations = [torch.zeros((self.representation_store_limit, hidden_size)) for _ in range(num_tasks)]
        self.task_representations_count = [0 for _ in range(num_tasks)]

        self.weight_distribution = [[] for _ in range(num_tasks)]

        # Create expert networks (each expert is an MLP)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU()
            ) for _ in range(num_experts)
        ])

        # not sure if this is always the case so just leaving it here
        self.task_embeddings = nn.Parameter(torch.randn(num_tasks, task_embeddings_dim))

        self.routing_matrix = nn.Parameter(torch.randn(input_dim + task_embeddings_dim, num_experts))

        self.basis_matrix = nn.Parameter(torch.randn(input_dim, input_dim))

        # self.apply(weights_init_) # removed this for now
        self.reset_parameters()

    def calculate_cosine_similarity(self, expert_outputs):
        # expert_outputs shape: (batch_size, num_experts, hidden_size)
        # Normalize across the hidden dimension for all samples
        normalized = F.normalize(expert_outputs, p=2, dim=-1)  # (batch_size, num_experts, hidden_size)
        
        # Compute pairwise cosine similarities via batched matrix multiplication
        # Result shape: (batch_size, num_experts, num_experts)
        sim_matrix = torch.bmm(normalized, normalized.transpose(1, 2))
        
        # Create a mask to select the upper-triangular (non-diagonal) entries for each sample
        mask = torch.triu(torch.ones(self.num_experts, self.num_experts, device=expert_outputs.device), diagonal=1).bool()
        sim_values = sim_matrix[:, mask]  # (batch_size, num_pairs)
        
        return sim_values.mean()

    def reset_parameters(self):
        # Use Xavier uniform initialization with gain 1
        # nn.init.xavier_uniform_(self.task_queries, gain=1)
        # nn.init.xavier_uniform_(self.key_matricies, gain=1)
        # nn.init.xavier_uniform_(self.value_matricies, gain=1)

        nn.init.kaiming_uniform_(self.task_embeddings, a=math.sqrt(5))  # Use `a=sqrt(5)` as recommended for uniform init
        nn.init.kaiming_uniform_(self.routing_matrix, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.basis_matrix, a=math.sqrt(5))


    def orthogonalise(self,x):

        """
        input as [batch_size, num_experts, hidden_size]
        """

        x = x.permute(1, 0, 2).contiguous()
        # from here it needs to be [n_models,n_samples,dim]
        x1 = torch.transpose(x, 0,1)
        basis = torch.unsqueeze(x1[:, 0, :] / (torch.unsqueeze(torch.linalg.norm(x1[:, 0, :], axis=1), 1)), 1)

        for i in range(1, x1.shape[1]):
            v = x1[:, i, :]
            v = torch.unsqueeze(v, 1)
            w = v - torch.matmul(torch.matmul(v, torch.transpose(basis, 2, 1)), basis)
            wnorm = w / (torch.unsqueeze(torch.linalg.norm(w, axis=2), 2))
            basis = torch.cat([basis, wnorm], axis=1)

        basis = torch.transpose(basis,0,1)

        # reshape back to (batch_size, num_experts, hidden_size)
        basis = basis.permute(1, 0, 2).contiguous()
        return basis




    def forward(self, backbone_output, task, record=False):

        expert_weights = F.softmax(torch.einsum('ni,ij->nj', torch.cat([backbone_output, self.task_embeddings[task]], dim=-1), self.routing_matrix), dim=-1)
        # top-k weights:
        #expert_weights, _ = torch.topk(expert_weights, k=2, dim=-1)
        expert_outputs = torch.stack([expert(backbone_output) for expert in self.experts], dim=1)

        # set every expert weight except the top-k to 0
        top_k_values, top_k_indices = torch.topk(expert_weights, k=1, dim=-1)
        expert_weights = torch.zeros_like(expert_weights)
        expert_weights.scatter_(1, top_k_indices, top_k_values)


        similarity = self.calculate_cosine_similarity(expert_outputs)
        similarity = 0

        expert_outputs = self.orthogonalise(expert_outputs)
        tower_input = torch.einsum('kn,kni->ki', expert_weights, expert_outputs)
        

        if record:
            for i in range(len(tower_input)):
                if self.task_representations_count[task[i]] < self.representation_store_limit:
                    self.weight_distribution[task[i]].append(expert_weights[i])
                    self.task_representations[task[i]][self.task_representations_count[task[i]]] = tower_input[i]
                    self.task_representations_count[task[i]] += 1


        # regularisation term
        eps = torch.ones_like(expert_weights) / (1e6)
        reg_loss_term = - (1 / self.num_experts) * self.mu * (torch.sum(torch.log(expert_weights + eps), dim=-1))
        #reg_loss_term = self.phi * torch.abs(similarity)
        return tower_input, reg_loss_term
    
    


class QNetwork(nn.Module):
    def __init__(self, obs_size, action_size, hidden_dim, num_tasks=10, num_experts=3, writer=None):
        super(QNetwork, self).__init__()

        self.num_tasks = num_tasks

        # self.single_moe_1 = MoELayer(obs_size-num_tasks, 1, num_tasks)
        # self.single_moe_2 = MoELayer(obs_size-num_tasks, 1, num_tasks)

        # Q1 architecture
        self.linear1_1 = nn.Linear(obs_size-num_tasks + action_size, hidden_dim)
        self.linear2_1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3_1 = nn.Linear(hidden_dim, hidden_dim)
        self.moe_1 = MoELayer(hidden_dim, hidden_dim, num_tasks, num_experts=num_experts)
        self.linear4_1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear5_1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6_1 = nn.Linear(hidden_dim, 1)

        # Q1 architecture
        self.linear1_2 = nn.Linear(obs_size-num_tasks + action_size, hidden_dim)
        self.linear2_2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3_2 = nn.Linear(hidden_dim, hidden_dim)
        self.moe_2 = MoELayer(hidden_dim, hidden_dim, num_tasks, num_experts=num_experts)
        self.linear4_2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear5_2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6_2 = nn.Linear(hidden_dim, 1)

        # self.apply(weights_init_)

    def forward(self, obs, action, record=False):
        obs, task = utils.format_obs(obs, num_tasks=self.num_tasks)
        xu = torch.cat([obs, action], 1)
        
        x1 = F.relu(self.linear1_1(xu))
        x1 = F.relu(self.linear2_1(x1))
        x1 = F.relu(self.linear3_1(x1))
        x1, reg_loss_1 = self.moe_1(x1, task, record=record)
        x1 = F.relu(self.linear4_1(x1))
        x1 = F.relu(self.linear5_1(x1))
        x1 = self.linear6_1(x1)


        x2 = F.relu(self.linear1_2(xu))
        x2 = F.relu(self.linear2_2(x2))
        x2 = F.relu(self.linear3_2(x2))
        x2, reg_loss_2 = self.moe_2(x2, task, record=record)
        x2 = F.relu(self.linear4_2(x2))
        x2 = F.relu(self.linear5_2(x2))
        x2 = self.linear6_2(x2)

        return x1, x2, reg_loss_1 + reg_loss_2


class GaussianPolicy(nn.Module):
    def __init__(self, obs_size, action_size, hidden_dim, action_space=None, num_tasks=10, num_experts=3, writer=None):
        super(GaussianPolicy, self).__init__()

        self.num_tasks = num_tasks
        
        self.linear1 = nn.Linear(obs_size-num_tasks, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.moe = MoELayer(hidden_dim, hidden_dim, num_tasks, num_experts=num_experts) 

        self.single_moe = MoELayer(obs_size-num_tasks, hidden_dim, num_tasks)

        self.mean_linear = nn.Linear(hidden_dim, action_size)
        self.log_std_linear = nn.Linear(hidden_dim, action_size)

        # self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, obs, record=False):
        obs, task = utils.format_obs(obs, num_tasks=self.num_tasks)
        x = F.relu(self.linear1(obs))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x, reg_loss = self.moe(x, task, record=record)
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std, reg_loss

    def sample(self, state, record=False):
        mean, log_std, reg_loss = self.forward(state, record=record)
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
        return action, log_prob, mean, reg_loss

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)
    
    def calculate_task_variance(self):
        task_representations = self.moe.task_representations
        weight_distributions = self.moe.weight_distribution
        mean_norm = []
        means = []
        variances = []
        angular_variances = []
        for i in range(self.num_tasks):
            weights = weight_distributions[i]
            weights = torch.stack(weights)
            weights = weights.mean(dim=0)
            weights = weights / weights.norm()
            reps = task_representations[i]
            mask = reps.norm(dim=1) != 0
            reps = reps[mask]
            mean_norm.append(reps.norm(dim=1).mean())
            means.append(reps.mean(dim=0) / reps.norm(dim=1).mean())
            variances.append(reps.var(dim=0))
            normalised_representations = reps / reps.norm(dim=1, keepdim=True)
            normalised_mean = means[i] / means[i].norm()
            dots = (normalised_representations * normalised_mean).sum(dim=-1).clamp(-1.0, +1.0)
            angles = torch.acos(dots)
            angular_variances.append(angles.var())
        means = torch.stack(means)
        angular_variances = torch.stack(angular_variances)
        variances = torch.stack(variances)
        return means, variances, angular_variances, mean_norm, weights
