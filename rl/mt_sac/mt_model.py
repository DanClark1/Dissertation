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
    norms = V.norm(dim=-1, keepdim=True)
    V = V / (norms + 1e-6)                      
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
    def __init__(self, input_dim, hidden_size, num_tasks, num_experts=3, activation=F.relu, mu=0.01, phi=0.1, task_embeddings_dim=100, project=False):
        super().__init__()
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        self.mu = mu
        self.phi = phi
        self.representation_store_limit = 1000
        self.task_representations = [torch.zeros((self.representation_store_limit, hidden_size)) for _ in range(num_tasks)]
        self.task_representations_count = [0 for _ in range(num_tasks)]


        self.representations = []
        self.gatings = []
        self.task_list = []
        self.store_limit_b = 2000
        self.store_count = 0

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

        self.project = project
        # not sure if this is always the case so just leaving it here
        self.task_embeddings = nn.Parameter(torch.randn(num_tasks, task_embeddings_dim))

        self.routing_matrix = nn.Parameter(torch.randn(input_dim + task_embeddings_dim, num_experts))

        self.basis_matrix = nn.Parameter(torch.randn(input_dim, input_dim))

        # self.apply(weights_init_) # removed this for now
        self.reset_parameters()

    def get_expert_projection_matrices(self):

        if self.project:
            A = self.basis_matrix
            K, dim = self.num_experts, A.shape[0]
            dsub = dim // K

            # 1) build Cayley orthogonal matrix
            S = A - A.t()                                # skew-symmetric
            I = torch.eye(dim, device=A.device, dtype=A.dtype)
            # solve (I - S) X = (I + S)
            Q = torch.linalg.solve(I - S, I + S)         # (dim, dim), orthogonal

            projection_matrices = []
            for i in range(K):
                Bi = Q[:, i*dsub:(i+1)*dsub]             # (dim, dsub)
                projection_matrices.append(Bi @ Bi.t())
            projection_matrices = torch.stack(projection_matrices, dim=0)  # (K, dim, dim)

            return projection_matrices

        else:
            projection_matrices = []
            for k in range(self.num_experts):
                # Weight of expert k's second linear layer
                W = self.experts[k][2].weight  # [hidden_size, hidden_size]
                # SVD on transpose for row-space basis
                U, S, Vt = torch.linalg.svd(W.t(), full_matrices=False)
                rank = (S > 1e-6).sum().item()
                U = U[:, :rank]
                Pk = U @ U.t()
                projection_matrices.append(Pk)
            return torch.stack(projection_matrices, dim=0)
    

        


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

        # # set every expert weight except the top-k to 0
        # top_k_values, top_k_indices = torch.topk(expert_weights, k=1, dim=-1)
        # expert_weights = torch.zeros_like(expert_weights)
        # expert_weights.scatter_(1, top_k_indices, top_k_values)


        similarity = self.calculate_cosine_similarity(expert_outputs)
        similarity = 0

        # if self.project:
        #     expert_outputs = project_to_unique_subspaces(expert_outputs, self.basis_matrix)
        # else:
        #     expert_outputs = self.orthogonalise(expert_outputs)

        tower_input = torch.einsum('kn,kni->ki', expert_weights, expert_outputs)
        

        if record:
            for i in range(len(tower_input)):
                if self.store_count < self.store_limit_b:
                    self.representations.append(tower_input[i])
                    self.gatings.append(expert_weights[i])
                    self.store_count += 1
                    self.task_list.append(task[i])
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
        from sklearn.decomposition import PCA
        import matplotlib.pyplot as plt
        from sklearn.cluster import KMeans
        import pandas as pd
        import numpy as np
        from sklearn.metrics import normalized_mutual_info_score

        # --- experiment 3 -----
        reps   = torch.stack(self.moe.representations)                             # (N, hidden)
        reps   = reps / reps.norm(dim=-1, keepdim=True)                            # normalize
        raw_tasks = np.array(self.moe.task_list, dtype=int)  
        unique = np.unique(raw_tasks)
        # if your labels aren’t already 0..n_tasks-1, remap them:
        task_map = {old: new for new, old in enumerate(unique)}
        tasks = np.array([task_map[x] for x in raw_tasks], dtype=int)
        # 2) cluster the reps (rows) into K clusters
        K = 10
        rep_km    = KMeans(n_clusters=K).fit(reps.detach().cpu().numpy())
        rep_labels = rep_km.labels_                                                 # cluster index per sample
                                        # “true” expert‐usage cluster

        # 4) build cross‐tab: rows=reps clusters, cols=gating clusters
        ct = pd.crosstab(rep_labels, tasks,
                        rownames=['rep_cluster'], colnames=['gate_cluster'])

        # 5) compute purity
        N = len(rep_labels)
        purity = ct.max(axis=1).sum() / N

        # 6) compute NMI
        nmi = normalized_mutual_info_score(tasks, rep_labels,
                                        average_method='geometric')

        print(f"Purity = {purity:.3f}, NMI = {nmi:.3f}")

        # 7) plot
        fig, ax = plt.subplots(figsize=(6, 4))
        im = ax.imshow(ct, aspect='auto', cmap='viridis')

        # annotate counts
        for i in range(ct.shape[0]):
            for j in range(ct.shape[1]):
                val = ct.iat[i, j]
                ax.text(j, i, val,
                        ha='center', va='center',
                        color='white' if val > ct.values.max()/2 else 'black')

        ax.set_xticks(np.arange(ct.shape[1]))
        ax.set_xticklabels(ct.columns)
        ax.set_yticks(np.arange(ct.shape[0]))
        ax.set_yticklabels(ct.index)
        ax.set_xlabel('Gate Cluster')
        ax.set_ylabel('Rep Cluster')
        ax.set_title(f'Cluster vs. Tasks\nPurity={purity:.3f}, NMI={nmi:.3f}')

        plt.tight_layout()
        plt.savefig('saved/cluster_vs_task_heatmap.svg', format='svg')
        plt.close(fig)


        # --- experiment 2 -----
        reps   = torch.stack(self.moe.representations)                             # (N, hidden)
        reps   = reps / reps.norm(dim=-1, keepdim=True)                            # normalize
        gatings = torch.stack(self.moe.gatings).detach().cpu().numpy()             # (N, K)

        # 2) cluster the reps (rows) into K clusters
        K = 5
        rep_km    = KMeans(n_clusters=K).fit(reps.detach().cpu().numpy())
        rep_labels = rep_km.labels_                                                 # cluster index per sample

        # 3) cluster the gatings (rows) into K clusters
        gate_km     = KMeans(n_clusters=K).fit(gatings)
        gate_labels = gate_km.labels_                                                # “true” expert‐usage cluster

        # 4) build cross‐tab: rows=reps clusters, cols=gating clusters
        ct = pd.crosstab(rep_labels, gate_labels,
                        rownames=['rep_cluster'], colnames=['gate_cluster'])

        # 5) compute purity
        N = len(rep_labels)
        purity = ct.max(axis=1).sum() / N

        # 6) compute NMI
        nmi = normalized_mutual_info_score(gate_labels, rep_labels,
                                        average_method='geometric')

        print(f"Purity = {purity:.3f}, NMI = {nmi:.3f}")

        # 7) plot
        fig, ax = plt.subplots(figsize=(6, 4))
        im = ax.imshow(ct, aspect='auto', cmap='viridis')

        # annotate counts
        for i in range(ct.shape[0]):
            for j in range(ct.shape[1]):
                val = ct.iat[i, j]
                ax.text(j, i, val,
                        ha='center', va='center',
                        color='white' if val > ct.values.max()/2 else 'black')

        ax.set_xticks(np.arange(ct.shape[1]))
        ax.set_xticklabels(ct.columns)
        ax.set_yticks(np.arange(ct.shape[0]))
        ax.set_yticklabels(ct.index)
        ax.set_xlabel('Gate Cluster')
        ax.set_ylabel('Rep Cluster')
        ax.set_title(f'Cluster vs. Gate‐Usage Frequency\nPurity={purity:.3f}, NMI={nmi:.3f}')

        plt.tight_layout()
        plt.savefig('saved/cluster_vs_gate_heatmap.svg', format='svg')
        plt.close(fig)



        



        # --- experiment 1 -----
        task_representations = self.moe.task_representations
        weight_distributions = self.moe.weight_distribution
        for i in range(self.num_tasks):
            weight_distributions[i] = torch.sum(torch.stack(weight_distributions[i]), dim=0)
            weight_distributions[i] = weight_distributions[i] / torch.linalg.vector_norm(weight_distributions[i])


        projection_matrices = self.moe.get_expert_projection_matrices()
        mean_norm = []
        means = []
        variances = []
        angular_variances = []
        all_weights = []
        task_direction_affinities = [[] for _ in range(self.num_tasks)]
        weight_dist = [[] for _ in range(self.num_tasks)]
        for i in range(self.num_tasks):
            print(i)
            # weights = weight_distributions[i]
            # weights = torch.stack(weights)
            # all_weights.append(weights)
            reps = task_representations[i]

            for j in range(self.moe.num_experts):
                task_direction_affinities[i].append(compute_projection_ratio(projection_matrices[j], reps))

            X = reps
            X_norm = X / torch.linalg.norm(X, dim=-1, keepdim=True)
            mean = X_norm.mean(dim=0, keepdim=True)
            X_centered = X_norm - mean
            cov = (X_centered.T @ X_centered) / (X_norm.shape[0] - 1)
            sign, logabsdet = torch.linalg.slogdet(cov)
            gen_var = sign * torch.exp(logabsdet)
            variances.append(gen_var)


            mask = reps.norm(dim=1) != 0
            reps = reps[mask]
            mean_norm.append(reps.norm(dim=1).mean())
            means.append(reps.mean(dim=0) / reps.norm(dim=1).mean())
            #variances.append(reps.var(dim=0))
            normalised_representations = reps / reps.norm(dim=1, keepdim=True)
            normalised_mean = means[i] / means[i].norm()
            dots = (normalised_representations * normalised_mean).sum(dim=-1).clamp(-1.0, +1.0)
            angles = torch.acos(dots)
            angular_variances.append(angles.var())
        means = torch.stack(means)
        angular_variances = torch.stack(angular_variances)
        variances = torch.stack(variances)
        return means, variances, angular_variances, mean_norm, weight_distributions, task_direction_affinities




def compute_projection_ratio(P, H):
    '''
    P -> projection matrix (d, d)
    H -> features (N, d)
    '''
    H_proj = H @ P.T 
    
    norm_orig = H.norm(dim=1)      # ||h_i||
    norm_proj = H_proj.norm(dim=1) # ||P h_i||
    
    eps = 1e-8
    ratios = norm_proj / (norm_orig + eps)
    
    return ratios.mean()