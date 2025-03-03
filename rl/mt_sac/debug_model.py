import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import utils
import cProfile
import pstats
import matplotlib.pyplot as plt
import numpy as np
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import pdist

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
    def __init__(self, input_dim, hidden_size, num_tasks, num_experts=3, activation=F.relu, writer=None, mu=0.01, name=""):
        super().__init__()
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        self.mu = mu
        self.name = name
        self.writer = writer
        self.expert_usage = [[] for _ in range(self.num_tasks)]
        self.cosine_similarities = []

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
        task_queries_dim = hidden_size
        self.task_queries = nn.Parameter(torch.randn(num_tasks, task_queries_dim))

        # Matrices to compute keys and values from the expert outputs.
        self.key_matricies = nn.Parameter(torch.randn(num_experts, task_queries_dim, hidden_size))
        self.value_matricies = nn.Parameter(torch.randn(num_experts, task_queries_dim, hidden_size))

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

    def forward(self, backbone_output, task):

        expert_outputs = torch.stack([expert(backbone_output) for expert in self.experts], dim=1)

        # Compute keys and values using einsum.
        expert_keys = torch.einsum('kli,lij->klj', expert_outputs, self.key_matricies)
        expert_values = torch.einsum('kli,lij->klj', expert_outputs, self.value_matricies)

        similarity = self.calculate_cosine_similarity(expert_values)
        self.cosine_similarities.append(similarity.detach())
        

        # Use the task query (indexed by the task) to compute attention scores.
        # Make sure to adjust dimensions if your task variable isn’t batch–wise.
        attention_scores = torch.einsum('kni,ki->kn', expert_keys, self.task_queries[task])
        attention_weights = torch.softmax(attention_scores, dim=-1)

        # place_holder = torch.ones_like(attention_weights) / attention_weights.size(-1)

        for weights, ind_task in zip(attention_weights, task):
            self.expert_usage[ind_task].append(weights)

        # Aggregate expert outputs.
        tower_input = torch.einsum('kn,kni->ki', attention_weights, expert_values)

        # Optionally compute a regularization term.
        eps = torch.ones_like(attention_weights) / (1e6)
        reg_loss_term = - (1 / self.num_experts) * self.mu * (torch.sum(torch.log(attention_weights + eps), dim=-1))
        reg_loss_term += self.mu * similarity
        return tower_input, reg_loss_term
    
    def save_moe_info(self):
        """
        1) Creates a grouped bar chart showing average usage of each expert per task.
        2) Creates a heatmap of the pairwise cosine similarities between task embeddings.
        3) Logs both figures to TensorBoard.
        """
        # --------------------------------------------------------------------
        # 1) Compute and plot average expert usage across tasks
        # --------------------------------------------------------------------

        print('average similarity:', np.array(self.cosine_similarities).mean())
        usage_per_expert = torch.zeros(self.num_experts, self.num_tasks)

        for i in range(self.num_tasks):
            # Each entry in self.expert_usage[i] is a 1D tensor of shape [num_experts]
            # usage_matrix shape: [num_experts, N], where N = number of samples recorded for this task
            usage_matrix = torch.stack(self.expert_usage[i], dim=0).T

            # Optionally, log histogram of usage for each expert
            for expert_idx in range(self.num_experts):
                self.writer.add_histogram(
                    f'expert_usage/{self.name}_task:_{expert_idx}',
                    usage_matrix[expert_idx],
                    i
                )

            # mean_usage_for_this_task: [num_experts]
            mean_usage_for_this_task = usage_matrix.mean(dim=1)
            usage_per_expert[:, i] = mean_usage_for_this_task

        # Create grouped bar chart
        fig, ax = plt.subplots(figsize=(7, 5))

        x_positions = np.arange(self.num_tasks)
        bar_width   = 0.8 / self.num_experts

        for e in range(self.num_experts):
            expert_x = x_positions + e * bar_width
            ax.bar(
                expert_x,
                usage_per_expert[e].detach().numpy(),
                width=bar_width,
                label=f'Expert {e+1}'
            )

        ax.set_xlabel("Task")
        ax.set_ylabel("Average Usage")
        ax.set_title("Average Expert Usage per Task")

        # Example task labels
        task_labels = [
            'reach-v2', 'push-v2', 'pick-place-v2', 'door-open-v2',
            'drawer-open-v2', 'drawer-close-v2', 'button-press-topdown-v2',
            'peg-insert-side-v2', 'window-open-v2', 'window-close-v2'
        ]
        # Center the tick labels
        ax.set_xticks(x_positions + bar_width*(self.num_experts-1)/2)
        ax.set_xticklabels(task_labels, rotation=45, ha='right')

        ax.legend()
        plt.tight_layout()

        # Log the usage figure
        self.writer.add_figure("evaluation/average_expert_usage", fig, global_step=0)

        # --------------------------------------------------------------------
        # 2) Compute and plot pairwise cosine similarities of task embeddings
        # --------------------------------------------------------------------
        # task_queries shape: [num_tasks, hidden_size]
        with torch.no_grad():
            # Normalize each task embedding
            normalized = self.task_queries / (self.task_queries.norm(dim=1, keepdim=True) + 1e-9)
            # Pairwise cosine similarities: [num_tasks, num_tasks]
            pairwise_sims = normalized @ normalized.T

        fig2, ax2 = plt.subplots(figsize=(6, 5))
        im = ax2.imshow(pairwise_sims.detach().cpu().numpy(), cmap='viridis', aspect='auto')

        # Add colorbar
        cbar = fig2.colorbar(im, ax=ax2)
        cbar.set_label('Cosine Similarity', rotation=90)

        ax2.set_title("Task Embedding Cosine Similarities")
        ax2.set_xlabel("Task")
        ax2.set_ylabel("Task")

        ax2.set_xticks(range(self.num_tasks))
        ax2.set_yticks(range(self.num_tasks))
        ax2.set_xticklabels(task_labels, rotation=45, ha='right')
        ax2.set_yticklabels(task_labels)

        plt.tight_layout()

        # Log the similarity figure
        self.writer.add_figure("evaluation/task_embedding_similarity", fig2, global_step=0)
        
        # --------------------------------------------------------------------
        # 3) Hierarchical clustering on task embeddings and dendrogram plotting
        # --------------------------------------------------------------------
        with torch.no_grad():
            # Convert task embeddings to numpy for clustering.
            task_embeddings_np = self.task_queries.detach().cpu().numpy()
            # Compute pairwise distances (Euclidean distance in this example)
            distances = pdist(task_embeddings_np, metric='euclidean')
            # Compute the linkage matrix using Ward's method
            linkage_matrix = sch.linkage(distances, method='ward')

        # Plot the dendrogram.
        fig3, ax3 = plt.subplots(figsize=(8, 5))
        sch.dendrogram(linkage_matrix, labels=task_labels, ax=ax3)
        ax3.set_title("Hierarchical Clustering Dendrogram of Task Embeddings")
        ax3.set_xlabel("Task")
        ax3.set_ylabel("Euclidean Distance")
        # Rotate x-axis labels to be vertical to avoid overlapping.
        plt.setp(ax3.get_xticklabels(), rotation=90, ha='right')
        plt.tight_layout()
        self.writer.add_figure("evaluation/task_embedding_hierarchical_clustering", fig3, global_step=0)





class DebugQNetwork(nn.Module):
    def __init__(self, obs_size, action_size, hidden_dim, num_tasks=10, writer=None, num_experts=3):
        super(DebugQNetwork, self).__init__()

        self.num_tasks = num_tasks

        # self.single_moe_1 = MoELayer(obs_size-num_tasks, 1, num_tasks)
        # self.single_moe_2 = MoELayer(obs_size-num_tasks, 1, num_tasks)

        # Q1 architecture
        self.linear1_1 = nn.Linear(obs_size-num_tasks + action_size, hidden_dim)
        self.linear2_1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3_1 = nn.Linear(hidden_dim, hidden_dim)
        self.moe_1 = MoELayer(hidden_dim, hidden_dim, num_tasks, writer=writer, num_experts=num_experts, name="critic_1")
        self.linear4_1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear5_1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6_1 = nn.Linear(hidden_dim, 1)

        # Q1 architecture
        self.linear1_2 = nn.Linear(obs_size-num_tasks + action_size, hidden_dim)
        self.linear2_2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3_2 = nn.Linear(hidden_dim, hidden_dim)
        self.moe_2 = MoELayer(hidden_dim, hidden_dim, num_tasks, writer=writer, num_experts=num_experts, name="critic_2")
        self.linear4_2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear5_2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6_2 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, obs, action):
        obs, task = utils.format_obs(obs, num_tasks=self.num_tasks)
        xu = torch.cat([obs, action], 1)
        
        x1 = F.relu(self.linear1_1(xu))
        x1 = F.relu(self.linear2_1(x1))
        x1 = F.relu(self.linear3_1(x1))
        x1, reg_loss_1 = self.moe_1(x1, task)
        x1 = F.relu(self.linear4_1(x1))
        x1 = F.relu(self.linear5_1(x1))
        x1 = self.linear6_1(x1)


        x2 = F.relu(self.linear1_2(xu))
        x2 = F.relu(self.linear2_2(x2))
        x2 = F.relu(self.linear3_2(x2))
        x2, reg_loss_2 = self.moe_2(x2, task)
        x2 = F.relu(self.linear4_2(x2))
        x2 = F.relu(self.linear5_2(x2))
        x2 = self.linear6_2(x2)

        return x1, x2, reg_loss_1 + reg_loss_2


class DebugGaussianPolicy(nn.Module):
    def __init__(self, obs_size, action_size, hidden_dim, action_space=None, num_tasks=10, writer=None, num_experts=3):
        super(DebugGaussianPolicy, self).__init__()

        self.num_tasks = num_tasks
        
        self.linear1 = nn.Linear(obs_size-num_tasks, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.moe = MoELayer(hidden_dim, hidden_dim, num_tasks, num_experts=num_experts, writer=writer, name="policy") 

        self.single_moe = MoELayer(obs_size-num_tasks, hidden_dim, num_tasks)

        self.mean_linear = nn.Linear(hidden_dim, action_size)
        self.log_std_linear = nn.Linear(hidden_dim, action_size)

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

    def forward(self, obs):
        obs, task = utils.format_obs(obs, num_tasks=self.num_tasks)
        x = F.relu(self.linear1(obs))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x, reg_loss = self.moe(x, task)
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std, reg_loss

    def sample(self, state):
        mean, log_std, reg_loss = self.forward(state)
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
        return super(DebugGaussianPolicy, self).to(device)
