import argparse
import datetime
import gymnasium as gym

import numpy as np
import itertools
import torch
import random
import metaworld
import imageio
from PIL import Image, ImageDraw
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
# Import your agent classes as before
from sac.sac import SAC
from mt_sac.mt_sac import MT_SAC
from big_sac.big_sac import BIG_SAC
from equal_expert_mt_sac.mt_sac import EE_MT_SAC
from replay_memory import ReplayMemory

# Import the SubprocVecEnv wrapper from Stable Baselines3
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# 1. Define a Wrapper to Append One-Hot Task Encoding to Observations
# -----------------------------------------------------------------------------
class OneHotTaskWrapper(gym.Wrapper):
    """
    Gymnasium wrapper that appends a one-hot task encoding to the observation.
    """
    def __init__(self, env, task_index, total_tasks):
        super(OneHotTaskWrapper, self).__init__(env)
        self.task_index = task_index
        self.total_tasks = total_tasks
        
        # Modify the observation space to include the one-hot encoding.
        orig_space = env.observation_space
        low = np.concatenate([orig_space.low, np.zeros(total_tasks, dtype=orig_space.dtype)])
        high = np.concatenate([orig_space.high, np.ones(total_tasks, dtype=orig_space.dtype)])
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=orig_space.dtype)

    def reset(self, seed=None, options=None):
        # Use the Gymnasium API: reset returns (obs, info)
        obs, info = self.env.reset(seed=seed, options=options)
        return self._append_one_hot(obs), info

    def step(self, action):
        # Gymnasium step returns (obs, reward, terminated, truncated, info)
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._append_one_hot(obs), reward, terminated, truncated, info

    def _append_one_hot(self, obs):
        one_hot = np.zeros(self.total_tasks, dtype=obs.dtype)
        one_hot[self.task_index] = 1.0
        return np.concatenate([obs, one_hot], axis=-1)


# -----------------------------------------------------------------------------
# 2. Environment-Creation Function for SubprocVecEnv
# -----------------------------------------------------------------------------
def make_env_func(env_cls, task, task_index, total_tasks, seed, rank):
    """
    Returns a function that creates a gym environment with a specified task.
    The environment is wrapped with OneHotTaskWrapper.
    """
    def _init():
        env = env_cls()
        env.set_task(task)
        env.seed(seed + rank)
        env = OneHotTaskWrapper(env, task_index, total_tasks)
        return env
    return _init


def analyse_moe(agent):
    agent.critic.moe_1.save_moe_info()
    agent.critic.moe_2.save_moe_info()
    agent.policy.moe.save_moe_info()


# -----------------------------------------------------------------------------
# 3. Main Training Script using SubprocVecEnv
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
    parser.add_argument('--env-name', default="",
                        help='Mujoco Gym environment (default: HalfCheetah-v2)')
    parser.add_argument('--run_name', default="", help='Name of the run')
    parser.add_argument('--policy', default="Gaussian",
                        help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
    parser.add_argument('--eval', type=bool, default=True,
                        help='Evaluate the policy every few episodes (default: True)')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='Discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                        help='Target smoothing coefficient τ (default: 0.005)')
    parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                        help='Learning rate (default: 0.0003)')
    parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                        help='Temperature parameter α (default: 0.2)')
    parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                        help='Automatically adjust α (default: False)')
    parser.add_argument('--seed', type=int, default=123456, metavar='N',
                        help='Random seed (default: 123456)')
    parser.add_argument('--batch_size', type=int, default=1024, metavar='N',
                        help='Batch size (default: 1024)')
    parser.add_argument('--num_steps', type=int, default=1000, metavar='N',
                        help='Maximum number of steps (default: 1000001)')
    parser.add_argument('--hidden_size', type=int, default=400, metavar='N',
                        help='Hidden size (default: 400)')
    parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                        help='Model updates per simulator step (default: 1)')
    parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                        help='Steps sampling random actions (default: 10000)')
    parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                        help='Value target update per number of updates (default: 1)')
    parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                        help='Size of replay buffer (default: 1000000)')
    parser.add_argument('--cuda', action="store_true",
                        help='Run on CUDA (default: False)')
    parser.add_argument('--model', default="",
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')
    parser.add_argument('--load_model', type=str, default="", metavar='N',)
    parser.add_argument('--do_50', action="store_true", help='Do 50 tasks (default: False)')
    
    args = parser.parse_args()

    print('runs/{}_{}'.format(
        args.load_model,
        datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # -------------------------------
    # Create the vectorised environments
    # -------------------------------

    if args.do_50 and False:
        mw = metaworld.MT50(seed=args.seed)
    else:
        mw = metaworld.MT10(seed=args.seed)
    total_tasks = len(mw.train_classes)  # e.g. 10 tasks in MT10

    env_fns = []
    task_names = []
    num_parallel_envs = 0

    # For simplicity, we create one environment per task. You could add more copies if desired.
    for i, (name, env_cls) in enumerate(mw.train_classes.items()):
        # Select a random task from the candidates for this environment
        # im pretty sure this just selects the 1 class
        task_candidates = [task for task in mw.train_tasks if task.env_name == name]
        task = random.choice(task_candidates)
        task_names.append(name)
        # Create the environment function; 'rank' can simply be the task index here.
        env_fn_1 = make_env_func(env_cls, task, task_index=i, total_tasks=total_tasks,
                               seed=args.seed, rank=i)
        env_fn_2 = make_env_func(env_cls, task, task_index=i, total_tasks=total_tasks,
                               seed=args.seed, rank=i)
        env_fns.append(env_fn_1)
        num_parallel_envs += 1

    # Create a vectorised environment using SubprocVecEnv.
    vector_env = SubprocVecEnv(env_fns)

    num_envs = vector_env.num_envs  # number of parallel environments

    # The observation space now includes the one-hot task encoding.
    obs_dim = vector_env.observation_space.shape[0]
    action_space = vector_env.action_space  # assumed same for all environments

    writer = SummaryWriter('runs/{}_{}'.format(
        args.load_model if args.load_model else 'SAC',
        datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    ))

    
    # -------------------------------
    # Instantiate the SAC (or variant) agent
    # -------------------------------
    if args.model == 'moe':
        print('MOE')
        agent = MT_SAC(obs_dim, action_space, writer, args, debug=True, num_experts=(10 if args.do_50 else 3), num_tasks=(50 if args.do_50 else 10))
    elif args.model == 'ee':
        print('EE')
        agent = EE_MT_SAC(obs_dim, action_space, writer, args, debug=True,num_experts=(10 if args.do_50 else 3), num_tasks=(50 if args.do_50 else 10))
    elif args.model == 'big':
        print('BIG')
        agent = BIG_SAC(obs_dim, action_space, writer, args, debug=True,)
    else:
        print('SAC')
        agent = SAC(obs_dim, action_space, writer, args, debug=True,)


    if args.load_model:

        agent.load_checkpoint('checkpoints/'+args.load_model)
    else:
        raise ValueError('No model specified to load')



    memory = ReplayMemory(args.replay_size, args.seed)

    # -------------------------------
    # Training Loop with Vectorised Environment
    # -------------------------------
    total_numsteps = 0
    updates = 0

    # Reset all environments to get initial batch of observations.
    states = vector_env.reset()  # shape: (num_envs, obs_dim)



    avg_reward = 0
    states = vector_env.reset()
    eval_episodes = 5
    avg_rewards = []
    avg_episode_rewards = np.zeros((num_envs,))

    for _ in range(eval_episodes):
        done_flags = [False] * num_envs
        eval_obs = vector_env.reset()
        episode_return = np.zeros(num_envs)

        # loop until every episdode is done
        while not all(done_flags):
            eval_actions = agent.select_action_batch(eval_obs, evaluate=True)
            next_obs, eval_rewards, eval_dones, _ = vector_env.step(eval_actions)

            for i in range(num_envs):

                if not done_flags[i]:
                    episode_return[i] += eval_rewards[i]

            done_flags = [done_flags[i] or eval_dones[i] for i in range(num_envs)]
            eval_obs = next_obs

        avg_episode_rewards += episode_return
        avg_rewards.append(episode_return.mean())

    avg_episode_rewards /= eval_episodes
    # Create a bar chart using the task names and the corresponding average rewards.
    fig, ax = plt.subplots()
    ax.bar(task_names, avg_episode_rewards)
    ax.set_xlabel("Task")
    ax.set_ylabel("Average Reward")
    ax.set_title("Evaluation Average Reward per Task")

    print(task_names)
    # Rotate the tick labels and align them so they don't overlap
    plt.xticks(rotation=45, ha='right')

    # Adjust layout to make sure labels fit
    plt.tight_layout()

    # Log the figure to TensorBoard.
    writer.add_figure("evaluation/average_reward_bar", fig, total_numsteps)

    print(avg_episode_rewards.mean())

    vector_env.close()

    analyse_moe(agent)

if __name__ == '__main__':
    main()
