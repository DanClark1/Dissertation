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

# Import your agent classes as before
from sac.sac import SAC
from mt_sac.mt_sac import MT_SAC
from big_sac.big_sac import BIG_SAC
from replay_memory import ReplayMemory

# Import the SubprocVecEnv wrapper from Stable Baselines3
from stable_baselines3.common.vec_env import SubprocVecEnv

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
    parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
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
    parser.add_argument('--use_moe', action="store_true", help='Use MOE (default: False)')
    parser.add_argument('--use_big', action="store_true", help='Use BIG (default: False)')
    
    args = parser.parse_args()

    print('runs/{}_{}'.format(
        args.run_name if args.run_name else 'SAC',
        datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # -------------------------------
    # Create the vectorised environments
    # -------------------------------
    mt10 = metaworld.MT10(seed=args.seed)
    total_tasks = len(mt10.train_classes)  # e.g. 10 tasks in MT10

    env_fns = []
    task_names = []
    # For simplicity, we create one environment per task. You could add more copies if desired.
    for i, (name, env_cls) in enumerate(mt10.train_classes.items()):
        # Select a random task from the candidates for this environment
        task_candidates = [task for task in mt10.train_tasks if task.env_name == name]
        task = random.choice(task_candidates)
        task_names.append(name)
        # Create the environment function; 'rank' can simply be the task index here.
        env_fn = make_env_func(env_cls, task, task_index=i, total_tasks=total_tasks,
                               seed=args.seed, rank=i)
        env_fns.append(env_fn)

    # Create a vectorised environment using SubprocVecEnv.
    vector_env = SubprocVecEnv(env_fns)
    num_envs = vector_env.num_envs  # number of parallel environments

    # The observation space now includes the one-hot task encoding.
    obs_dim = vector_env.observation_space.shape[0]
    action_space = vector_env.action_space  # assumed same for all environments

    
    # -------------------------------
    # Instantiate the SAC (or variant) agent
    # -------------------------------
    if args.use_moe:
        agent = MT_SAC(obs_dim, action_space, args)
    elif args.use_big:
        agent = BIG_SAC(obs_dim, action_space, args)
    else:
        agent = SAC(obs_dim, action_space, args)

    writer = SummaryWriter('runs/{}_{}'.format(
        args.run_name if args.run_name else 'SAC',
        datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    ))

    print('Logging at:', 'runs/{}_{}-SAC_{}_{}'.format(
        datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        'moe' if args.use_moe else '',
        args.policy,
        "autotune" if args.automatic_entropy_tuning else ""
    ))

    # -------------------------------
    # Create replay memory
    # -------------------------------
    memory = ReplayMemory(args.replay_size, args.seed)

    # -------------------------------
    # Training Loop with Vectorised Environment
    # -------------------------------
    total_numsteps = 0
    updates = 0

    # Reset all environments to get initial batch of observations.
    states = vector_env.reset()  # shape: (num_envs, obs_dim)

    while total_numsteps < args.num_steps:
        # Select actions for all environments.
        if total_numsteps < args.start_steps:
            actions = np.array([action_space.sample() for _ in range(num_envs)])
        else:
            # It is assumed that your SAC agent has a method to select actions in batch.
            actions = agent.select_action_batch(states)

        # Step all environments in parallel.
        next_states, rewards, dones, infos = vector_env.step(actions)

        # Save each transition to replay memory.
        for i in range(num_envs):
            mask = 0.0 if dones[i] else 1.0
            memory.push(states[i], actions[i], rewards[i], next_states[i], mask)

        states = next_states
        total_numsteps += num_envs

        # Update the agent when enough samples have been collected.
        if len(memory) > args.batch_size:
            for _ in range(args.updates_per_step):
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(
                    memory, args.batch_size, updates)
                writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                writer.add_scalar('loss/policy', policy_loss, updates)
                writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                updates += 1

            
            
        for i, name in enumerate(task_names):
            writer.add_scalar(f'rewards/{name}', rewards[i], total_numsteps)

        # Optionally, handle individual environment resets.
        # Many vectorised environments auto-reset on done, but if not, you can do:
        for i, done in enumerate(dones):
            if done:
                # Reset the i-th environment
                # SubprocVecEnv does not offer a built-in partial reset,
                # so we call the reset method on that environment explicitly.
                states[i] = vector_env.env_method('reset', indices=i)[0][0]

        # Logging every few steps
        if total_numsteps % (num_envs * 1000) == 0:
            print("Total Steps: {}".format(total_numsteps))

        # record embeddings every 5% of total steps
        if total_numsteps % (args.num_steps // 20) == 0:
            agent.log_embeddings(writer, total_numsteps, task_names)  

        
        if total_numsteps % 1000 == 0 and args.eval is True:
            avg_reward = 0
            states = vector_env.reset()
            eval_episodes = 5
            episode_rewards = []
            for _ in range(eval_episodes):
                done_flags = [False] * num_envs
                eval_obs = vector_env.reset()
                episode_return = np.zeros(num_envs)
                while not all(done_flags):
                    eval_actions = agent.select_action_batch(eval_obs, evaluate=True)
                    next_obs, eval_rewards, eval_dones, _ = vector_env.step(eval_actions)
                    for i in range(num_envs):
                        if not done_flags[i]:
                            episode_return[i] += eval_rewards[i]
                    done_flags = [done_flags[i] or eval_dones[i] for i in range(num_envs)]
                    eval_obs = next_obs
                episode_rewards.append(episode_return.mean())

            writer.add_scalar("evaluation/average_reward", np.mean(episode_rewards), total_numsteps)



    # Optionally: Evaluation loop (can also be done in parallel or on a single environment)

    # Save the model checkpoint.
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    agent.save_checkpoint("", ckpt_path=f"checkpoints/sac_checkpoint_{current_time}_")
    vector_env.close()

if __name__ == '__main__':
    main()
