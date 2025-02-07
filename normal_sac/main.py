import argparse
import datetime
import gym
import numpy as np
import itertools
import torch
from sac import SAC
from mt_sac import MT_SAC
from torch.utils.tensorboard import SummaryWriter
from replay_memory import ReplayMemory
import random
import metaworld
import imageio
from PIL import Image, ImageDraw
import cProfile
import pstats

def format_obs(o, num_tasks=10):
        '''
        Extracts the one-hot encoding from the observation vector,
        identifies the task and returns the observation vector
        without the one-hot encoding

        :param o: observation vector
        :param num_tasks: number of tasks
        :return: observation vector without the one-hot encoding
        '''
        task = np.argmax(o[...,-num_tasks:], axis=-1)

        return o[..., :-num_tasks], task

# Define a custom environment wrapper to include the one-hot task encoding
class MultiTaskEnv(gym.Env):
    def __init__(self, env_list, env_names=None):
        super().__init__()
        self.num_envs = len(env_list)
        self.envs = env_list
        self.env_names = env_names
        print(self.num_envs)
        # observation space is max and min of all envs (not sure if this is the same for each anyway?)
        # and then the one hot vector on the end
        self.set_max_min_obs()
        
        self.observation_space = gym.spaces.Box(
            low=np.concatenate([self.obs_min, np.zeros(self.num_envs)]),
            high=np.concatenate([self.obs_max, np.ones(self.num_envs)]),
            dtype=np.float64
        )


        self.action_space = self.envs[0].action_space # assume all envs have the same action space

        self.current_episode_rewards = 0
        self.active_index = 0
        self.rewards = [[] for i in self.envs]
        # initialise a random environment
        self.sample_active_env()

        self.has_reward_been_recorded = False

    def get_active_env_name(self):
        '''
        Returns the name of the active environment
        '''
        return self.env_names[self.active_index]

    def reset(self, randomise=True, **kwargs):
        '''
        Resets the current the environment, 
        then resamples another random environemt
        and returns the observation state of that'''


        if self.has_reward_been_recorded:
            self.rewards[self.active_index].append(self.current_episode_rewards)
            self.current_episode_rewards = 0 
        

        self.active_env.reset(**kwargs)
        # resample
        self.sample_active_env(randomise)
        obs = self.active_env.reset(**kwargs)

        self.has_reward_been_recorded = False

        return self._get_obs(obs)
    
    def get_env_names(self):
        '''
        Returns a list of environment 
        names if they have been set'''
        return self.env_names
    
    def sample_active_env(self, randomise=True):
        '''
        Randomly set an active environment
        '''
        if not randomise:
            self.active_index += 1
            self.active_index %= len(self.envs)
        else:
            self.active_index = random.randint(0, len(self.envs) - 1)
        self.active_env = self.envs[self.active_index]

    def render(self):
        self.active_env.render_mode = 'rgb_array'
        frame = self.active_env.render()
        self.active_env.render_mode = None
        return frame


    def step(self, action):
        obs, reward, terminated, truncated, info, *kwargs = self.active_env.step(action)
        done = terminated or truncated

        self.has_reward_been_recorded = True
        self.current_episode_rewards += reward

        return self._get_obs(obs), reward, terminated, truncated, *kwargs

    def _get_obs(self, obs):
        # Create a one-hot encoding of the task ID
        task_one_hot = np.zeros(self.num_envs)
        task_one_hot[self.active_index] = 1.0

        # sometimes obs is a tuple??
        while type(obs) == tuple:
            obs = obs[0]

        return np.concatenate([obs, task_one_hot])
        

    def set_max_min_obs(self):
        '''
        Finds the maximum and minimum values for the observation spaces
        '''
        if self.num_envs != 0:
            self.obs_max = self.envs[0].observation_space.high
            self.obs_min = self.envs[0].observation_space.low

            if self.num_envs > 1:
                for i in range(1, len(self.envs)):
                    self.obs_max = np.maximum(self.obs_max, self.envs[i].observation_space.high)
                    self.obs_min = np.minimum(self.obs_min, self.envs[i].observation_space.low)

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env-name', default="",
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Automaically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')
parser.add_argument('--use_moe', action="store_true", help='use MOE (default: False)')
args = parser.parse_args()


def train():
    # Environment
    # env = NormalizedActions(gym.make(args.env_name))
    mt10 = metaworld.MT10(seed=args.seed) 
    training_envs = []
    names = []
    render = True
    for name, env_cls in mt10.train_classes.items():
        if render:
            env = env_cls()
            render = False
        else:
            env = env_cls()
        task = random.choice([task for task in mt10.train_tasks
                            if task.env_name == name])
        env.set_task(task)
        training_envs.append(env)
        names.append(name)
    env = MultiTaskEnv(training_envs, env_names=names)



    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Agent
    if args.use_moe:
        agent = MT_SAC(env.observation_space.shape[0], env.action_space, args)
    else:
        agent = SAC(env.observation_space.shape[0], env.action_space, args)

    #Tesnorboard
    writer = SummaryWriter('runs/{}_{}-SAC_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), 'moe' if args.use_moe else '',
                                                                args.policy, "autotune" if args.automatic_entropy_tuning else ""))

    print('Logging at:', 'runs/{}_{}-SAC_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), 'moe' if args.use_moe else '',
                                                                args.policy, "autotune" if args.automatic_entropy_tuning else ""))
    # Memory
    memory = ReplayMemory(args.replay_size, args.seed)

    # Training Loop
    total_numsteps = 0
    updates = 0

    for i_episode in itertools.count(1):
        episode_reward = 0
        episode_steps = 0
        done = False
        state = env.reset()
        while not done:
            if args.start_steps > total_numsteps:
                action = env.action_space.sample()  # Sample random action
            else:
                action = agent.select_action(state)  # Sample action from policy

            if len(memory) > args.batch_size:
                # Number of updates per step in environment
                for i in range(args.updates_per_step):
                    # Update parameters of all the networks
                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, updates)

                    writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                    writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                    writer.add_scalar('loss/policy', policy_loss, updates)
                    writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                    writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                    updates += 1

            next_state, reward, terminated, truncated, *_ = env.step(action) # Step
            done = terminated or truncated
            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward

            # Ignore the "done" signal if it comes from hitting the time horizon.
            # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
            mask = float(not done)

            memory.push(state, action, reward, next_state, mask) # Append transition to memory
            
            state = next_state

        if total_numsteps > args.num_steps:
            break

        writer.add_scalar('reward/train', episode_reward, i_episode)
        print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))

        if i_episode % 10 == 0 and args.eval is True:
            avg_reward = 0.
            episodes = 10
            for _  in range(episodes):
                # do this deterministically
                state = env.reset(randomise=False)
                episode_reward = 0
                done = False
                while not done:
                    action = agent.select_action(state, evaluate=True)

                    next_state, reward, terminated, truncated, *_ = env.step(action)
                    done = terminated or truncated
                    episode_reward += reward

                    state = next_state

                avg_reward += episode_reward
            avg_reward /= episodes


            writer.add_scalar('avg_reward/test', avg_reward, i_episode)

            print("----------------------------------------")
            print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
            print("----------------------------------------")


    # save model
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    agent.save_checkpoint("", ckpt_path=f"checkpoints/sac_checkpoint_{current_time}_".format(args.env_name, ""))
    env.close()


def record_agent_video(agent_filename):
    # Set up environment just like in train
    mt10 = metaworld.MT10(seed=args.seed)
    training_envs = []
    names = []
    render_env = True
    for name, env_cls in mt10.train_classes.items():
        env_instance = env_cls()
        task = random.choice([task for task in mt10.train_tasks if task.env_name == name])
        env_instance.set_task(task)
        training_envs.append(env_instance)
        names.append(name)
        if render_env:
            render_env = False
    
    env = MultiTaskEnv(training_envs, env_names=names)
    
    # Load agent
    agent = SAC(env.observation_space.shape[0], env.action_space, args)
    agent.load_checkpoint(ckpt_path=agent_filename)
    
    # Record frames
    frames = []
    for _ in range(5):
        obs = env.reset()
        print(env.get_active_env_name())
        done = False
        while not done:
            action = agent.select_action(obs, evaluate=True)
            obs, _, term, trunc, *_ = env.step(action)
            done = term or trunc
            frame_img = Image.fromarray(env.render())
            draw = ImageDraw.Draw(frame_img)
            draw.text((10, 20), env.get_active_env_name(), fill=(255, 255, 255))
            frames.append(np.array(frame_img))
            frames.append(env.render())
    
    # Save to mp4
    imageio.mimsave("agent_evaluation.mp4", frames, fps=30)
    env.close()

cProfile.run('train()', 'profile_stats')

p = pstats.Stats('profile_stats')
p.sort_stats('cumtime').print_stats(20)

# record_agent_video("checkpoints/sac_final")