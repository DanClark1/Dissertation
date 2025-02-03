import metaworld
import random
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
import time
from sac.sac import sac
import sac.core as core
import atsac.no_expert_moe_core as no_expert_moe_core
import atsac.at_moe_core as at_moe_core
from atsac.mt_sac import MT_SAC
import torch
import imageio

TIMESTEPS = 1000000


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
        self.rewards = [[] for i in self.envs]
        # initialise a random environment
        self.sample_active_env()

        self.has_reward_been_recorded = False

    def reset(self, **kwargs):
        '''
        Resets the current the environment, 
        then resamples another random environemt
        and returns the observation state of that'''


        if self.has_reward_been_recorded:
            self.rewards[self.active_index].append(self.current_episode_rewards)
            self.current_episode_rewards = 0 
        

        self.active_env.reset(**kwargs)
        # resample
        self.sample_active_env()
        obs = self.active_env.reset(**kwargs)

        self.has_reward_been_recorded = False

        return self._get_obs(obs), {}
    
    def get_env_names(self):
        '''
        Returns a list of environment 
        names if they have been set'''
        return self.env_names
    
    def sample_active_env(self):
        '''
        Randomly set an active environment
        '''
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

        return self._get_obs(obs), reward, terminated, truncated, info, *kwargs

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
            

def setup_env():
    mt10 = metaworld.MT10(seed=SEED) 
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
    _ = env.reset()
    return env, names

def train_moe_attention_sac(env, num_tasks, num_experts, names, epochs=50):
    model = MT_SAC(lambda: env, num_experts=num_experts, num_tasks=num_tasks, actor_critic=no_expert_moe_core.MoEActorCritic, ac_kwargs=dict(), 
    gamma=0.99, seed=SEED, epochs=50)

    model.train()

    print(env.rewards)

    # Plot reward graphs for each environment
    for i, rewards in enumerate(env.rewards):
        plt.figure()
        plt.plot(rewards)
        plt.title(f"Rewards for Environment {i}")
        plt.xlabel("Episodes")
        plt.ylabel("Rewards")
        plt.grid()
        plt.savefig(f'mt_images/{names[i]}_mt_sac_baseline.png')


start = time.time()

SEED = 0

print('setting up environment...')
env, names = setup_env()

print('training...')

# train_moe_attention_sac(env, num_tasks=10, num_experts=3, names=names, epochs=50)
# sac(lambda: env, actor_critic=core.MLPActorCritic, ac_kwargs=dict(hidden_sizes=[256, 256]), 
#     gamma=0.99, seed=SEED, epochs=50)



attention_model = MT_SAC(lambda: env, num_experts=3, num_tasks=10, actor_critic=at_moe_core.MoEActorCritic, ac_kwargs=dict(num_tasks=10, num_experts=3), 
    gamma=0.99, seed=SEED, timesteps=1000000, start_steps=3000, model_name='attention_moe_sac_2_feb', env_names=names,
    lr=0.0003)

test_model = attention_model = MT_SAC(lambda: env, num_experts=1, num_tasks=10, actor_critic=no_expert_moe_core.EActorCritic, ac_kwargs=dict(num_tasks=10, num_experts=3), 
    gamma=0.99, seed=SEED, timesteps=1000000, start_steps=3000, model_name='no_expert_moe', env_names=names,
    lr=0.0003)


regular_model = MT_SAC(lambda: env, num_experts=3, num_tasks=10, actor_critic=core.MLPActorCritic, 
    gamma=0.99, seed=SEED, timesteps=1000000, model_name='regular_sac', env_names=names)

regular_model.train()
# test_model.load_model()
# test_model.evaluate(episodes=1000)
# test_model.create_video()
# regular_model.load_model()
# regular_model.evaluate(episodes=1000)
# regular_model.create_video()
# attention_model.train()
# attention_model.create_video()







print(f'{time.time() - start} seconds since start')

