import metaworld
import random
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
import time
from sac.sac import sac
import sac.core as core

TIMESTEPS = 1000000

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
            

start = time.time()

SEED = 0
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


# env = MultiTaskEnv(training_envs, names)

# model = SAC("MlpPolicy", env, verbose=1)
# model.learn(total_timesteps=TIMESTEPS, log_interval=4)

# train spinningup sac on the environments

env = MultiTaskEnv(training_envs, names)
obs = env.reset()

print('training...')
sac(lambda: env, actor_critic=core.MLPActorCritic, ac_kwargs=dict(hidden_sizes=[256, 256]), 
    gamma=0.99, seed=SEED, epochs=50)

names = env.get_env_names()
print(env.rewards)

# Plot reward graphs for each environment
for i, rewards in enumerate(env.rewards):
    plt.figure()
    plt.plot(rewards)
    plt.title(f"Rewards for Environment {i}")
    plt.xlabel("Episodes")
    plt.ylabel("Rewards")
    plt.grid()
    plt.savefig(f'images/{names[i]}_sac_baseline.png')


obs = env.reset()




print(f'{time.time() - start} seconds since start')

