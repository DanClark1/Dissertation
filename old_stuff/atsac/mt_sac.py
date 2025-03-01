from copy import deepcopy
import itertools
import numpy as np
import torch
from torch.optim import Adam
import gymnasium as gym
import time
import atsac.at_moe_core as attention_core
import sac.core as core
from tqdm import tqdm
import imageio
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import datetime
import pickle
import os

# probably a better way to do without global variables but i cant be bothered
log_step = 0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def get_task(o, num_tasks):
        '''
        Gets the task from the observation vector

        :param o: observation vector
        :param num_tasks: number of tasks
        :return: observation vector without the one-hot encoding
        '''
        task = np.argmax(o[...,-num_tasks:], axis=-1)

        return task

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(attention_core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(attention_core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(attention_core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        
        # use cuda if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return {k: torch.as_tensor(v, dtype=torch.float32).to(device) for k,v in batch.items()}

    
class MT_SAC:
    """
    Soft Actor-Critic (SAC), Gymnasium Version

    Note:
        * Key differences from old Gym:
            - env.reset() -> (obs, info)
            - env.step(action) -> (obs, reward, terminated, truncated, info)
            - Combine (terminated or truncated) into a single `done` to store in replay
    """

    def __init__(self, env_fn, num_tasks, num_experts, actor_critic=attention_core.MoEActorCritic, ac_kwargs=dict(), seed=0, 
        timesteps=10000, replay_size=int(1e6), gamma=0.99, 
        polyak=0.995, lr=1e-3, alpha=0.2, batch_size=128, start_steps=10000, 
        update_after=1000, update_every=50, num_test_episodes=100, max_ep_len=1000, 
        logger_kwargs=dict(), save_freq=10000, model_save_path=f'models/', video_save_location='videos/', model_name='my_model',
        env_names=None):
        self.num_tasks = num_tasks
        self.num_experts = num_experts
        self.video_save_location = video_save_location + model_name + '.mp4'
        self.model_save_path = model_save_path + model_name + '.pt'
        self.seed = seed
        self.timesteps = timesteps
        self.replay_size = replay_size
        self.gamma = gamma
        self.polyak = polyak
        self.alpha = alpha
        self.batch_size = batch_size * num_tasks
        self.start_steps = start_steps
        self.update_after = update_after
        self.update_every = update_every
        self.num_test_episodes = num_test_episodes
        self.max_ep_len = max_ep_len
        self.save_freq = save_freq
        self.env_names=env_names
        self.model_name = model_name

        # for loading in from a saved state
        self.loaded_timesteps = None

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        self.writer = SummaryWriter(f'logs/{self.model_name}_{current_time}')

        
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # Create both train and test envs
        self.env = env_fn()
        self.test_env = env_fn()

        # Get observation and action dimensions
        self.obs_dim = self.env.observation_space.shape
        self.act_dim = self.env.action_space.shape[0]

        # Action limit for clamping: assumes all dimensions share the same bound!
        self.act_limit = self.env.action_space.high[0]

        # Create actor-critic module and target networks
        self.ac = actor_critic(self.env.observation_space, self.env.action_space, writer=self.writer, **ac_kwargs)
        self.ac_targ = deepcopy(self.ac)

        # Freeze target networks (only update via polyak averaging)
        for p in self.ac_targ.parameters():
            p.requires_grad = False
            
        # Combine Q-net parameters
        self.q_params = itertools.chain(self.ac.q1.parameters(), self.ac.q2.parameters())

        # Experience buffer
        self.replay_buffer = ReplayBuffer(obs_dim=list(self.obs_dim), act_dim=self.act_dim, size=replay_size)

        # Count variables
        self.var_counts = tuple(attention_core.count_vars(module) for module in [self.ac.pi, self.ac.q1, self.ac.q2])

        # Set up optimizers
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=lr)
        self.q_optimizer = Adam(self.q_params, lr=lr)


    def compute_loss_q(self, data, timestep, log=False):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        # if avaiable, use cuda
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        o = torch.as_tensor(o, dtype=torch.float32).to(device)
        a = torch.as_tensor(a, dtype=torch.float32).to(device)
        r = torch.as_tensor(r, dtype=torch.float32).to(device)
        o2 = torch.as_tensor(o2, dtype=torch.float32).to(device)
        d = torch.as_tensor(d, dtype=torch.float32).to(device)

        q1, reg_term_q1 = self.ac.q1(o, a)
        q2, reg_term_q2 = self.ac.q2(o, a)

        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2, _ = self.ac.pi(o2)
            # Target Q-values
            q1_pi_targ, _ = self.ac_targ.q1(o2, a2)
            q2_pi_targ, _ = self.ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + self.gamma * (1 - d) * (q_pi_targ - self.alpha * logp_a2)

        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()

        # expert balancing regularisation
        if reg_term_q1 is not None:
            reg_term_mean_q1 = reg_term_q1.mean()
            reg_term_mean_q2 = reg_term_q2.mean()
            if log:
                self.writer.add_scalar('ExpertUtil/Q1', reg_term_mean_q1, timestep)
                self.writer.add_scalar('ExpertUtil/Q2', reg_term_mean_q2, timestep)
            loss_q1 += reg_term_mean_q1
            loss_q2 += reg_term_mean_q2

        loss_q = loss_q1 + loss_q2

        q_info = dict(Q1Vals=q1.detach(),
                      Q2Vals=q2.detach())
        
        if log:
            self.writer.add_scalar('Loss/Q1', loss_q1, timestep)
            self.writer.add_scalar('Loss/Q2', loss_q2, timestep)
        return loss_q, q_info

    def compute_loss_pi(self, data, timestep, log=False):
        o = data['obs']
        pi, logp_pi, reg_term = self.ac.pi(o)
        q1_pi, _ = self.ac.q1(o, pi)
        q2_pi, _ = self.ac.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (self.alpha * logp_pi - q_pi).mean()

        if reg_term is not None:
            if log:
                self.writer.add_scalar('ExpertUtil/pi', reg_term.mean(), timestep)
            loss_pi += reg_term.mean()

        pi_info = dict(LogPi=logp_pi.detach())
        if log:
            self.writer.add_scalar('Loss/Pi', loss_pi, timestep)
        return loss_pi, pi_info

    

    def update(self, timestep, data, log=False):
        # Update Q-networks
        self.q_optimizer.zero_grad()
        loss_q, q_info = self.compute_loss_q(data, timestep, log=log)
        loss_q.backward()
        self.q_optimizer.step()

        # Freeze Q-networks to avoid extra gradients during pi update
        for p in self.q_params:
            p.requires_grad = False

        # Update policy (pi)
        self.pi_optimizer.zero_grad()
        loss_pi, pi_info = self.compute_loss_pi(data, timestep, log=log)
        loss_pi.backward()
        self.pi_optimizer.step()

        # Unfreeze Q-network parameters
        for p in self.q_params:
            p.requires_grad = True

        # Finally, update target networks by polyak averaging
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)


    def get_action(self, o, deterministic=False):
        action = self.ac.act(torch.as_tensor(o, dtype=torch.float32).to(device), deterministic)
        return action


    def evaluate(self, episodes=None, render=False, save_plots=True):
        if render:  
            frames = []
    
        episodes = self.num_test_episodes if episodes is None else episodes

        rewards = [[] for _ in range(self.num_tasks)]
        for _ in tqdm(range(episodes)):
            # Gymnasium reset -> (obs, info)
            obs, info = self.test_env.reset()
            done = False
            ep_ret, ep_len = 0, 0
            episode_reward = 0

            while not (done or (ep_len == self.max_ep_len)):
                act = self.get_action(obs, deterministic=True).cpu().numpy()
                act = act.squeeze()
                obs2, r, terminated, truncated, _info = self.test_env.step(act)
                done = terminated or truncated
                ep_ret += r
                ep_len += 1
                obs = obs2
                episode_reward += r
            
                if render and hasattr(self.env, 'render'):
                    frame = self.env.render()
                    frames.append(frame)

                if done:
                    # record reward
                    rewards[get_task(obs, self.num_tasks)].append(episode_reward)
                    episode_reward = 0

        if save_plots:
            # Plot reward graphs for each environment
            for i, rewards in enumerate(rewards):
                plt.figure()
                plt.plot(rewards)
                plt.title(f"Rewards for Environment {i}")
                plt.xlabel("Episodes")
                plt.ylabel("Rewards")
                plt.grid()
                # check if the directory exists and if not create it
                try:
                    plt.savefig(f'images/{self.model_name}/{self.env_names[i]}_mt_sac_baseline.png')
                except:
                    import os
                    os.makedirs(f'images/{self.model_name}')
                    plt.savefig(f'images/{self.model_name}/{self.env_names[i]}_mt_sac_baseline.png')

            plt.close()
            
        if render:
            imageio.mimsave(self.video_save_location, frames, fps=30)
            print(f"Video saved to {self.video_save_location}")

        # needs to happen for the multi-task env
        if done:
            self.test_env.reset()


    def train(self):
        log_step = 0
        # Main SAC loop
        print('timesetps: ', self.timesteps)
        if self.loaded_timesteps is not None:
            self.timesteps -= self.loaded_timesteps
        
        policy_time = 0
        training_time = 0
        environment_time = 0
        # Gymnasium reset: we ignore the "info" part here
        o, info = self.env.reset()
        ep_ret, ep_len = 0, 0

        # check if directory exists and if not create it
        try:
            torch.save(self.ac.state_dict(), f'models/model.pt')
        except:
            import os
            os.makedirs('models')
            torch.save(self.ac.state_dict(), f'models/model.pt')

        for t in tqdm(range(self.timesteps)):
            
            # Uniform random actions until start_steps
            if t > self.start_steps:  
                start_time = time.time()
                a = self.get_action(o).cpu().numpy()
                policy_time += (time.time() - start_time)
                a = a.squeeze()
            else:
                a = self.env.action_space.sample()

            # Take a step in the environment (Gymnasium)
            start_time = time.time()
            o2, r, terminated, truncated, info = self.env.step(a)
            environment_time += (time.time() - start_time)

            done = terminated or truncated

            ep_ret += r
            ep_len += 1

            # Store experience
            self.replay_buffer.store(o, a, r, o2, float(done))

            # Update most recent observation
            o = o2

            # End of trajectory handling
            if done or (ep_len == self.max_ep_len):
                start_time = time.time()
                o, info = self.env.reset()
                environment_time += (time.time() - start_time)
                ep_ret, ep_len = 0, 0

            # Update handling
            if t >= self.update_after and t % self.update_every == 0:
                for _ in range(self.update_every):
                    batch = self.replay_buffer.sample_batch(self.batch_size)
                    
                    start_time = time.time()
                    # log every 2% of the timesteps
                    log = (t % (self.timesteps / 2000) == 0 and _ == 0)
                    self.update(t, data=batch, log=log)
                    if log:
                        self.writer.add_scalar('Reward', ep_ret, t)
                        print(f'Logging at timestep {t} out of {self.timesteps}')
                        log_step += 1
                    
                    training_time += (time.time() - start_time)

            # testing agent and saving the model 
            if (t+1) % self.save_freq == 0:
                print(f'Saving model at timestep {t}')
                self.save_model(temp=True)

        self.save_model()

        total_time = environment_time + training_time + policy_time
        print(f'Environment time: ', environment_time/total_time * 100)
        print(f'Policy (forward) time: ', policy_time/total_time * 100)
        print(f'Backprop time: ', training_time/total_time * 100)

    
    def save_model(self, temp=False):
        '''Save the model to the model_save_path'''
        torch.save(self.ac.state_dict(), self.model_save_path)


    def load_model(self):
        '''Load the model from the model_save_path'''
        self.ac.load_state_dict(torch.load(self.model_save_path))
        
    def create_video(self):
        '''
        Evaluates the model on the environment
        '''
        self.evaluate(render=True, episodes=10, save_plots=False)
        

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v4')
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='sac_gymnasium')
    args = parser.parse_args()

    torch.set_num_threads(torch.get_num_threads())
