from copy import deepcopy
import itertools
import numpy as np
import torch
from torch.optim import Adam
import gymnasium as gym
import time
import atsac.mt_core as attention_core
import sac.core as core
from tqdm import tqdm
import imageio
from torch.utils.tensorboard import SummaryWriter





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
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}


# def at_sac(env_fn, num_tasks, num_experts, actor_critic=core.MoEActorCritic, ac_kwargs=dict(), seed=0, 
#         steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99, 
#         polyak=0.995, lr=1e-3, alpha=0.2, batch_size=100, start_steps=10000, 
#         update_after=1000, update_every=50, num_test_episodes=10, max_ep_len=1000, 
#         logger_kwargs=dict(), save_freq=1):
    
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
        polyak=0.995, lr=1e-3, alpha=0.2, batch_size=100, start_steps=10000, 
        update_after=1000, update_every=50, num_test_episodes=10, max_ep_len=1000, 
        logger_kwargs=dict(), save_freq=1000, model_save_path=f'models/', video_save_location='videos/', model_name='my_model'):

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
        self.batch_size = batch_size
        self.start_steps = start_steps
        self.update_after = update_after
        self.update_every = update_every
        self.num_test_episodes = num_test_episodes
        self.max_ep_len = max_ep_len
        self.save_freq = save_freq

        self.writer = SummaryWriter(f'logs/{model_name}')
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

    def compute_loss_q(self, data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

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

        loss_q1 = ((q1 - backup)**2).mean() + reg_term_q1.mean()
        loss_q2 = ((q2 - backup)**2).mean() + reg_term_q2.mean()
        loss_q = loss_q1 + loss_q2

        q_info = dict(Q1Vals=q1.detach().numpy(),
                      Q2Vals=q2.detach().numpy())
        
        self.writer.add_scalar('Loss/Q1', loss_q1, self.timesteps)
        self.writer.add_scalar('Loss/Q2', loss_q2, self.timesteps)
        return loss_q, q_info

    def compute_loss_pi(self, data):
        o = data['obs']
        pi, logp_pi, reg_term = self.ac.pi(o)
        q1_pi, _ = self.ac.q1(o, pi)
        q2_pi, _ = self.ac.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (self.alpha * logp_pi - q_pi).mean() + reg_term.mean()

        pi_info = dict(LogPi=logp_pi.detach().numpy())
        self.writer.add_scalar('Loss/Pi', loss_pi, self.timesteps)
        return loss_pi, pi_info

    

    def update(self, data):
        # Update Q-networks
        self.q_optimizer.zero_grad()
        loss_q, q_info = self.compute_loss_q(data)
        loss_q.backward()
        self.q_optimizer.step()

        # Freeze Q-networks to avoid extra gradients during pi update
        for p in self.q_params:
            p.requires_grad = False

        # Update policy (pi)
        self.pi_optimizer.zero_grad()
        loss_pi, pi_info = self.compute_loss_pi(data)
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
        action, *_ = self.ac.act(torch.as_tensor(o, dtype=torch.float32), deterministic)
        return action

    def test_agent(self, episodes=None, render=False):
        if render:  
            frames = []
    
        episodes = self.num_test_episodes if episodes is None else episodes
        for _ in range(self.num_test_episodes):
            # Gymnasium reset -> (obs, info)
            obs, info = self.test_env.reset()
            done = False
            ep_ret, ep_len = 0, 0

            while not (done or (ep_len == self.max_ep_len)):
                act = self.get_action(obs, deterministic=True)
                act = act.squeeze()
                obs2, r, terminated, truncated, _info = self.test_env.step(act)
                done = terminated or truncated
                ep_ret += r
                ep_len += 1
                obs = obs2
                
                if render and hasattr(self.env, 'render'):
                    frame = self.env.render()
                    frames.append(frame)

        if render:
            imageio.mimsave(self.video_save_location, frames, fps=30)
            print("Video saved to trained_model_demo.mp4")

        # needs to happen for the multi-task env
        if done:
            self.test_env.reset()


    def train(self):
        # Main SAC loop
        total_steps = self.timesteps
        
        policy_time = 0
        training_time = 0
        environment_time = 0
        # Gymnasium reset: we ignore the "info" part here
        o, info = self.env.reset()
        ep_ret, ep_len = 0, 0
        torch.save(self.ac.state_dict(), f'models/model.pt')

        for t in tqdm(range(total_steps)):
            
            # Uniform random actions until start_steps
            if t > self.start_steps:
                
                start_time = time.time()
                a = self.get_action(o)
                policy_time += (time.time() - start_time)
                a = a.squeeze()
            else:
                a = self.env.action_space.sample()

            # Take a step in the environment (Gymnasium)\
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
                    self.update(data=batch)
                    training_time += (time.time() - start_time)

            # testing agent and saving the model 
            #if (t+1) % self.save_freq == 0:
                # Test agent
                # self.test_agent() do we even want to do this?
        self.save_model()

        total_time = environment_time + training_time + policy_time
        print(f'Environment time: ', environment_time/total_time * 100)
        print(f'Policy (forward) time: ', policy_time/total_time * 100)
        print(f'Backprop time: ', training_time/total_time * 100)

    
    def save_model(self):
        torch.save(self.ac.state_dict(), self.model_save_path)

    def load_model(self):
        self.ac.load_state_dict(torch.load(self.model_save_path, weights_only=True))

    def create_video(self):
        '''
        Evaluates the model on the environment
        '''
        self.test_agent(render=True, episodes=1)
        




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

    # Now we make the environment with Gymnasium
    # at_sac(lambda: gym.make(args.env), 
    #     actor_critic=core.MoEActorCritic,
    #     ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), 
    #     gamma=args.gamma, 
    #     seed=args.seed, 
    #     epochs=args.epochs)
