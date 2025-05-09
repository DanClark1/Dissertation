
def play_random_env_video(env):

    # Instantiate and load the trained model
    loaded_model = mt_core.MoEActorCritic(
        observation_space=env.observation_space,
        action_space=env.action_space,
        num_tasks=10,
        num_experts=3
    )
    loaded_model.load_state_dict(torch.load('models/model.pt'))

    # Run one episode in a random environment and record frames
    frames = []
    obs, info = env.reset()
    done = False
    while not done:
        with torch.no_grad():
            # In this example, last 10 values are assumed as task encoding
            obs_trunc, task = format_obs(obs)
            action = loaded_model.act(
                obs_trunc,
                task,
                deterministic=True
            )
        obs, reward, terminated, truncated, info = env.step(action)
        if hasattr(env, 'render'):
            frame = env.render(mode='rgb_array')
            frames.append(frame)
        done = terminated or truncated

    # Save recorded frames as a video
    imageio.mimsave('trained_model_demo.mp4', frames, fps=30)
    print("Video saved to trained_model_demo.mp4")


play_random_env_video(env)





# ------------------------------------------------------------





from copy import deepcopy
import itertools
import numpy as np
import torch
from torch.optim import Adam
import gymnasium as gym
import time
import atsac.no_expert_moe_core as core
from tqdm import tqdm


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
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


def at_sac(env_fn, num_tasks, num_experts, actor_critic=core.MoEActorCritic, ac_kwargs=dict(), seed=0, 
        steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99, 
        polyak=0.995, lr=1e-3, alpha=0.2, batch_size=100, start_steps=10000, 
        update_after=1000, update_every=50, num_test_episodes=10, max_ep_len=1000, 
        logger_kwargs=dict(), save_freq=1):
    """
    Soft Actor-Critic (SAC), Gymnasium Version

    Note:
        * Key differences from old Gym:
            - env.reset() -> (obs, info)
            - env.step(action) -> (obs, reward, terminated, truncated, info)
            - Combine (terminated or truncated) into a single `done` to store in replay
    """

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create both train and test envs
    env = env_fn()
    test_env = env_fn()

    # Get observation and action dimensions
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape[0]

    # Action limit for clamping: assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]

    # Create actor-critic module and target networks
    ac = actor_critic(env.observation_space, env.action_space, num_tasks, num_experts,  **ac_kwargs)
    ac_targ = deepcopy(ac)

    # Freeze target networks (only update via polyak averaging)
    for p in ac_targ.parameters():
        p.requires_grad = False
        
    # Combine Q-net parameters
    q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())

    # Experience buffer
    replay_obs_dim = list(obs_dim)
    replay_obs_dim[-1] = replay_obs_dim[-1] + num_tasks
    replay_buffer = ReplayBuffer(obs_dim=tuple(replay_obs_dim), act_dim=act_dim, size=replay_size)

    # Count variables
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.q1, ac.q2])

    def compute_loss_q(data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
        o_trunc, task = format_obs(o, num_tasks)
        o2_trunc, _ = format_obs(o2, num_tasks) # task shouldn't change

        q1 = ac.q1(o_trunc, task, a)
        q2 = ac.q2(o_trunc, task, a)

        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = ac.pi(o2_trunc, task)
            # Target Q-values
            q1_pi_targ = ac_targ.q1(o2_trunc, task, a2)
            q2_pi_targ = ac_targ.q2(o2_trunc, task, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + gamma * (1 - d) * (q_pi_targ - alpha * logp_a2)

        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        q_info = dict(Q1Vals=q1.detach().numpy(),
                      Q2Vals=q2.detach().numpy())
        return loss_q, q_info

    def compute_loss_pi(data):
        o = data['obs']
        o_trunc, task = format_obs(o, num_tasks)
        pi, logp_pi = ac.pi(o_trunc, task)
        q1_pi = ac.q1(o_trunc, task, pi)
        q2_pi = ac.q2(o_trunc, task, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (alpha * logp_pi - q_pi).mean()

        pi_info = dict(LogPi=logp_pi.detach().numpy())
        return loss_pi, pi_info

    # Set up optimizers
    pi_optimizer = Adam(ac.pi.parameters(), lr=lr)
    q_optimizer = Adam(q_params, lr=lr)

    def update(data):
        # Update Q-networks
        q_optimizer.zero_grad()
        loss_q, q_info = compute_loss_q(data)
        loss_q.backward()
        q_optimizer.step()

        # Freeze Q-networks to avoid extra gradients during pi update
        for p in q_params:
            p.requires_grad = False

        # Update policy (pi)
        pi_optimizer.zero_grad()
        loss_pi, pi_info = compute_loss_pi(data)
        loss_pi.backward()
        pi_optimizer.step()

        # Unfreeze Q-network parameters
        for p in q_params:
            p.requires_grad = True

        # Finally, update target networks by polyak averaging
        with torch.no_grad():
            for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)

    def get_action(o, deterministic=False):
        o_trunc, task = format_obs(o, num_tasks)
        return ac.act(torch.as_tensor(o_trunc, dtype=torch.float32), task, deterministic)
    
    def format_obs(o, num_tasks):
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

    def test_agent():
        for _ in range(num_test_episodes):
            # Gymnasium reset -> (obs, info)
            obs, info = test_env.reset()
            done = False
            ep_ret, ep_len = 0, 0

            while not (done or (ep_len == max_ep_len)):
                act = get_action(o, deterministic=True)
                act = act.squeeze()
                obs2, r, terminated, truncated, _info = test_env.step(act)
                done = terminated or truncated
                ep_ret += r
                ep_len += 1
                obs = obs2
        # needs to happen for the multi-task env
        if done:
            test_env.reset()

    # Main SAC loop
    total_steps = steps_per_epoch * epochs
    start_time = time.time()

    # Gymnasium reset: we ignore the "info" part here
    o, info = env.reset()
    ep_ret, ep_len = 0, 0
    torch.save(ac.state_dict(), f'models/model.pt')


    for t in tqdm(range(total_steps)):
        
        # Uniform random actions until start_steps
        if t > start_steps:
            
            a = get_action(o)
            a = a.squeeze()
        else:
            a = env.action_space.sample()

        # Take a step in the environment (Gymnasium)
        o2, r, terminated, truncated, info = env.step(a)

        done = terminated or truncated

        ep_ret += r
        ep_len += 1

        # If the episode ended due to max_ep_len, we often set done=False
        # so we don't confuse the agent about "terminal states".
        # That is optional. For example:
        # if ep_len == max_ep_len:
        #     done = True  # or keep it as is, depending on preference

        # Store experience
        replay_buffer.store(o, a, r, o2, float(done))

        # Update most recent observation
        o = o2

        # End of trajectory handling
        if done or (ep_len == max_ep_len):
            o, info = env.reset()
            ep_ret, ep_len = 0, 0

        # Update handling
        if t >= update_after and t % update_every == 0:
            for _ in range(update_every):
                batch = replay_buffer.sample_batch(batch_size)
                update(data=batch)

        # End of epoch handling
        if (t+1) % steps_per_epoch == 0:
            epoch = (t+1) // steps_per_epoch
            # Test agent
            test_agent()

            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs):
                torch.save(ac.state_dict(), f'models/model.pt')

    


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
    at_sac(lambda: gym.make(args.env), 
        actor_critic=core.MoEActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), 
        gamma=args.gamma, 
        seed=args.seed, 
        epochs=args.epochs)



# ------------------------------------------------------------
class EpochLogger(Logger):
    """
    A variant of Logger tailored for tracking average values over epochs.

    Typical use case: there is some quantity which is calculated many times
    throughout an epoch, and at the end of the epoch, you would like to 
    report the average / std / min / max value of that quantity.

    With an EpochLogger, each time the quantity is calculated, you would
    use 

    .. code-block:: python

        epoch_logger.store(NameOfQuantity=quantity_value)

    to load it into the EpochLogger's state. Then at the end of the epoch, you 
    would use 

    .. code-block:: python

        epoch_logger.log_tabular(NameOfQuantity, **options)

    to record the desired values.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epoch_dict = dict()

    def store(self, **kwargs):
        """
        Save something into the epoch_logger's current state.

        Provide an arbitrary number of keyword arguments with numerical 
        values.
        """
        for k,v in kwargs.items():
            if not(k in self.epoch_dict.keys()):
                self.epoch_dict[k] = []
            self.epoch_dict[k].append(v)

    def log_tabular(self, key, val=None, with_min_and_max=False, average_only=False):
        """
        Log a value or possibly the mean/std/min/max values of a diagnostic.

        Args:
            key (string): The name of the diagnostic. If you are logging a
                diagnostic whose state has previously been saved with 
                ``store``, the key here has to match the key you used there.

            val: A value for the diagnostic. If you have previously saved
                values for this key via ``store``, do *not* provide a ``val``
                here.

            with_min_and_max (bool): If true, log min and max values of the 
                diagnostic over the epoch.

            average_only (bool): If true, do not log the standard deviation
                of the diagnostic over the epoch.
        """
        if val is not None:
            super().log_tabular(key,val)
        else:
            v = self.epoch_dict[key]
            vals = np.concatenate(v) if isinstance(v[0], np.ndarray) and len(v[0].shape)>0 else v
            stats = mpi_statistics_scalar(vals, with_min_and_max=with_min_and_max)
            super().log_tabular(key if average_only else 'Average' + key, stats[0])
            if not(average_only):
                super().log_tabular('Std'+key, stats[1])
            if with_min_and_max:
                super().log_tabular('Max'+key, stats[3])
                super().log_tabular('Min'+key, stats[2])
        self.epoch_dict[key] = []

    def get_stats(self, key):
        """
        Lets an algorithm ask the logger for mean/std/min/max of a diagnostic.
        """
        v = self.epoch_dict[key]
        vals = np.concatenate(v) if isinstance(v[0], np.ndarray) and len(v[0].shape)>0 else v
        return mpi_statistics_scalar(vals)