import time
import random
import numpy as np
import metaworld

def main():
    seed = 123456
    random.seed(seed)
    np.random.seed(seed)

    # Use MT10 to get a set of tasks (MT10 contains 10 tasks including push).
    mw = metaworld.MT10(seed=seed)
    
    # Choose the push task. The key here is the task name; in MT10 the push task is labeled "push".
    push_name = "push-v2"

    print(mw.train_classes)
    if push_name not in mw.train_classes:
        print("Push task not found in the available tasks.")
        return

    env_cls = mw.train_classes[push_name]
    # Filter for push tasks among the training tasks.
    push_tasks = [task for task in mw.train_tasks if task.env_name == push_name]
    if not push_tasks:
        print("No push tasks found in the training set.")
        return

    # Select a random push task
    task = random.choice(push_tasks)
    
    # Instantiate the environment and set the task.
    env = env_cls(render_mode="human")
    env.set_task(task)
    env.seed(seed)
    
    # Optional: Inspect or modify goal values.
    # For instance, the environment stores the goal in self._target_pos
    print("Original target position:", env._target_pos)
    # You can modify the goal here if desired. For example:
    # env._target_pos = np.array([0.0, 0.85, 0.25])
    # And then update the corresponding simulation site:
    env.model.site("goal").pos = env._target_pos

    # Visualize the environment over several episodes.
    num_episodes = 10
    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0.0
        print(f"\nStarting episode {ep+1}")
        i = 0
        while not done and i < 100:
            env.render()  # Visualize the current state.
            action = env.action_space.sample()  # Use a random action.
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            #time.sleep(0.05)  # Slow down rendering for visualization.
            i += 1
            print(i)
        print(f"Episode {ep+1} finished with total reward: {total_reward}")
    
    env.close()

if __name__ == "__main__":
    main()
