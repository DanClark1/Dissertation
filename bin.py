
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