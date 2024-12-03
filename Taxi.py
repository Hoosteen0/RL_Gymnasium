import gymnasium as gym

environment = gym.make('CartPole-v1', render_mode='human')
initial_obs, initial_info = environment.reset()

finished = False
while not finished:
    environment.render()

    action = environment.action_space.sample()
    observation, reward, terminated, truncated, info = environment.step(action)

    finished = terminated or truncated # if either then we are finished
    if finished:
        print(f'episode finished')
        observation, info = environment.reset()

environment.close()