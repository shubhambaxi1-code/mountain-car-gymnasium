import gymnasium as gym

env = gym.make("MountainCar-v0", render_mode="human")

obv, info = env.reset()

epi_over = False
total_reward = 0

while not epi_over:
    action = env.action_space.sample()

    obv, reward, terminated, truncated, info = env.step(action)

    total_reward += reward
    episode_over = terminated or truncated

print(f"Episode finished! Total reward: {total_reward}")
env.close()
