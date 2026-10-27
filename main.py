# Import Libraries
import gymnasium as gym
import time
import numpy as np

#Intialize the env
env = gym.make("MountainCar-v0", render_mode="ansi")

DISCRETE_BINS = [50,50] #Define the number of bins as per preferences

discrete_win_size = (
    env.observation_space.high - env.observation_space.low
) / DISCRETE_BINS # Calculate width of each bin

def get_discrete_state(state):
    """Converts continuous states like [-0.5234, 0.012] into discrete index tuples like (7, 11)"""
    discrete_state = (state - env.observation_space.low) / discrete_win_size
    #Clip the index values between 0 and 50-1 in this situation so it never crashes at boundaries
    discrete_state = np.clip(
        discrete_state.astype(int), 0, np.array(DISCRETE_BINS) - 1
    )
    return tuple(discrete_state)

# Initialize Q-table with shape (20, 20, 3) -> 20 positions x 20 velocities x 3 actions
q_table = np.zeros(DISCRETE_BINS + [env.action_space.n])

# 3. Hyperparameters
total_episodes = 50000
learning_rate = 0.1  # alpha
discount_rate = 0.95  # gamma

# Exploration parameters
epsilon = 1.0
max_epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.002

try:
    for episode in range(total_episodes):
        total_reward = 0

        state, info = env.reset()
        state_discrete = get_discrete_state(state)

        done = False #Is episode over?

        print(f"Starting episode {episode}")

        while not done:
            if np.random.uniform(0, 1) > epsilon: # Pick a random value to decide whether to explore or not
                action = np.argmax(q_table[state_discrete])
            else:
                action = env.action_space.sample() # Do a random action

            next_state, reward, terminated, truncated, info = env.step(action)
            next_state_discrete = get_discrete_state(next_state)
            total_reward += reward
            done = terminated or truncated

            # Bellman equation update rule
            old_q = q_table[state_discrete][action]
            max_future_q = np.max(q_table[next_state_discrete])

            q_table[state_discrete][action] = old_q + learning_rate * (
                reward + discount_rate * max_future_q - old_q
            )

            state_discrete = next_state_discrete

        # Decay exploration rate over time
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(
            -decay_rate * episode
        )

        print(f"Episode finished! Total reward: {total_reward}")
except KeyboardInterrupt:
    print("\nSkipping training till how much done")

print("\n--- WATCHING TRAINED AGENT PLAY ---")
env.close()  # Close text environment

# Open graphic environment
env = gym.make("MountainCar-v0", render_mode="human")
current_state, info = env.reset()
state_discrete = get_discrete_state(current_state)
done = False

while not done:
    # Always exploit the learned values
    action = np.argmax(q_table[state_discrete])
    next_state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    state_discrete = get_discrete_state(next_state)
    time.sleep(0.01)  # Keeps frame rates looking smooth

env.close()
print("Simulation complete!")
