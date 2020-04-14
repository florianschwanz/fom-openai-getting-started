import gym
import numpy as np

# 1. Load Environment and Q-table structure
env = gym.make('FrozenLake8x8-v0')
Q = np.zeros([env.observation_space.n, env.action_space.n])
# env.obeservation.n, env.action_space.n gives number of states and action in env loaded
# 2. Parameters of Q-leanring
learning_rate = .628
discount_factor = .9
epis = 5000
rev_list = []  # rewards per episode calculate
# 3. Q-learning Algorithm
for i in range(epis):
    # Reset environment
    s = env.reset()
    rAll = 0
    done = False
    j = 0
    # The Q-Table learning algorithm
    while j < 99:
        env.render()
        j += 1
        # Choose action from Q table
        a = np.argmax(Q[s, :] + np.random.randn(1, env.action_space.n) * (1. / (i + 1)))
        # Get new state & reward from environment
        observation, reward, done, _ = env.step(a)
        # Update Q-Table with new knowledge
        Q[s, a] = Q[s, a] + learning_rate * (reward + discount_factor * np.max(Q[observation, :]) - Q[s, a])
        rAll += reward
        s = observation
        if done == True:
            break
    rev_list.append(rAll)
    env.render()
print("Reward Sum on all episodes " + str(sum(rev_list) / epis))
print("Final Values Q-Table")
print(Q)
