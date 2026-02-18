import game2048_env
import numpy as np

env = game2048_env.Env2048()

state = env.reset()
print("Initial:", state)

total_reward = 0

for i in range(50):
    action = np.random.randint(0, 4)

    next_state, reward, done = env.step(action)

    total_reward += reward

    print(f"{i}: action={action}, reward={reward}, done={done}")

    if done:
        break

print("Total reward:", total_reward)
