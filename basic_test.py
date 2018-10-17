import gym
import gym_cartpole_visual

env = gym.make("cartpole-visual-v1")

for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break