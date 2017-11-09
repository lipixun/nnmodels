# encoding=utf8

""" The environment
    Author: lipixun
    File Name: env.py
    Description:

"""

import gym
import numpy as np

class EnvGroup(object):
    """The env group
    """
    def __init__(self, name, num):
        """Create a new EnvGroup
        """
        self.num = num
        self.envs = [gym.make(name) for _ in range(num)]
        self.envTerminates = [False] * num

    @property
    def actionNums(self):
        """Get the action nums
        """
        return self.envs[0].action_space.n

    def render(self):
        """Render
        """
        self.envs[0].render()

    def reset(self):
        """Reset all envs
        """
        self.envTerminates = [False] * self.num
        return np.stack([env.reset() for env in self.envs])

    def step(self, actions):
        """Run envs
        """
        index = 0
        indexes, states, rewards, terminates, infos = [], [], [], [], []
        for i, env in enumerate(self.envs):
            if self.envTerminates[i]:
                continue
            s, r, t, info = env.step(actions[index])
            index += 1
            indexes.append(i)
            states.append(s)
            rewards.append(r)
            terminates.append(t)
            infos.append(info)
            if t:
                self.envTerminates[i] = True
        return indexes, np.stack(states), np.stack(rewards), np.stack(terminates), infos
