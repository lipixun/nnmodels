# encoding=utf8
# Copy from: https://github.com/lipixun/DeepRL-Agents
# pylint: disable=missing-docstring,singleton-comparison
#
#   Ref: https://github.com/awjuliani/DeepRL-Agents/blob/master/gridworld.py
#

import random
import itertools

import numpy as np
import scipy.misc

ImageSize = 84
ImageDepth = 3

class GameOb(object):
    """The game object
    """
    def __init__(self, coordinates, size, intensity, channel, reward, name):
        """Create a new GameOb
        """
        self.x = coordinates[0]
        self.y = coordinates[1]
        self.size = size
        self.intensity = intensity
        self.channel = channel
        self.reward = reward
        self.name = name

class GameEnv(object):
    """The game environment
    """
    def __init__(self, partial, size, outsideReward=0.0, stepCost=0.0):
        """Create a new GameEnv
        """
        self.sizeX = size
        self.sizeY = size
        self.actions = 4
        self.objects = []
        self.partial = partial
        self.outsideReward = outsideReward
        self.stepCost = stepCost

    def reset(self):
        """Reset the environment
        """
        positions = self.randomPositions(7)
        self.objects = [
            # Here
            GameOb(positions[0], 1, 1, 2, None, "hero"),
            # Goal
            GameOb(positions[1], 1, 1, 1, 1, "goal"),
            # Fire
            GameOb(positions[2], 1, 1, 0, -1, "fire"),
            # Goal
            GameOb(positions[3], 1, 1, 1, 1, "goal"),
            # Fire
            GameOb(positions[4], 1, 1, 0, -1, "fire"),
            # Goal
            GameOb(positions[5], 1, 1, 1, 1, "goal"),
            # Goal
            GameOb(positions[6], 1, 1, 1, 1, "goal"),
        ]
        # Render and return the state
        return self.render()

    def move(self, direction):
        """Move
        Returns:
            float: The reward
        """
        # 0 - up, 1 - down, 2 - left, 3 - right
        hero = self.objects[0]
        heroX = hero.x
        heroY = hero.y
        if direction == 0 and hero.y >= 1:
            hero.y -= 1
        if direction == 1 and hero.y <= self.sizeY-2:
            hero.y += 1
        if direction == 2 and hero.x >= 1:
            hero.x -= 1
        if direction == 3 and hero.x <= self.sizeX-2:
            hero.x += 1
        # Check if moved or not
        if hero.x == heroX and hero.y == heroY:
            # Not moved
            return self.outsideReward + self.stepCost
        # Get reward
        for obj in self.objects[1:]:
            if hero.x == obj.x and hero.y == obj.y:
                # Found one
                self.objects.remove(obj)
                if obj.reward == 1:
                    self.objects.append(GameOb(self.newPosition(), 1, 1, 1, 1, "goal"))
                else:
                    self.objects.append(GameOb(self.newPosition(), 1, 1, 0, -1, "fire"))
                return obj.reward + self.stepCost
        # Done
        return self.stepCost

    def randomPositions(self, size):
        """Return a list of random positions
        """
        points = list(itertools.product(range(self.sizeX), range(self.sizeY)))
        return random.sample(points, size)

    def newPosition(self):
        """Generate a new position
        """
        points = list(itertools.product(range(self.sizeX), range(self.sizeY)))
        currentPositions = []
        for obj in self.objects:
            if (obj.x, obj.y) not in currentPositions:
                currentPositions.append((obj.x,obj.y))
        for pos in currentPositions:
            points.remove(pos)
        return random.choice(points)

    def render(self):
        """Render the env
        Returns:
            np.array: The image array
        """
        state = np.ones([self.sizeY+2, self.sizeX+2, 3])
        state[1:-1, 1:-1, :] = 0
        for obj in self.objects:
            state[obj.y+1:obj.y+obj.size+1, obj.x+1:obj.x+obj.size+1, obj.channel] = obj.intensity
        # Check partial
        if self.partial:
            hero = self.objects[0]
            state = state[hero.y:hero.y+3, hero.x:hero.x+3, :]
        a = scipy.misc.imresize(state[:, :, 0], [84, 84, 1], interp="nearest")
        b = scipy.misc.imresize(state[:, :, 1], [84, 84, 1], interp="nearest")
        c = scipy.misc.imresize(state[:, :, 2], [84, 84, 1], interp="nearest")
        state = np.stack([a, b, c], axis=2)
        return state

    def step(self, action):
        """Step
        """
        reward = self.move(action)
        state = self.render()
        # Done
        return state, reward, False, None
