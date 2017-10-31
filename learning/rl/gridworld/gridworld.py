# encoding=utf8
# Copy from: https://github.com/lipixun/DeepRL-Agents
# pylint: disable=missing-docstring,singleton-comparison
#
#   Ref: https://github.com/awjuliani/DeepRL-Agents/blob/master/gridworld.py
#

import itertools

import numpy as np
import scipy.misc
import matplotlib.pyplot as plt

ImageSize = 84
ImageDepth = 3

class GameOb(object):
    """The game observation
    """
    def __init__(self, coordinates, size, intensity, channel, reward, name):
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
    def __init__(self,partial,size):
        self.sizeX = size
        self.sizeY = size
        self.actions = 4
        self.objects = []
        self.partial = partial
        state = self.reset()
        plt.imshow(state, interpolation="nearest")

    def reset(self):
        self.objects = []
        hero = GameOb(self.newPosition(),1,1,2,None,'hero')
        self.objects.append(hero)
        bug = GameOb(self.newPosition(),1,1,1,1,'goal')
        self.objects.append(bug)
        hole = GameOb(self.newPosition(),1,1,0,-1,'fire')
        self.objects.append(hole)
        bug2 = GameOb(self.newPosition(),1,1,1,1,'goal')
        self.objects.append(bug2)
        hole2 = GameOb(self.newPosition(),1,1,0,-1,'fire')
        self.objects.append(hole2)
        bug3 = GameOb(self.newPosition(),1,1,1,1,'goal')
        self.objects.append(bug3)
        bug4 = GameOb(self.newPosition(),1,1,1,1,'goal')
        self.objects.append(bug4)
        state = self.renderEnv()
        self.state = state
        return state

    def moveChar(self, direction):
        # 0 - up, 1 - down, 2 - left, 3 - right
        hero = self.objects[0]
        heroX = hero.x
        heroY = hero.y
        penalize = 0.
        if direction == 0 and hero.y >= 1:
            hero.y -= 1
        if direction == 1 and hero.y <= self.sizeY-2:
            hero.y += 1
        if direction == 2 and hero.x >= 1:
            hero.x -= 1
        if direction == 3 and hero.x <= self.sizeX-2:
            hero.x += 1
        if hero.x == heroX and hero.y == heroY:
            penalize = 0.0
        self.objects[0] = hero
        return penalize

    def newPosition(self):
        iterables = [ range(self.sizeX), range(self.sizeY)]
        points = []
        for t in itertools.product(*iterables):
            points.append(t)
        currentPositions = []
        for objectA in self.objects:
            if (objectA.x,objectA.y) not in currentPositions:
                currentPositions.append((objectA.x,objectA.y))
        for pos in currentPositions:
            points.remove(pos)
        location = np.random.choice(range(len(points)),replace=False)   # pylint: disable=no-member
        return points[location]

    def checkGoal(self):
        others = []
        for obj in self.objects:
            if obj.name == 'hero':
                hero = obj
            else:
                others.append(obj)
        ended = False
        for other in others:
            if hero.x == other.x and hero.y == other.y:
                self.objects.remove(other)
                if other.reward == 1:
                    self.objects.append(GameOb(self.newPosition(),1,1,1,1,'goal'))
                else:
                    self.objects.append(GameOb(self.newPosition(),1,1,0,-1,'fire'))
                return other.reward, False
        if ended == False:
            return 0.0, False

    def renderEnv(self):
        state = np.ones([self.sizeY+2, self.sizeX+2, 3])
        state[1:-1, 1:-1, :] = 0
        hero = None
        for item in self.objects:
            state[item.y+1:item.y+item.size+1, item.x+1:item.x+item.size+1, item.channel] = item.intensity
            if item.name == "hero":
                hero = item
        if self.partial:
            state = state[hero.y:hero.y+3, hero.x:hero.x+3, :]
        a = scipy.misc.imresize(state[:, :, 0], [84, 84, 1], interp="nearest")
        b = scipy.misc.imresize(state[:, :, 1], [84, 84, 1], interp="nearest")
        c = scipy.misc.imresize(state[:, :, 2], [84, 84, 1], interp="nearest")
        state = np.stack([a, b, c], axis=2)
        return state

    def step(self, action):
        penalty = self.moveChar(action)
        reward, done = self.checkGoal()
        state = self.renderEnv()
        return state, (reward + penalty), done
