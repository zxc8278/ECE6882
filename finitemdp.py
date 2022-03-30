"""
finiteMDP.py

Project: ECE6882
Author: Kyle Crandall <crandallk@gwu.edu>

Class for a finite Markov Descision Process
"""

import numpy as np
import copy

class FiniteMDP(object):
    def __init__(self, states, actions, dynamics, reward, init_state=None):
        self.states = states
        self.actions = actions
        self.dynamics_func = dynamics
        self.reward_func = reward
        
        if init_state is None:
            self._state = np.random.choice(self.states)
        else:
            self._state = init_state

        assert self._state in self.states

    def __call__(self, action):
        assert action in self.actions
        new_state = np.random.choice(self.states, p=[self.dynamics(self._state, self.states[k], action) for k in self.states])
        reward = self.reward(self._state, new_state, action)
        self._state = new_state
        return new_state, reward

    def dynamics(self, state, state_prime, action):
        assert state in self.states
        assert state_prime in self.states
        assert action in self.actions
        return self.dynamics_func(state, state_prime, action)

    def reward(self, state, state_prime, action):
        assert state in self.states
        assert state_prime in self.states
        assert action in self.actions
        return self.reward_func(state, state_prime, action)


class DiceGame(FiniteMDP):
    def __init__(self):
        states = ["playing", "end"]
        actions = ["continue", "quit"]
        super().__init__(states, actions, self.dice_dynamics, self.reward, "playing")
    
    def dice_dynamics(self, state, state_prime, action):
        # return probability that taking an action from state will result in state_prime
        if state == "playing" and action == "continue":
            if state_prime == "end":
                return 0.33 # rolling 1 or 2 ends dice game
            else:
                return 0.67
        elif state == "playing" and action == "quit":
            if state_prime == "end":
                return 1.0
            else:
                return 0.0
        elif state == "end":
            if state_prime == "end":
                return 1.0
            else:
                return 0.0
            
    def roll_dice(self):
        sides = [1, 2, 3, 4, 5, 6]
        return np.random.choice(sides)

    def sar_dynamics(self, state, action):
        if state == "playing" and action == "quit":
            state_prime = "end"
            reward = 5.0
        elif state == "playing" and action == "continue":
            reward = 3.0 # $3 reward for staying in the game
            n = self.roll_dice()
            if n > 2:
                state_prime = "playing"
            else:
                state_prime = "end"
        else:
            # not playing, can't continue
            state_prime = "end"
            reward = 0.0
        return state_prime, reward
    

class GridWorld(FiniteMDP):
    def __init__(self, rows=None, cols=None, goals=None, obs=None, goal_percent=0.01, obs_percent=0.25, map=None, init_pos=None):
        if map is None:
            map = np.zeros((rows, cols))
            if goals is None:
                n_goals = np.maximum(1, int(rows*cols*goal_percent))
                goals = [(i,j) for i, j in zip(np.random.randint(0, rows, size=n_goals), np.random.randint(0, cols, size=n_goals))]
            if obs is None:
                n_obs = int(rows*cols*obs_percent)
                obs = [(i,j) for i,j in zip(np.random.randint(0, rows, size=n_obs), np.random.randint(0, cols, size=n_obs))]
            for i, j in goals:
                map[i, j] = -1
            for i, j in obs:
                map[i, j] = 1
        if init_pos is not None:
            init_pos = f"{init_pos[0]},{init_pos[1]}"
        
        self.map = map
        
        states = ["goal"]
        for i in range(self.map.shape[0]):
            for j in range(self.map.shape[1]):
                name = f"{i},{j}"
                if self.map[i, j] == 0:
                    states.append(name)
        actions = ["up", "down", "left", "right"]
        super().__init__(states, actions, self.gw_dynamics, self.gw_reward, init_pos)
    
    def gw_dynamics(self, state, new_state, action):
        if state == "goal":
            if new_state == "goal":
                return 1.0
            else:
                return 0.0
        else:
            print(state, type(state))
            pos = [int(k) for k in state.split(",")]
            if action == "up" and pos[0] != 0:
                pos[0] -= 1
            elif action == "down" and pos[0] != self.map.shape[0]-1:
                pos[0] += 1
            elif action == "right" and pos[1] != self.map.shape[1]-1:
                pos[1] += 1
            elif action == "left" and pos[1] != 0:
                pos[1] -= 1
            
            new_name = f"{pos[0]},{pos[1]}"
            if  self.map[pos[0], pos[1]] == -1:
                new_name = "goal"
            if self.map[pos[0], pos[1]] == 1:
                new_name = state

            if new_name == new_state:
                return 1.0
            else:
                return 0.0
            
    def sar_dynamics(self, state, action):
        reward = -1.0
        if state == "goal":
            reward = 0.0
            new_name = "goal"

        else:
            pos = [int(k) for k in state.split(",")]
            if action == "up" and pos[0] != 0:
                pos[0] -= 1
            elif action == "down" and pos[0] != self.map.shape[0]-1:
                pos[0] += 1
            elif action == "right" and pos[1] != self.map.shape[1]-1:
                pos[1] += 1
            elif action == "left" and pos[1] != 0:
                pos[1] -= 1
            
            new_name = f"{pos[0]},{pos[1]}"
            if  self.map[pos[0], pos[1]] == -1:
                new_name = "goal"
                reward = 0.0
            if self.map[pos[0], pos[1]] == 1:
                new_name = state
                reward = -1.0
            
        return new_name, reward

    def gw_reward(self, state, new_state, action):
        if new_state == "goal":
            return 0.0
        else:
            return -1.0
    
    def save_map(self, filename):
        with open(filename, "w") as f:
            for i in range(self.map.shape[0]):
                for j in range(self.map.shape[1]):
                    f.write(f"{int(self.map[i, j])}")
                    if j != self.map.shape[1]-1:
                        f.write(" ")
                f.write("\n")
    
    @staticmethod
    def from_file(filename):
        map = []
        with open(filename, "r") as f:
            for line in f.readlines():
                map.append([int(s) for s in line.split(" ")])
                print(map)
        print(np.array(map))
        return GridWorld(map=np.array(map))

class SlipperyGridWorld(GridWorld):
    def __init__(self, rows=None, cols=None, goals=None, obs=None, goal_percent=0.01, obs_percent=0.25, map=None, init_pos=None, p=0.1):
        super().__init__(rows=rows, cols=cols, goals=goals, obs=obs, goal_percent=goal_percent, obs_percent=obs_percent, map=map, init_pos=init_pos)
        self.p = p

    def gw_dynamics(self, state, new_state, action):
        if state == "goal":
            if new_state == "goal":
                return 1.0
            else:
                return 0.0
        pos = np.array([int(k) for k in state.split(",")])
        
        delta = np.zeros(2, "int")
        if action == "up":
            delta[0] = -1
        elif action == "down":
            delta[0] = 1
        elif action == "left":
            delta[1] = -1
        elif action == "right":
            delta[1] = 1

        ps = [0.0]
        p_goal = 0.0
        new_pos = copy.copy(pos)
        p = 1.0
        while 1:
            new_pos += delta
            if new_pos[0] >= self.map.shape[0] or new_pos[0] < 0 or new_pos[1] >= self.map.shape[1] or new_pos[1] < 0 or self.map[new_pos[0], new_pos[1]] == 1:
                ps[-1] += p
                break
            ps.append(p*(1 - self.p))
            p *= self.p
            if self.map[new_pos[0], new_pos[1]] == -1:
                p_goal += ps[-1]
        
        if new_state == "goal":
            return p_goal
        
        rel_pos = np.array([int(k) for k in new_state.split(",")]) - pos
        if np.linalg.norm(rel_pos) !=0 and np.any(rel_pos/np.linalg.norm(rel_pos) != delta):
            return 0.0
        rel_pos = np.abs(np.sum(rel_pos))
        if rel_pos >= len(ps):
            return 0.0
        return ps[rel_pos]
    
    @staticmethod
    def from_file(filename, p=0.1):
        gw = super(SlipperyGridWorld, SlipperyGridWorld).from_file(filename)
        return SlipperyGridWorld(map=gw.map, p=p)
