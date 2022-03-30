import numpy as np
from bokeh.plotting import figure, save, show
import copy
from finitemdp import GridWorld

N_EPISODES = 1000
MAX_LOOPS = 100
GAMMA = 0.9
EPSILON = 0.01

class Q_routing(object):
     def __init__(self, num_nodes, num_actions, distance, nlinks):
        self.config = {
            "init_mean" : 0.0,      # Initialize Q values with this mean
            "init_std" : 0.0,       # Initialize Q values with this standard deviation
            "learning_rate" : 0.7,
            "eps": 0.1,            # Epsilon in epsilon greedy policies
            "discount": 1,
            "n_iter": 10000000}        # Number of iterations
        self.q = np.zeros((num_nodes,num_nodes,num_actions))

        for src in range(num_nodes):
            for dest in range(num_nodes):
                for action in range(nlinks[src]):
                    self.q[src][dest][action] = distance[src][dest]

    def act(self, state, nlinks,  best=False):
        n = state[0]
        dest = state[1]

        if best is True:
            best = self.q[n][dest][0]
            best_action = 0
            for action in range(nlinks[n]):
                if self.q[n][dest][action] < best:  #+ eps:
                    best = self.q[n][dest][action]
                    best_action = action
        else:
            best_action = int(np.random.choice((0.0, nlinks[n])))

        return best_action


    def learn(self, current_event, next_event, reward, action, done, nlinks):

        n = current_event[0]
        dest = current_event[1]

        n_next = next_event[0]
        dest_next = next_event[1]

        future = self.q[n_next][dest][0]
        for link in range(nlinks[n_next]):
            if self.q[n_next][dest][link] < future:
                future = self.q[n_next][dest][link]

        #Q learning
        self.q[n][dest][action] += (reward + self.config["discount"]*future - self.q[n][dest][action])* self.config["learning_rate"]

def main():
    # world = GridWorld(ROWS, COLS, goals=GOALS, obs=OBS)
    # world.save_map("map.txt")
    world = GridWorld.from_file("mesh4x4.txt")
    source = "0,0"
    dest = "3,2"
    p1_solver = SARSA(world, 0.01, source, dest)
    p2_solver = SARSA(world, 0.01, "1,0", "2,2")
    s = None
    cnt = 0
    for _ in range(N_EPISODES):
        while s != "goal" and cnt < 16:
            s, p1_state, prev_p1s = p1_solver()
            print(p1_state, type(p1_state))
            p2_solver.world.map[p1_state[0], p1_state[1]] = 1 # p1 state is an obstacle for p2
            p2_solver.world.map[prev_p1s[0], p1_state[1]] = 0 # clear the old state
            s, p2_state, prev_p2s = p2_solver()
            p1_solver.world.map[p2_state[0], p2_state[1]] = 1
            p1_solver.world.map[prev_p2s[0], p2_state[1]] = 0
            cnt += 1
            
    plot_policy(p1_solver.world, p1_solver.policy, p1_solver.values, "p1sarsa_final.html", "SARSA")
    plot_policy(p2_solver.world, p2_solver.policy, p2_solver.values, "p2sarsa_final.html", "SARSA")
    
  
if __name__ == "__main__":
    main()
