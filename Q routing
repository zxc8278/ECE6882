import numpy as np
from bokeh.plotting import figure, save, show
import copy
from finitemdp import GridWorld

N_EPISODES = 1000
MAX_LOOPS = 100
GAMMA = 0.9
EPSILON = 0.01

class SARSA(object):
    def __init__(self, world, alpha, source, dest):
        self.world = world
        dest = dest.split(",")
        self.world.map[int(dest[0]), int(dest[1])] = -1
        print(self.world.map)
        self.source = source
        self.alpha = alpha
        
        self.q_func = {s: {"up": len(world.states)/400,
                      "down": len(world.states)/400,
                      "left": len(world.states)/400,
                      "right": len(world.states)/400} for s in world.states}
        self.policy = {s: np.random.choice(world.actions) for s in world.states}
        self.values = {s: 0.0 for s in world.states}
        self.q_func["goal"] = {"up": 0, "down": 0, "left": 0, "right": 0}
        
        self.hist = []
        self.rewards = []
        self.actions = []
        self.states = []
    
    def __call__(self):
        s = self.source #np.random.choice(self.world.states) #self.source # P [s0, a0] > 0 for all s in STATES, a in ACTIONS
        
        # generate episode using policy
        #while s != "goal":
        if np.random.random() <= EPSILON: # e-greedy SARSA
            # explore
            a = np.random.choice(self.world.actions)
        else:
            # use optimal action from policy
            a = self.policy.get(s)
        self.states.append(s)
        self.actions.append(a)
        s_new, r = self.world.sar_dynamics(s, a)
        self.rewards.append(r)
        
        if np.random.random() <= EPSILON: # e-greedy SARSA
            # explore
            a_new = np.random.choice(self.world.actions)
        else:
            a_new = self.policy.get(s)
                
        #print(s, s_new, a, a_new)
        # now in s_new, it is an obstacle in other worlds now so remove it.
        # s is available again so add it back to the other worlds.

        self.q_func[s][a] = self.q_func[s][a] + self.alpha*(r + GAMMA*self.q_func[s_new][a_new] - self.q_func[s][a])
        q_list = [self.q_func[s]["up"], self.q_func[s]["down"], self.q_func[s]["left"], self.q_func[s]["right"]]
        optimal_action = np.argmax(q_list)
        self.policy[s] = {0:"up", 1:"down", 2:"left", 3:"right"}.get(optimal_action)
        self.values[s] = q_list[optimal_action]
        #print(self.policy[s])
        s_prev = s.split(",")
        s_current = s_new.split(",")
        s = s_new
        a = a_new
        self.hist.append(copy.copy(self.q_func))

        return s, [int(x) for x in s_current], [int(y) for y in s_prev]
            
def plot_policy(world, pol, values, filename, method_title="Dynamic Programming"):
    max_val = np.max([v for v in values.values()])
    min_val = np.min([v for v in values.values()])
    if min_val == max_val:
        values_norm = {k: 0.0 for k in values.keys()}
    else:
        values_norm = {k: (v - min_val)/(max_val - min_val) for k, v in values.items()}

    x_pos = []
    y_pos = []
    color = []
    arrow = []
    dir = {"up": 0.0, "down": np.pi, "right": -np.pi/2.0, "left": np.pi/2.0}

    n_feat = 0
    for i in range(world.map.shape[0]):
        for j in range(world.map.shape[1]):
            if world.map[i, j] == 1:
                x_pos.append(f"{j}")
                y_pos.append(f"{i}")
                color.append("#FF0000")
                n_feat += 1
            elif world.map[i, j] == -1:
                print("found goal")
                x_pos.append(f"{j}")
                y_pos.append(f"{i}")
                color.append("#00FF00")
                n_feat += 1

    for s in world.states:
        if s == "goal":
            continue
        i, j = [int(v) for v in s.split(",")]
        x_pos.append(f"{j}")
        y_pos.append(f"{i}")
        color.append(f"#0000FF{format(int(255*values_norm[s]), '02X')}")
        arrow.append(dir[pol[s]])

    fig = figure(
        title=f"{method_title}: GridWorld Policy and Values",
        x_range=[f"{i}" for i in np.arange(world.map.shape[0])],
        y_range=[f"{i}" for i in np.flip(np.arange(world.map.shape[1]))]
    )
    fig.rect(x_pos, y_pos, color=color, width=1, height=1)
    fig.triangle(x_pos[n_feat:], y_pos[n_feat:], angle=arrow, size=400/np.sum(world.map.shape), color="#FF0000", alpha=0.5)
     
    save(fig, filename)
    show(fig)

def plot_values(world, values, filename):
    max_val = np.max([v for v in values.values()])
    min_val = np.min([v for v in values.values()])
    if min_val == max_val:
        values_norm = {k: 0.0 for k in values.keys()}
    else:
        values_norm = {k: (v - min_val)/(max_val - min_val) for k, v in values.items()}

    x_pos = []
    y_pos = []
    color = []

    n_feat = 0
    for i in range(world.map.shape[0]):
        for j in range(world.map.shape[1]):
            if world.map[i, j] == 1:
                x_pos.append(f"{j}")
                y_pos.append(f"{i}")
                color.append("#FF0000")
                n_feat += 1
            elif world.map[i, j] == -1:
                x_pos.append(f"{j}")
                y_pos.append(f"{i}")
                color.append("#00FF00")
                n_feat += 1 

    for s in world.states:
        if s == "goal":
            continue
        i, j = [int(v) for v in s.split(",")]
        x_pos.append(f"{j}")
        y_pos.append(f"{i}")
        color.append(f"#0000FF{format(int(255*values_norm[s]), '02X')}")

    fig = figure(
        title="Dynamic Programing: GridWorld Values",
        x_range=[f"{i}" for i in np.arange(world.map.shape[0])],
        y_range=[f"{i}" for i in np.flip(np.arange(world.map.shape[1]))]
    )
    fig.rect(x_pos, y_pos, color=color, width=1, height=1)

    save(fig, filename)
    show(fig)


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
