import numpy as np
from bokeh.plotting import figure, save, show

# Types of nodes in the network
DESTINATION = -1    # Destination of our packet
ACTIVE = 0          # Node that's active and can forward packets
INACTIVE = 1        # Node that's inactive and cannot forward packets

# Routing actions
RIGHT = 0
LEFT = 1
UP = 2
DOWN = 3

actions = {
    RIGHT: [0, 1],
    LEFT: [0, -1],
    UP: [1, 0],
    DOWN: [-1, 0]
}

TIMEOUT = -1


class NetworkMdp:
    def __init__(self, map_file):
        self.nodes = np.loadtxt(open(map_file, "rb"), delimiter=" ")
        self.values = np.zeros(self.nodes.shape)
        self.policies = np.random.randint(0, 4, self.nodes.shape)

    # Plot the network and its policies
    def render(self, method, filename):
        x_pos = []
        y_pos = []
        color = []
        arrow = []
        arrow_dir = {DOWN: 0.0, UP: np.pi, RIGHT: -np.pi / 2.0, LEFT: np.pi / 2.0}

        n_feat = 0
        for i in range(self.nodes.shape[0]):
            for j in range(self.nodes.shape[1]):
                if self.nodes[i, j] == INACTIVE:
                    x_pos.append(f"{j}")
                    y_pos.append(f"{i}")
                    color.append("#000000")
                    n_feat += 1
                elif self.nodes[i, j] == DESTINATION:
                    x_pos.append(f"{j}")
                    y_pos.append(f"{i}")
                    color.append("#00FF00")
                    n_feat += 1

        for i in range(self.nodes.shape[0]):
            for j in range(self.nodes.shape[1]):
                if self.nodes[i, j] == ACTIVE:
                    x_pos.append(f"{j}")
                    y_pos.append(f"{i}")
                    color.append(f"#FFFFFF")
                    arrow.append(arrow_dir[self.policies[i, j]])

        fig = figure(
            title=f"{method}: Network Policy",
            x_range=[f"{i}" for i in np.arange(self.nodes.shape[0])],
            y_range=[f"{i}" for i in np.flip(np.arange(self.nodes.shape[1]))]
        )
        fig.rect(x_pos, y_pos, color=color, width=1, height=1)
        fig.triangle(x_pos[n_feat:], y_pos[n_feat:], angle=arrow, size=400 / np.sum(self.nodes.shape), color="#FF0000",
                     alpha=0.5)

        save(fig, filename)
        show(fig)

    # Given a current node and an action taken, return the next node and the given reward
    def next_node(self, current_node, action):
        next_node = (current_node[0] + action[0], current_node[1] + action[1])
        if not self.in_bounds(next_node):
            # Can't route off of the network (no node exists here), stay in the current state
            next_node = current_node

        if self.node(next_node) == INACTIVE:
            # Can't route to this node (it's inactive), stay in the current state and exit
            next_node = current_node

        # Reward is -1 unless the action takes us to the goal (in which case the reward is 0)
        reward = -1
        if self.node(next_node) == DESTINATION:
            reward = 0

        return [next_node, reward]

    # Check if a node is in bounds, i.e. on the network
    def in_bounds(self, node):
        if node[0] < 0 or node[0] >= self.nodes.shape[0]:
            return False
        elif node[1] < 0 or node[1] >= self.nodes.shape[1]:
            return False
        else:
            return True

    # Get the type of node (e.g. destination, inactive, active) for a given state
    def node(self, node):
        return self.nodes[node[0], node[1]]

    # Get the policy for a given node
    def policy(self, node):
        return self.policies[node[0], node[1]]

    # Set the policy for a given node
    def set_policy(self, node, action):
        self.policies[node[0], node[1]] = action

    # Convenience function for getting the value function of a given node
    def value(self, node):
        return self.values[node[0], node[1]]

    def send_packet(self, origin, max_hops=100):
        if self.node(origin) == INACTIVE:
            return TIMEOUT

        hops = 0
        current_node = origin
        while self.node(current_node) != DESTINATION:
            action = actions[self.policy(current_node)]
            current_node, reward = self.next_node(current_node, action)
            hops = hops + 1

            if hops > max_hops:
                return TIMEOUT

        return hops
