import NetworkMdp
import TrafficGenerator
import numpy as np
import random

Alpha = 0.9
EPSILON = 0.05
Clambda = 0.9

class Q_routing:
     def __init__(self, mapfile, alpha, EPSILON, Clambda):
          
        self.epsilon = epsilon
        self.alpha = alpha
        self.confidence_based = confidence_based
        self.clambda = clambda
          
        self.q = 10000 * np.ones((self.network.nodes.shape[0], self.network.nodes.shape[1], len(NetworkMdp.actions)))

    def send_packet(self, origin, max_hops=100):
        current_node = origin
        action = self.get_action(current_node)

        hops = 0
        nodes = []
        while self.network.node(current_node) != NetworkMdp.DESTINATION:
            next_node, reward = self.network.next_node(current_node, NetworkMdp.actions[action])
            next_action = self.get_action(next_node)

            q_current = self.q[current_node[0]][current_node[1]][action]
            q_next = self.best_q(next_node)

            new_q = q_current + self.alpha * (q_next - q_current)

            self.q[current_node[0]][current_node[1]][action] = new_q
            self.network.set_policy(current_node, action)

            nodes.append((current_node, action))
            current_node = next_node
            action = next_action

            hops = hops + 1

            if hops > max_hops:
                break

        time = hops
        for node, action in nodes:
            old_estimate = self.q[node[0]][node[1]][action]
            
            self.q[node[0]][node[1]][action] = time
            
            if self.confidence_based:
                # update confidence levels
                if (time > old_estimate):
                    # decrease confidence, limited to 0.0
                    self.c[node[0]][node[1]][action] *= 0.9
                if (time <= old_estimate):
                    # increase confidence, limited to 1.0
                    self.c[node[0]][node[1]][action] *= 1.1
                    if self.c[node[0]][node[1]][action] > 1:
                        self.c[node[0]][node[1]][action] = 1.0
                for a in NetworkMdp.actions.keys():
                    self.c[node[0]][node[1]][a] *= self.clambda # discount factor
            time = time - 1

        return hops

    def get_action(self, node):
        if random.uniform(0, 1) <= self.epsilon:
            # Randomly return an action
            return random.randint(0, len(NetworkMdp.actions) - 1)

        best_q = float('inf')
        best_actions = []
        for action in NetworkMdp.actions.keys():
            if self.confidence_based:
                q = self.q[node[0]][node[1]][action]*(2-self.c[node[0]][node[1]][action])
            else:
                q = self.q[node[0]][node[1]][action]

            if q < best_q:
                best_q = q
                best_actions.clear()
                best_actions.append(action)
            elif q == best_q:
                best_actions.append(action)

        if len(best_actions) == 0:
            # No actions found that maximize the Q function, randomly pick one
            return np.random.randint(0, len(NetworkMdp.actions) - 1)
        else:
            # Randomly pick an action from the best actions that maximized the Q function
            return best_actions[np.random.randint(0, len(best_actions))]

    def best_q(self, node):
        best_q = float('inf')
        for action in NetworkMdp.actions.keys():
            q = self.q[node[0]][node[1]][action]

            if q < best_q:
                best_q = q

        return best_q

def main():
     qroute = Q_Routing("mesh4x4.txt")
     traffic = TrafficGenerator.TrafficGenerator(qroute)
     traffic.simulate(10000, "Q_rounting", True)
    
  
if __name__ == "__main__":
    main()
