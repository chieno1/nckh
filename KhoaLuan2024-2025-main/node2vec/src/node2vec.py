import numpy as np
import networkx as nx
import random


class Graph:
    def __init__(self, nx_G, is_directed, p, q):
        self.G = nx_G
        self.is_directed = is_directed
        self.p = p
        self.q = q
        self.alias_nodes = {}
        self.alias_edges = {}

    def node2vec_walk(self, walk_length, start_node):
        """
        Simulate a random walk starting from the start node.
        """
        G = self.G
        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges

        walk = [start_node]

        while len(walk) < walk_length:
            cur = walk[-1]

            if cur not in alias_nodes:
                print(f"❌ LỖI: Đỉnh {cur} không có trong alias_nodes!")
                break  # Tránh lỗi KeyError

            cur_nbrs = sorted(G.neighbors(cur))
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    walk.append(cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
                else:
                    prev = walk[-2]
                    if (prev, cur) in alias_edges:
                        walk.append(cur_nbrs[alias_draw(alias_edges[(prev, cur)][0], alias_edges[(prev, cur)][1])])
                    else:
                        print(f"⚠️ CẢNH BÁO: Không có alias edge cho ({prev}, {cur})")
                        break
            else:
                break

        return walk

    def simulate_walks(self, num_walks, walk_length):
        """
        Repeatedly simulate random walks from each node.
        """
        G = self.G
        walks = []
        nodes = list(G.nodes())
        print("Walk iteration:")

        for walk_iter in range(num_walks):
            print(f"{walk_iter + 1} / {num_walks}")
            random.shuffle(nodes)
            for node in nodes:
                walks.append(self.node2vec_walk(walk_length=walk_length, start_node=node))

        return walks

    def get_alias_edge(self, src, dst):
        """
        Get the alias edge setup lists for a given edge.
        """
        G = self.G
        p = self.p
        q = self.q

        unnormalized_probs = []
        for dst_nbr in sorted(G.neighbors(dst)):
            if dst_nbr == src:
                unnormalized_probs.append(G[dst][dst_nbr].get('weight', 1.0) / p)
            elif G.has_edge(dst_nbr, src):
                unnormalized_probs.append(G[dst][dst_nbr].get('weight', 1.0))
            else:
                unnormalized_probs.append(G[dst][dst_nbr].get('weight', 1.0) / q)

        norm_const = sum(unnormalized_probs)
        if norm_const == 0:
            print(f"⚠️ CẢNH BÁO: Cạnh ({src}, {dst}) có tổng trọng số = 0!")
            return alias_setup([1.0])  # Tránh lỗi chia cho 0

        normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]
        return alias_setup(normalized_probs)

    def preprocess_transition_probs(self):
        """
        Preprocessing of transition probabilities for guiding the random walks.
        """
        G = self.G
        is_directed = self.is_directed

        alias_nodes = {}
        for node in G.nodes():
            neighbors = list(G.neighbors(node))
            if not neighbors:
                print(f"⚠️ CẢNH BÁO: Đỉnh {node} không có cạnh nào, bỏ qua.")
                continue
            #unnormalized_probs = [np.exp(-G[node][nbr].get('weight', 1.0)) for nbr in sorted(neighbors)]
            unnormalized_probs = [G[node][nbr].get('weight', 1.0) for nbr in sorted(neighbors)]
            norm_const = sum(unnormalized_probs)

            if norm_const == 0:
                print(f"⚠️ CẢNH BÁO: Đỉnh {node} có tổng trọng số = 0, bỏ qua.")
                continue

            normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]
            alias_nodes[node] = alias_setup(normalized_probs)

        alias_edges = {}

        if is_directed:
            for edge in G.edges():
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
        else:
            for edge in G.edges():
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
                alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])

        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges


def alias_setup(probs):
    """
    Compute utility lists for non-uniform sampling from discrete distributions.
    Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    for details.
    """
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int64)

    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K * prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while smaller and larger:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q


def alias_draw(J, q):
    """
    Draw sample from a non-uniform discrete distribution using alias sampling.
    """
    K = len(J)

    kk = int(np.floor(np.random.rand() * K))
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]
