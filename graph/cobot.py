import sys
import numpy as np

sys.path.extend(['../'])
from graph import tools

num_node = 48
self_link = [(i, i) for i in range(num_node)]
inward =[
                # Arm & shoulders
                (42, 44), (44, 46), (43, 45), (45, 47), (42, 43),
                # Right hand connections
                (0,1), (1,2), (2,3), (3,4),
                (5,6), (6,7), (7,8),
                (9,10), (10,11), (11,12),
                (13,14), (14,15), (15,16),
                (17,18), (18,19), (19,20),
                # Right palm chain
                (0,5), (5,9), (9,13), (13,17), (0,17),
                # Left thumb/index/middle/ring/pinky chains
                (21,22), (22,23), (23,24), (24,25),
                (26,27), (27,28), (28,29),
                (30,31), (31,32), (32,33),
                (34,35), (35,36), (36,37),
                (38,39), (39,40), (40,41),
                # Left palm chain
                (21,26), (26,30), (30,34), (34,38), (21,38)
            ]
# inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward
edge = self_link + inward

class Graph:
    def __init__(self, labeling_mode='spatial', max_hop=1, dilation=1):
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor
        self.max_hop = max_hop
        self.dilation = dilation
        self.edge = edge
        self.hop_dis = get_hop_distance(self.num_node, self.edge, max_hop=max_hop)
        self.A = self.get_adjacency_matrix(labeling_mode)

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        elif labeling_mode == 'distance':
            valid_hop = range(0, self.max_hop + 1, self.dilation)
            adjacency = np.zeros((self.num_node, self.num_node))
            for hop in valid_hop:
                adjacency[self.hop_dis == hop] = 1
            normalize_adjacency = normalize_digraph(adjacency)
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis ==
                                                                hop]
        else:
            raise ValueError()
        return A

def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    # compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis

def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    AD = np.dot(A, Dn)
    return AD
