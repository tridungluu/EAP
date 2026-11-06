import numpy as np

class Graph:
    def __init__(self, layout='cobot', strategy='spatial', max_hop=1, dilation=1, **kwargs):
        if 'labeling_mode' in kwargs:
            strategy = kwargs['labeling_mode']
        self.max_hop = max_hop
        self.dilation = dilation
        self.num_node = 48
        self.center = 0 # Khớp gốc (hub) số 0 hoặc 21 tùy cấu trúc robot
        
        # Lấy danh sách xương nối từ code AimCLR của bạn
        self.neighbor_link = [
    # Cụm Bàn tay 1 (Hand 1 - nodes 0-20)
    (0, 1), (1, 2), (2, 3), (3, 4), 
    (0, 5), (5, 6), (6, 7), (7, 8), 
    (5, 9), (9, 10), (10, 11), (11, 12), 
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),

    # Cụm Bàn tay 2 (Hand 2 - nodes 21-41)
    (21, 22), (22, 23), (23, 24), (24, 25), 
    (21, 26), (26, 27), (27, 28), (28, 29),
    (26, 30), (30, 31), (31, 32), (32, 33), 
    (30, 34), (34, 35), (35, 36), (36, 37),
    (34, 38), (21, 38), (38, 39), (39, 40), (40, 41),

    # Nodes 42-47: kết nối nội bộ + GẮN VÀO graph chính
    (42, 44), (44, 46), (43, 45), (45, 47), (42, 43),
    
    
    ]
        
        self.self_link = [(i, i) for i in range(self.num_node)]
        self.edge = self.self_link + self.neighbor_link
        
        # Tính toán ma trận khoảng cách hop
        self.hop_dis = self.get_hop_distance(self.num_node, self.edge, max_hop=max_hop)
        
        # Khởi tạo ma trận kề theo chiến lược Spatial
        self.A = self.get_adjacency(strategy)

    def get_hop_distance(self, num_node, edge, max_hop=1):
        A = np.zeros((num_node, num_node))
        for i, j in edge:
            A[i, j] = 1
            A[j, i] = 1
        hop_dis = np.zeros((num_node, num_node)) + np.inf
        transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
        for d in range(max_hop + 1):
            hop_dis[transfer_mat[d] > 0] = np.minimum(hop_dis[transfer_mat[d] > 0], d)
        return hop_dis

    def get_adjacency(self, strategy):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = self.normalize_undigraph(adjacency)

        if strategy == 'spatial':
            return self.get_spatial_graph(normalize_adjacency)
        
        
        elif strategy == 'distance':
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis ==
                                                                hop]
            return A
        
        else:
            return normalize_adjacency

    def normalize_undigraph(self, A):
        # Chuẩn hóa ma trận kề chuẩn của GCN
        Dl = np.sum(A, 0)
        num_node = A.shape[0]
        Dn = np.zeros((num_node, num_node))
        for i in range(num_node):
            if Dl[i] > 0:
                Dn[i, i] = Dl[i]**(-0.5)
        DAD = np.dot(np.dot(Dn, A), Dn)
        return DAD

    def get_spatial_graph(self, normalize_adjacency):
        # Chia làm 3 tập hợp (subset) theo chiến lược của CTR-GCN
        A = np.zeros((3, self.num_node, self.num_node))
        for i in range(self.num_node):
            for j in range(self.num_node):
                if self.hop_dis[i, j] <= self.max_hop:
                    if self.hop_dis[i, j] == 0:
                        A[0, i, j] = normalize_adjacency[i, j]
                    elif self.hop_dis[self.center, j] > self.hop_dis[self.center, i]:
                        A[1, i, j] = normalize_adjacency[i, j]
                    else:
                        A[2, i, j] = normalize_adjacency[i, j]
        return A