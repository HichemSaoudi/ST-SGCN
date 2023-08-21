import numpy as np
import torch
from mediapipe import solutions


def create_adj_matrix(dataset):
    if dataset == 'Briareo' or dataset == 'IPN':
        n_lands = len(solutions.hands.HAND_CONNECTIONS)
        A = np.zeros((n_lands, n_lands))
        for i, j in solutions.hands.HAND_CONNECTIONS:
            A[i][j] = 1.
            A[j][i] = 1.
        A += np.eye(A.shape[0])
        edge = set(solutions.hands.HAND_CONNECTIONS)
        
        return A, edge

    elif dataset == 'SHREC17': 
        num_node = 22
        A = np.zeros((num_node, num_node))
        link = [(i, i) for i in range(num_node)]
        neighbor_link = [(0, 1),
                     (0, 2),
                     (1, 0),
                     (1, 6),
                     (1, 10),
                     (1, 14),
                     (1, 18),
                     (2, 0),
                     (2, 3),
                     (3, 2),
                     (3, 4),
                     (4, 3),
                     (4, 5),
                     (5, 4),
                     (6, 1),
                     (6, 7),
                     (7, 6),
                     (7, 8),
                     (8, 7),
                     (8, 9),
                     (9, 8),
                     (10, 1),
                     (10, 11),
                     (11, 10),
                     (11, 12),
                     (12, 11),
                     (12, 13),
                     (13, 12),
                     (14, 1),
                     (14, 15),
                     (15, 14),
                     (15, 16),
                     (16, 15),
                     (16, 17),
                     (17, 16),
                     (18, 1),
                     (18, 19),
                     (19, 18),
                     (19, 20),
                     (20, 19),
                     (20, 21),
                     (21, 20)]
        edge = link + neighbor_link
        for i, j in edge:
            A[i][j] = 1.
        
        A += np.eye(A.shape[0])
        
        return A, edge


    elif dataset == 'SHREC21':
        num_node = 20
        A = np.zeros((num_node, num_node))
        link = [(i, i) for i in range(num_node)]
        neighbor_link = [(0, 1),
                     (1, 2),
                     (2, 3),
                     (0, 4),
                     (4, 5),
                     (5, 6),
                     (6, 7),
                     (0, 8),
                     (8, 9),
                     (9, 10),
                     (10, 11),
                     (0, 12),
                     (12, 13),
                     (13, 14),
                     (14, 15),
                     (0, 16),
                     (16, 17),
                     (17, 18),
                     (18, 19)]
        edge = link + neighbor_link
        for i, j in edge:
            A[i][j] = 1.
            A[j][i] = 1.
        
        A += np.eye(A.shape[0])
        
        return A, edge

    else:
        raise NotImplementedError("The adjacency matrix is implemented for datasets {'Briareo', 'IPN', 'SHREC17', 'SHREC21'}! for another dataset, you need to implement the corresponding adjacency matrix")
        


def get_sgcn_identity(shape, device):
    identity_spatial = torch.from_numpy(np.array([hand_adj_matrix] * shape[1])).type(torch.float32).to(device)
    identity_temporal = torch.triu(torch.ones((shape[2], shape[1], shape[1]), device=device), diagonal=1)        
    return [identity_spatial, identity_temporal]


def calculate_connectivity(adj_matrix, edges):
    connectivity = {}
    num_nodes = len(adj_matrix)

    for i in range(num_nodes):
        connectivity[i] = 0  # Initialize connectivity to 0 for each node

    for edge in edges:
        source, target = edge
        connectivity[source] += 1

    return connectivity

