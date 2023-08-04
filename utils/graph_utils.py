
import numpy as np
import torch
from mediapipe import solutions


def adj_matrix():
    n_lands = len(solutions.hands.HAND_CONNECTIONS)
    A = np.zeros((n_lands, n_lands))
    for i, j in solutions.hands.HAND_CONNECTIONS:
        A[i][j] = 1.
        A[j][i] = 1.
        
    A += np.eye(A.shape[0])
        
    return A


hand_adj_matrix = adj_matrix()


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

