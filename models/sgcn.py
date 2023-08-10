import random
import numpy as np
import math

import torch
import torch.nn as nn
from torch.nn import functional as F


def reset_parameters_linear(layer): 
    torch.nn.init.kaiming_uniform_(layer.weight, a=np.sqrt(5)) 
    if layer.bias is not None: 
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(layer.weight) 
        bound = 1 / np.sqrt(fan_in) 
        torch.nn.init.uniform_(layer.bias, -bound, bound) 


class AsymmetricConvolution(nn.Module):

    def __init__(self, in_cha, out_cha):
        super(AsymmetricConvolution, self).__init__()

        self.conv1 = nn.Conv2d(in_cha, out_cha, kernel_size=(3, 1), padding=(1, 0), bias=False)
        self.conv2 = nn.Conv2d(in_cha, out_cha, kernel_size=(1, 3), padding=(0, 1))

        self.shortcut = nn.Identity()

        if in_cha != out_cha:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_cha, out_cha, 1, bias=False)
            )

        self.activation = nn.PReLU()

    def forward(self, x):

        shortcut = self.shortcut(x)

        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x2 = self.activation(x2 + x1)

        return x2 + shortcut


class InteractionMask(nn.Module):

    def __init__(self, n_asymmetric_convs, spatial_channels, temporal_channels):
        super(InteractionMask, self).__init__()

        self.n_asymmetric_convs = n_asymmetric_convs

        self.spatial_asymmetric_convolutions = nn.ModuleList()
        self.temporal_asymmetric_convolutions = nn.ModuleList()

        for i in range(self.n_asymmetric_convs):
            self.spatial_asymmetric_convolutions.append(
                AsymmetricConvolution(spatial_channels, spatial_channels)
            )
            self.temporal_asymmetric_convolutions.append(
                AsymmetricConvolution(temporal_channels, temporal_channels)
            )

        self.spatial_output = nn.Sigmoid()
        self.temporal_output = nn.Sigmoid()

    def forward(self, dense_spatial_interaction, dense_temporal_interaction, threshold=0.5):

        assert len(dense_temporal_interaction.shape) == 4
        assert len(dense_spatial_interaction.shape) == 4

        for j in range(self.n_asymmetric_convs):
            dense_spatial_interaction = self.spatial_asymmetric_convolutions[j](dense_spatial_interaction)
            dense_temporal_interaction = self.temporal_asymmetric_convolutions[j](dense_temporal_interaction)

        spatial_interaction_mask = self.spatial_output(dense_spatial_interaction)
        temporal_interaction_mask = self.temporal_output(dense_temporal_interaction)

        spatial_zero = torch.zeros_like(spatial_interaction_mask, device=dense_spatial_interaction.device)
        temporal_zero = torch.zeros_like(temporal_interaction_mask, device=dense_temporal_interaction.device)

        spatial_interaction_mask = torch.where(spatial_interaction_mask > threshold, spatial_interaction_mask,
                                               spatial_zero)

        temporal_interaction_mask = torch.where(temporal_interaction_mask > threshold, temporal_interaction_mask,
                                               temporal_zero)

        return spatial_interaction_mask, temporal_interaction_mask


class ZeroSoftmax(nn.Module):

    def __init__(self):
        super(ZeroSoftmax, self).__init__()

    def forward(self, x, dim=0, eps=1e-5):
        x_exp = torch.pow(torch.exp(x) - 1, exponent=2)
        x_exp_sum = torch.sum(x_exp, dim=dim, keepdim=True)
        x = x_exp / (x_exp_sum + eps)
        return x


class SelfAttention(nn.Module):

    def __init__(self, in_dims, d_model, n_heads, device):
        super(SelfAttention, self).__init__()

        self.embedding = nn.Linear(in_dims, d_model)
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)

        self.scaled_factor = torch.sqrt(torch.Tensor([d_model])).to(device)
        self.softmax = nn.Softmax(dim=-1)

        self.n_heads = n_heads
        
        reset_parameters_linear(self.embedding)
        reset_parameters_linear(self.query)
        reset_parameters_linear(self.key)

    def split_heads(self, x):

        x = x.reshape(x.shape[0], -1, self.n_heads, x.shape[-1] // self.n_heads).contiguous()

        return x.permute(0, 2, 1, 3) 

    def forward(self, x, mask=False, multi_head=False):

        assert len(x.shape) == 3

        embeddings = self.embedding(x)  
        query = self.query(embeddings) 
        key = self.key(embeddings)   

        if multi_head:
            query = self.split_heads(query)  
            key = self.split_heads(key) 
            attention = torch.matmul(query, key.permute(0, 1, 3, 2))  
        else:
            attention = torch.matmul(query, key.permute(0, 2, 1))

        attention = self.softmax(attention / self.scaled_factor)

        if mask is True:

            mask = torch.ones_like(attention)
            attention = attention * torch.tril(mask)

        return attention, embeddings

    
class SpatialTemporalFusion(nn.Module):

    def __init__(self, n_frames):
        super(SpatialTemporalFusion, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(n_frames, n_frames, 1),
            nn.BatchNorm2d(n_frames),
            nn.PReLU(),
            nn.Conv2d(n_frames, n_frames, 1),
            nn.BatchNorm2d(n_frames)
        )

        self.shortcut = nn.Sequential(
            nn.Conv2d(n_frames, n_frames, 1),
            nn.BatchNorm2d(n_frames)
        )

    def forward(self, x):

        out = self.conv(x)
        out += self.shortcut(x)

        out = F.relu(out)

        return out.squeeze()


class SparseWeightedAdjacency(nn.Module):

    def __init__(self, spa_in_dims, tem_in_dims, embedding_dims, n_heads, n_frames, dropout, n_asymmetric_convs, n_nodes, device):
        super(SparseWeightedAdjacency, self).__init__()

        # positional encoder
        #self.positional_encoder = PositionalEncoder(embedding_dims)
        self.pes = PositionalEncoding (channel = spa_in_dims, joint_num = n_nodes, time_len = n_frames, domain = "spatial")
        self.pet = PositionalEncoding (channel = tem_in_dims, joint_num = n_nodes, time_len = n_frames, domain = "temporal")

        ##Regulation
        #self.attention0s = nn.Parameter(torch.ones(1, n_heads, 22, 22) + torch.eye(22),
                                 #               requires_grad=True)
        # dense interaction
        self.spatial_attention = SelfAttention(spa_in_dims, embedding_dims, n_heads, device)
        self.temporal_attention = SelfAttention(tem_in_dims, embedding_dims, n_heads, device)

        # attention fusion
        self.spa_fusion = SpatialTemporalFusion(n_frames=n_frames)

        # interaction mask
        self.interaction_mask = InteractionMask(n_asymmetric_convs=n_asymmetric_convs,
                                                spatial_channels=4,
                                                temporal_channels=4)

        self.dropout = dropout
        self.zero_softmax = ZeroSoftmax()
        self.n_nodes = n_nodes

    def forward(self, graph, identity):

        assert len(graph.shape) == 3

        spatial_graph = graph[:, :, :]

        temporal_graph_emb = spatial_graph.permute(1, 0, 2)

        spatial_graph_emb = self.pet(spatial_graph)

        dense_spatial_interaction, spatial_embeddings = self.spatial_attention(spatial_graph_emb, multi_head=True)
        
        dense_temporal_interaction, temporal_embeddings = self.temporal_attention(temporal_graph_emb, multi_head=True)
        
        #dense_spatial_interaction = dense_spatial_interaction + self.attention0s.repeat(self.n_nodes, 1, 1, 1)

        # attention fusion
        st_interaction = self.spa_fusion(dense_spatial_interaction.permute(1, 0, 2, 3)).permute(1, 0, 2, 3)
        ts_interaction = dense_temporal_interaction
        spatial_mask, temporal_mask = self.interaction_mask(st_interaction, ts_interaction)

        # self-connected
        spatial_mask = spatial_mask + identity[0].unsqueeze(1)
        temporal_mask = temporal_mask + identity[1].unsqueeze(1)

        normalized_spatial_adjacency_matrix = self.zero_softmax(dense_spatial_interaction * spatial_mask, dim=-1)
        normalized_temporal_adjacency_matrix = self.zero_softmax(dense_temporal_interaction * temporal_mask, dim=-1)

        return normalized_spatial_adjacency_matrix, normalized_temporal_adjacency_matrix,\
               spatial_embeddings, temporal_embeddings
    


class GraphConvolution(nn.Module):

    def __init__(self, in_dims, embedding_dims, dropout):
        super(GraphConvolution, self).__init__()

        self.embedding = nn.Linear(in_dims, embedding_dims, bias=False)
        self.activation = nn.PReLU()

        self.dropout = dropout
        
        reset_parameters_linear(self.embedding)

    def forward(self, graph, adjacency):

        gcn_features = self.embedding(torch.matmul(adjacency, graph))
        gcn_features = F.dropout(self.activation(gcn_features), p=self.dropout)
        return gcn_features

class SparseGraphConvolution(nn.Module):

    def __init__(self, num_layers, in_dims, embedding_dims, dropout=0.0):
        super(SparseGraphConvolution, self).__init__()

        self.dropout = dropout

        self.spatial_temporal_sparse_gcn = nn.ModuleList()
        self.temporal_spatial_sparse_gcn = nn.ModuleList()

        for i in range(num_layers):
            if i == 0:
                self.spatial_temporal_sparse_gcn.append(GraphConvolution(in_dims, embedding_dims, dropout=dropout))
                self.temporal_spatial_sparse_gcn.append(GraphConvolution(in_dims, embedding_dims, dropout=dropout))
            else:
                self.spatial_temporal_sparse_gcn.append(GraphConvolution(embedding_dims, embedding_dims, dropout=dropout))
                self.temporal_spatial_sparse_gcn.append(GraphConvolution(embedding_dims, embedding_dims, dropout=dropout)) 

    def forward(self, graph, normalized_spatial_adjacency_matrix, normalized_temporal_adjacency_matrix):

        spa_graph = graph.permute(1, 0, 2, 3)  
        tem_graph = spa_graph.permute(2, 1, 0, 3)  

        gcn_spatial_features = self.spatial_temporal_sparse_gcn[0](spa_graph, normalized_spatial_adjacency_matrix)
        gcn_spatial_features = gcn_spatial_features.permute(2, 1, 0, 3)

        for i in range(1, len(self.spatial_temporal_sparse_gcn)):
            gcn_spatial_features = self.spatial_temporal_sparse_gcn[i](gcn_spatial_features, normalized_temporal_adjacency_matrix)
            #gcn_spatial_features = F.dropout(gcn_spatial_features, self.dropout, training=self.training)

        gcn_temporal_features = self.temporal_spatial_sparse_gcn[0](tem_graph,
                                                                   normalized_temporal_adjacency_matrix)
        gcn_temporal_features = gcn_temporal_features.permute(2, 1, 0, 3)

        for i in range(1, len(self.temporal_spatial_sparse_gcn)):
            gcn_temporal_features = self.temporal_spatial_sparse_gcn[i](gcn_temporal_features,
                                                                            normalized_spatial_adjacency_matrix)
            #gcn_temporal_features = F.dropout(gcn_temporal_features, self.dropout, training=self.training)

        gcn_temporal_spatial_features = gcn_temporal_features.permute(2, 1, 0, 3)

        return gcn_spatial_features, gcn_temporal_spatial_features

class PositionalEncoding(nn.Module):

    def __init__(self, channel, joint_num, time_len, domain):
        super(PositionalEncoding, self).__init__()
        self.joint_num = joint_num
        self.time_len = time_len

        self.domain = domain

        if domain == "temporal":
            # temporal embedding
            pos_list = []
            for t in range(self.time_len):
                for j_id in range(self.joint_num):
                    pos_list.append(t)
        elif domain == "spatial":
            # spatial embedding
            pos_list = []
            for t in range(self.time_len):
                for j_id in range(self.joint_num):
                    pos_list.append(j_id)

        position = torch.from_numpy(np.array(pos_list)).unsqueeze(1).float()
        # pe = position/position.max()*2 -1
        # pe = pe.view(time_len, joint_num).unsqueeze(0).unsqueeze(0)
        # Compute the positional encodings once in log space.
        pe = torch.zeros(self.time_len * self.joint_num, channel)

        div_term = torch.exp(torch.arange(0, channel, 2).float() *
                             -(math.log(10000.0) / channel))  # channel//2
        pe[:, 0::2] = torch.sin(position * div_term[1])
        pe[:, 1::2] = torch.cos(position * div_term[1])
        pe = pe.view(time_len, joint_num, channel).squeeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):  # nctv
        x = x + self.pe[:, :, :x.size(2)]
        return x

class SGCNModel(nn.Module):

    def __init__(self, args):
        super(SGCNModel, self).__init__()

        self.num_gcn_layers = args.num_gcn_layers
        self.dropout = args.dropout

        ## sparse graph learning
        self.sparse_weighted_adjacency_matrices = SparseWeightedAdjacency(spa_in_dims=args.num_features,
                                                                          tem_in_dims=args.num_features,
                                                                          embedding_dims=args.embedding_dims,
                                                                          n_heads=args.num_heads,
                                                                          n_frames=args.max_seq_len,
                                                                          dropout=args.dropout,
                                                                          n_asymmetric_convs=args.num_asymmetric_convs,
                                                                          n_nodes=args.num_nodes,
                                                                          device=args.device)

        ## graph convolution
        self.stsgcn = nn.ModuleList()
        for _ in range(args.num_gcn_layers):
            self.stsgcn.append(SparseGraphConvolution(num_layers=args.num_gcn_layers,
                                                 in_dims=args.num_features, 
                                                 embedding_dims=args.embedding_dims // args.num_heads,
                                                 dropout=args.dropout))
        
        self.fusion_ = nn.Conv2d(args.num_heads, args.num_heads, kernel_size=1, bias=False)

        ## spatial & temoral edges weights
        self.temporal_edge_weights = nn.Sequential(nn.Linear(args.num_features, args.num_heads, bias=True),
                                                   nn.Linear(args.num_heads, args.num_heads*args.max_seq_len, bias=True))
        
        self.spatial_edge_weights  = nn.Sequential(nn.Linear(args.num_features, args.num_heads, bias=True),
                                                   nn.Linear(args.num_heads, args.num_heads*args.num_nodes, bias=True))
        ## adaptive pooling
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, args.num_heads))
        
        ## MLP
        self.mlp = nn.Sequential(
            nn.Flatten(start_dim=0),
            nn.Linear((args.embedding_dims * args.num_nodes  * args.max_seq_len ) // args.num_heads, 512),
            nn.PReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(512, 128),
            nn.PReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(128, args.num_classes)
        )


    def forward(self, graph, identity):

        # graph 1 obs_len N 3
        normalized_spatial_adjacency_matrix, normalized_temporal_adjacency_matrix, spatial_embeddings, temporal_embeddings = \
            self.sparse_weighted_adjacency_matrices(graph.squeeze(), identity)
        
        #W_spatio = self.spatial_edge_weights(graph)
        #W_spatio = W_spatio.reshape(normalized_spatial_adjacency_matrix.shape)
        
        #W_tempo = self.temporal_edge_weights(graph)
        #W_tempo = W_tempo.reshape(normalized_temporal_adjacency_matrix.shape)
        
        #weighted_normalized_spatial_adjacency_matrix  = normalized_spatial_adjacency_matrix * W_spatio
        #weighted_normalized_temporal_adjacency_matrix = normalized_temporal_adjacency_matrix * W_tempo
        
        for layer in self.stsgcn:
            gcn_temporal_spatial_features, gcn_spatial_temporal_features = layer(
                graph, normalized_spatial_adjacency_matrix, normalized_temporal_adjacency_matrix
            )

        gcn_representation = self.fusion_(gcn_temporal_spatial_features) + gcn_spatial_temporal_features
        gcn_representation = gcn_representation.permute(0, 2, 1, 3)
        gcn_representation = torch.mean(gcn_representation, dim=-2)
        prediction = self.mlp(gcn_representation)

        return prediction.unsqueeze(dim=0), gcn_representation, normalized_spatial_adjacency_matrix, normalized_temporal_adjacency_matrix
