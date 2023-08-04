
import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from utils.data_utils import top_k, interpolate_landmarks, upsample, compute_motion_features



class ShrecDataset(Dataset):
    def __init__(self,
                 data_dir, 
                 annotations_file,
                 max_seq_len,
                 connectivity,
                 labels_encoder=None):
        
        ## data
        self.data_dir = data_dir
        self.annotations_file = annotations_file
        
        ## sequence arguments
        self.max_seq_len = max_seq_len
        self.labels_encoder = labels_encoder
        
        ## moving & static lands
        self.static_lands = [0, 5, 9, 13, 17]
        self.moving_lands = list(set(range(21)) - set(self.static_lands))
        
        self.lambda_moving = 10
        self.lambda_static = 3
        self.lambda_global = 1
        
        self.connectivity = connectivity
        self.num_connections = {land:0 for land in range(21)}
        for i, j in solutions.hands.HAND_CONNECTIONS:
            self.num_connections[i] += 1

        ## load all sequences paths
        self.data = self.get_sequences_paths()   
        
    
    def read_text_file(self, src_path):
        with open(src_path, 'r') as file:
            content = file.read()
        return content
            

    def get_sequences_paths(self):
        
        data = []
        with open(self.annotations_file) as f:
            seq = f.read()
        seq = seq.split("\n")
        for line in seq[:-1]:
            params = line.split(" ")
            g_id = params[0]
            f_id = params[1]
            sub_id = params[2]
            e_id = params[3]
            src_path = self.data_dir + \
                "/gesture_{}/finger_{}/subject_{}/essai_{}/skeletons_world.txt".format(
                    g_id, f_id, sub_id, e_id)
            landmarks = self.load_landmarks(src_path)
            label = int(g_id) - 1
            if landmarks is not None:
                #aug_landmarks = self.random_moving(landmarks)
                data.append((landmarks, label))
            #if (self.use_data_aug) and (landmarks is not None) and (torch.rand(1)>0.5) :
             #   landmarks = self.noise(landmarks)
             #   landmarks = self.random_moving(landmarks)
             #   data.append((landmarks, label))

        return data


    def load_landmarks(self, txt_file):
        sequence_landmarks = []
        with open(txt_file) as f:
            data = f.read()

        data = data.split("\n")
        for line in data:
            line = line.split("\n")[0]
            dat = line.split(" ")
            point = []
            landmarks = []
            i=0
            for data_ele in dat :
                if data_ele == '' :
                    point.append(float(0))
                else :
                    point.append(float(data_ele))
                if (len(landmarks) == 22) :
                        continue    
                #print(data_ele)
                if len(point) == 3:
                    
                    #point = [float(x) for x in point]
                    point = [self.connectivity[i]/6] + [float(x) for x in point]
                    #point = [float(x) for x in point]
                    #print(point)
                    #spher_coords = self.cartesian_to_polar(coords)
                    landmarks.append(point)
                    #print(f"The shape of the landmarks is {len(landmarks)}")
                    point = []
                    i+=1
                    
                
                #landmarks = [[-1.0, -1.0, -1.0, -1.0]] * 21
            
            if len(landmarks) < 2 :
                continue
            landmarks = np.array(landmarks).astype(np.float32)
            #print(f"The shape of the landmarks is {landmarks.shape}")
            #speed, accel = self.compute_motion_features(landmarks)
            #print(f"The shape of the speed is {speed.shape}")
            #print(f"The shape of the accel is {accel.shape}")
            #features = np.hstack((landmarks, speed, accel))
            #print(f"The shape of the accel is {features.shape}")
            sequence_landmarks.append(landmarks)
        
        sequence_landmarks = np.array(sequence_landmarks).astype(np.float32)
        #print(sequence_landmarks.shape)
        sequence_landmarks = self.normalize_sequence_length(sequence_landmarks, self.max_seq_len)
        #print(sequence_landmarks.shape)
        return sequence_landmarks


    def __len__(self):
        return len(self.data)
    

    def get_delta(self, landmarks):
        delta_moving = np.mean(landmarks[1:, self.moving_lands, :3] - landmarks[:-1, self.moving_lands, :3], axis=(1, 2))
        delta_static = np.mean(landmarks[1:, self.static_lands, :3] - landmarks[:-1, self.static_lands, :3], axis=(1, 2))
        delta_global = np.mean(landmarks[1:, :, :3] - landmarks[:-1, :, :3], axis=(1, 2))
        
        delta = self.lambda_moving * delta_moving + self.lambda_static * delta_static + self.lambda_global * delta_global
        delta = np.concatenate(([0], delta))
        
        return delta


    def normalize_sequence_length(self, sequence, max_length):
        """
        """
        if len(sequence) > max_length:
            delta = self.get_delta(sequence)
            norm_sequence = sequence[top_k(delta, max_length)][0]
            
        elif len(sequence) < max_length:
            
            #norm_sequence = self.upsample(sequence, max_length)
            norm_sequence = interpolate_landmarks(sequence, max_length)
        else:
            norm_sequence = sequence
        
        return norm_sequence
        
    
    def __getitem__(self, index):
        
        ## get files paths
        landmarks, label = self.data[index]

        ## covert data to tensors
        landmarks = torch.from_numpy(landmarks).type(torch.float32)
        label = torch.tensor(label).type(torch.long)

        return landmarks, label