import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from utils.data_utils import top_k, interpolate_landmarks, upsample, compute_motion_features



class Shrec21Dataset(Dataset):
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
        self.static_lands = [0, 1, 4, 8, 12, 16]
        self.moving_lands = list(set(range(20)) - set(self.static_lands))
        
        self.lambda_moving = 10
        self.lambda_static = 3
        self.lambda_global = 1
        
        self.connectivity = connectivity
        #self.num_connections = {land:0 for land in range(21)}
        #for i, j in solutions.hands.HAND_CONNECTIONS:
        #    self.num_connections[i] += 1

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

        for line in seq :
            line = line.strip().split(';')
            seq_id = line[0]
            gest_combined = line[1:-1]
            src_path = self.annotations_file.rsplit('/', 1)[0] + "/sequences"
                
            for i in range(0, len(gest_combined), 3):
                ges = gest_combined[i:i+3]
                print(ges)
                file_path = os.path.join(src_path, f"{seq_id}.txt")
                with open(file_path, 'r') as f:
                    ges_seq = f.read()
                ges_seq = ges_seq.split("\n")
                ges_seq = ges_seq[int(ges[1])-1:int(ges[2])+1]
                label = self.get_label(ges[0])
                landmarks = self.load_landmarks(ges_seq)
                if landmarks is not None:
                    data.append((landmarks, label))

                    
        return data

  
    def get_label(self , gesture):
        for i,j in enumerate(self.shrec21_labels) : 
            if gesture == j :
                return i

    def load_landmarks(self, txt_file):
        sequence_landmarks = []
        
        for line in ges_seq:
            line = line.strip().split(';')
            line = line[:-1]
            point = []
            landmarks = []
            i=0
            for data_ele in line :
                point.append(float(data_ele))
                if len(point) == 7:   
                    point = [self.connectivity[i]/6] + [float(x) for x in point]
                    landmarks.append(point)
                    point = []
                    i+=1
            
            if len(landmarks) < 2 :
                continue
            landmarks = np.array(landmarks).astype(np.float32)
            #speed, accel = self.compute_motion_features(landmarks)
            #print(f"The shape of the landmarks is {landmarks.shape}")
            #print(f"The shape of the speed is {speed.shape}")
            #print(f"The shape of the accel is {accel.shape}")
            #features = np.hstack((landmarks, speed, accel))
            sequence_landmarks.append(landmarks)
            
        
        sequence_landmarks = np.array(sequence_landmarks).astype(np.float32)
        sequence_landmarks = self.normalize_sequence_length(sequence_landmarks, self.max_seq_len)
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
