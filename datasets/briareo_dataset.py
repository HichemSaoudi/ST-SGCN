
import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from utils.data_utils import top_k, interpolate_landmarks, upsample, compute_motion_features


class BriareoDataset(Dataset):
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

        files = []
        for ele in np.load(self.annotations_file, allow_pickle=True).values():
            for d in ele:
                seq = d['data']
                seq_path = os.path.join(*seq[0].split('/')[:-2])
                files.append((os.path.join(self.data_dir, seq_path), d['label']))
            
        data = []
        for seq_path, label in tqdm(files, desc='loading landamrks....'):
            file_path = os.path.join(seq_path, os.listdir(seq_path)[0])
            landmarks = self.load_landmarks(file_path)
            if landmarks is not None:
                data.append((landmarks, label))
        return data
    

    def load_landmarks(self, txt_file):
        sequence_landmarks = []
        with open(txt_file) as f:
            data = f.read()
            
        sequence = data.split('\n\n')
        for frame in sequence:
            lines = frame.split('\n')
            landmarks = []
            for e, line in enumerate(lines):
                
                if len(line) == 1:
                    coords = [-1.0, -1.0, -1.0, -1.0]
                else:
                    coords = line.split(';')
                    coords = list(filter(lambda x: len(x), coords))
                    #coords = [self.num_connections[e]/3] + [float(x) for x in coords]
                #spher_coords = self.cartesian_to_polar(coords)
                landmarks.append(coords)
                
            if len(landmarks) < 2:
                continue
                #landmarks = [[-1.0, -1.0, -1.0, -1.0]] * 21
                
            landmarks = np.array(landmarks).astype(np.float32)
            #speed, accel = self.compute_motion_features(landmarks)
            #features = np.hstack((landmarks, speed, accel))
            sequence_landmarks.append(landmarks)
            
        if len(sequence_landmarks) > 1:
            sequence_landmarks = np.array(sequence_landmarks).astype(np.float32)
            sequence_landmarks = self.normalize_sequence_length(sequence_landmarks, self.max_seq_len)
            return sequence_landmarks
        return None
  

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