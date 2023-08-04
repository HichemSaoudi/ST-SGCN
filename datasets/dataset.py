
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from utils.data_utils import top_k, interpolate_landmarks, upsample, compute_motion_features



class array2tensor(object):
    """converts a numpy array to a torch tensor"""
        
    def __call__(self, array):
        
        ## numpy: H x W x C => torch: C x H x W
        if len(array.shape) > 3:
            array = array.transpose((0, 3, 1, 2)).astype(np.float32)
        else:
            array = array.transpose((2, 0, 1)).astype(np.float32)
        tensor = torch.from_numpy(array)
        return tensor


def tensor2array(tensor):
    """converts a torch tensor to a numpy array"""
        
    array = tensor.detach().cpu().numpy()
    
    ## torch: C x H x W => numpy: H x W x C
    if len(array.shape) > 3:
        array = array.transpose((0, 2, 3, 1)).astype(np.float32)
    else:
        array = array.transpose((1, 2, 0)).astype(np.float32)
    return array


class LandsDataset(Dataset):
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
        self.data = self.get_sequences_ipn_paths()   
        
    
    def read_text_file(self, src_path):
        with open(src_path, 'r') as file:
            content = file.read()
        return content
    

    def get_sequences_ipn_paths(self):
        data = []
        with open(self.annotations_file, 'r') as file:
            lines = file.readlines()
        #print("lines",lines)
        for line in lines:
            #print('line before strip',line)
            line = line.strip()
            #print('line after strip',line)
            parts = line.split(',')
            folder_name = parts[0]
            folder_name = str(folder_name)
            label = int(parts[2])-1
            
            text_file_name = f"{parts[1]}_{parts[2]}_{parts[3]}_{parts[4]}_{parts[5]}"
        
            src_path = self.data_dir + "/" + folder_name + "/" + text_file_name + ".txt"
            
            if not os.path.exists(src_path):
                print("The file does not exist. Continuing...")
                continue
            
            landmarks = self.load_landmarks_ipn(src_path)
            if landmarks is not None:
                data.append((landmarks, label))
        return data
            

    def load_landmarks_ipn(self, txt_file):
        print(txt_file)
        sequence_landmarks = []
        with open(txt_file) as f:
            data = f.read()
            
        sequence = data.split('\n\n')
        sequence = sequence[:-1]
        
        for frame in sequence:
            lines = frame.split('\n')
            landmarks = []
            i=0
            for e, line in enumerate(lines):
                if len(line) == 1 :
                    coords = [-1.0, -1.0, -1.0, -1.0]
                else:
                    coords = line.split(';')
                    coords = list(filter(lambda x: len(x), coords))
                    coords = [float(x) for x in coords] + [self.connectivity[i]/3]
                #spher_coords = self.cartesian_to_polar(coords)
                landmarks.append(coords)
                i += 1
                
            #if len(landmarks) < 2:
             #   continue
                #landmarks = [[-1.0, -1.0, -1.0, -1.0]] * 21
                
            landmarks = np.array(landmarks).astype(np.float32)
            
            if len(frame) == 1 :
                landmarks = np.repeat(landmarks, 21, axis=0)
            #speed, accel = self.compute_motion_features(landmarks)
            #features = np.hstack((landmarks, speed, accel))
            sequence_landmarks.append(landmarks)
            
        if len(sequence_landmarks) > 1:
            sequence_landmarks = np.array(sequence_landmarks).astype(np.float32)
            print("landmarks",sequence_landmarks.shape)
            sequence_landmarks = self.normalize_sequence_length(sequence_landmarks, self.max_seq_len)
            print("sequence_landmarks",sequence_landmarks.shape)
            return sequence_landmarks
        return None
            

    def get_sequences_shrec_paths(self):
        
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
            landmarks = self.load_landmarks_shrec(src_path)
            label = int(g_id) - 1
            if landmarks is not None:
                #aug_landmarks = self.random_moving(landmarks)
                data.append((landmarks, label))
            #if (self.use_data_aug) and (landmarks is not None) and (torch.rand(1)>0.5) :
             #   landmarks = self.noise(landmarks)
             #   landmarks = self.random_moving(landmarks)
             #   data.append((landmarks, label))

        return data


    def load_landmarks_shrec(self, txt_file):
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


    def get_sequences_briareo_paths(self):

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