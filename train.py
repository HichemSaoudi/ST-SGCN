
## libraries
import random
import torch
import numpy as np

import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from mediapipe import solutions

import math
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

import sys
import shutil
import os
from tqdm import tqdm
import glob

from utils.checkpoints import load_checkpoint, save_checkpoint
from utils.graph_utils import calculate_connectivity, get_sgcn_identity, hand_adj_matrix, edge
from datasets.ipn_dataset import IPNDataset
from datasets.briareo_dataset import BriareoDataset
from datasets.shrec17_dataset import ShrecDataset
from models.sgcn import SGCNModel


## seed
def seed_everything(seed=0):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Args:
    ...
    
    def __str__(self):
        return 'Args:\n\t> ' + f'\n\t> '.join([f'{key:.<20}: { val}' for key, val in self.__dict__.items()])
    
    __repr__ = __str__
    

args = Args()
args.seed = 1234
seed_everything(args.seed)

args.datasetname = 'Briareo' #Briareo,SHREC17,SHREC21,IPN


def train_epoch(epoch, num_epochs, model, optimizer, dataloader, criterion):
    model.train()

    pbar = tqdm(dataloader, total=len(dataloader))
    total_loss = 0

    for (V, y) in pbar:
        V = V.to(args.device)
        y = y.to(args.device)
        identity = get_sgcn_identity(V.shape, args.device)
        optimizer.zero_grad()
        pred, _, pred_spatial_adj, _ = model(V, identity)
        loss_train = criterion(pred, y)
        loss_train.backward()
        optimizer.step()
        total_loss += loss_train.item()
        pbar.set_description(f'[%.3g/%.3g] train loss. %.2f' % (epoch, num_epochs, total_loss/len(y)))
    scheduler.step()
    
    return total_loss
        

def validate_epoch(model, dataloader, criterion):
    acc = 0.0
    n = 0
    pred_labels, true_labels = [], []
    total_loss = 0.0
    model.eval()
    with torch.no_grad():
        pbar = tqdm(dataloader, total=len(dataloader))
        for (V, y) in pbar:
            V = V.to(args.device)
            y = y.to(args.device)
            true_labels.append(y[0].item())
            identity = get_sgcn_identity(V.shape, args.device)
            output, *_ = model(V, identity)
            loss_valid = criterion(output, y)
            acc += (output.argmax(dim=1) == y.flatten()).sum().item()
            n += len(y.flatten())
            total_loss += loss_valid.item()
            
            pred_labels.append(output.argmax(dim=1)[0].item())
            desc = '[VALID]> loss. %.2f > acc. %.2g%%' % (total_loss/len(y), (acc / n)*100)
            pbar.set_description(desc)
                
    return total_loss, true_labels, pred_labels


## device
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## paths
args.data_dir = "/home/hichemsaoudi/Downloads/Briareo_landmarks_splits/landmarks"
args.train_annot_file = "/home/hichemsaoudi/Downloads/Briareo_landmarks_splits/splits/train/depth_train.npz"
args.valid_annot_file = "/home/hichemsaoudi/Downloads/Briareo_landmarks_splits/splits/train/depth_val.npz"
args.test_annot_file  = "/home/hichemsaoudi/Downloads/Briareo_landmarks_splits/splits/test/depth_test.npz"

#edge = set(solutions.hands.HAND_CONNECTIONS)
connectivity = calculate_connectivity(hand_adj_matrix, edge)

## params
args.num_nodes = 21
args.max_seq_len = 60
args.labels_encoder = None



train_dataset = BriareoDataset(
                           args.data_dir,
                           args.train_annot_file,
                           args.max_seq_len,
                           connectivity,
                          )


valid_dataset = BriareoDataset(
                           args.data_dir,
                           args.valid_annot_file,
                           args.max_seq_len,
                           connectivity,
                          )

test_dataset = BriareoDataset(
                          args.data_dir,
                          args.test_annot_file,
                          args.max_seq_len,
                          connectivity,
                         )



args.num_classes = len(set([y for _, y in train_dataset.data]))

args.batch_size = 1

train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
test_dataloader  = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

print(f"#Train: {len(train_dataloader)} | #Valid: {len(valid_dataloader)} | #Test: {len(test_dataloader)}")


## training params
args.num_epochs = 200
args.pre_trained = None

## model parameters
args.num_features = 6
args.num_asymmetric_convs = 3
args.embedding_dims = 64
args.num_gcn_layers = 1
args.num_heads = 4
args.dropout = 0.5

## optimizer parameters
args.lr = 1e-3
args.weight_decay = 5e-2
args.T_max = 20

## save models path
args.save_path = "./weights/"
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path, exist_ok=True)


print(args)

## TRAIN 
sgcn = SGCNModel(args)
optimizer = optim.AdamW(sgcn.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.T_max, eta_min=0, last_epoch=-1, verbose=False)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

if args.pre_trained:
    sgcn, optimizer, scheduler, start_epoch, best_val = load_checkpoint(sgcn, optimizer, scheduler, args.pre_trained)

    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(args.device)

else:
    start_epoch = 1
    best_val = float("inf")

## to device
sgcn = nn.DataParallel(sgcn).to(args.device)


# Training loop
for epoch in range(start_epoch, args.num_epochs+1):
    train_loss = train_epoch(epoch, args.num_epochs, sgcn, optimizer, train_dataloader, criterion)
    valid_loss,true_labels, pred_labels = validate_epoch(sgcn, valid_dataloader, criterion)
    
    
    is_best = valid_loss < best_val
    best_val = min(valid_loss, best_val)
    save_checkpoint({'epoch': epoch,
                     'state_dict': sgcn.module.state_dict(),
                     'optimizer': optimizer.state_dict(),
                     'scheduler': scheduler.state_dict(),
                     'best_loss': best_val},
                      args.save_path, 'epoch_{}.pth'.format(epoch), is_best)


