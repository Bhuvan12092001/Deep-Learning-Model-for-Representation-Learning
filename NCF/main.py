import os
import time
import argparse
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

import data_utils
import model
import utils
import evalute
from tqdm import tqdm 

parser = argparse.ArgumentParser()
parser.add_argument("--seed", 
	type=int, 
	default=42, 
	help="Seed")
parser.add_argument("--lr", 
	type=float, 
	default=0.001, 
	help="learning rate")
parser.add_argument("--dropout", 
	type=float,
	default=0.2,  
	help="dropout rate")
parser.add_argument("--batch_size", 
	type=int, 
	default=256, 
	help="batch size for training")
parser.add_argument("--num_epochs", 
	type=int,
	default=30,  
	help="training epoches")
parser.add_argument("--top_k", 
	type=int, 
	default=10, 
	help="compute metrics@top_k")
parser.add_argument("--factor_num", 
	type=int,
	default=32, 
	help="predictive factors numbers in the model")
parser.add_argument("--layers",
    nargs='+', 
    default=[64,32,16,8],
    help="MLP layers. Note that the first layer is the concatenation of user and item embeddings. So layers[0]/2 is the embedding size.")
parser.add_argument("--num_neg_train", 
	type=int,
	default=4, 
	help="Number of negative samples for training set")
parser.add_argument("--num_neg_test", 
	type=int,
	default=100, 
	help="Number of negative samples for test set")
parser.add_argument("--out", 
	default=True,
	help="save model or not")

# set device and parameters
args = parser.parse_args()
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

# seed for Reproducibility
utils.seed_everything(args.seed)

# load data

print("\nLoading Data")
ml_1m = pd.read_csv(
	'data/ratings.dat', 
	sep="::", 
	names = ['user_id', 'item_id', 'rating', 'timestamp'], 
	engine='python')

# set the num_users, items
num_users = ml_1m['user_id'].nunique()+1
num_items = ml_1m['item_id'].nunique()+1

print("\nSeperating Training and Testing Data")
# construct the train and test datasets
data = data_utils.NCF(args, ml_1m)
train_loader =data.get_train_data()
test_loader =data.get_test_data()

print("\nData Preprocessing Done")


model = model.NeuMF(args,num_users , num_items)
model = model.to(device)

loss_function = nn.BCELoss()
optimizer = optim.Adam(model.parameters() , lr=args.lr)

print("\nTraining Period")
best_hr = 0
for epoch in range(1,args.num_epochs+1):
    model.train()
    start_time = time.time()
    for user,item,label in train_loader:
        user = user.to(device)
        item = item.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        prediction = model(user,item)
        loss = loss_function(prediction,label)
        loss.backward()
        optimizer.step()
    model.eval()
    HR , NDCG = evalute.metrics(model , test_loader , args.top_k , device)
    print(f'\nEpoch[{epoch}] - Loss : {loss.item()}  HR : {round(HR,3)}  NDCG : {round(NDCG,3)}')
    if(HR > best_hr):
        best_hr , best_ndcg , best_epoch = HR , NDCG , epoch
        torch.save(model,'models/Final_Model.pth')

print(f"\nEND . Best Epoch {best_epoch} HR : {round(best_hr,3)} NDCG : {round(best_ndcg,3)}")
        
        
    
    
        
