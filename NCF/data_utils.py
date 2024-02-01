import random
import numpy as np
import pandas as pd
import torch

class NCF():
    def __init__(self,args,ratings):
        self.ratings = ratings
        self.num_neg_train = args.num_neg_train
        self.num_neg_test = args.num_neg_test
        self.batch_size = args.batch_size

        self.preprocess_ratings = self.preprocess(self.ratings)

        self.users = set(self.ratings['user_id'].unique())
        self.items = set(self.ratings['item_id'].unique())

        print(f'Total Users : {len(self.users)}')
        print(f'Total Items : {len(self.items)}')

        self.train_ratings , self.test_ratings = self.leave_one_out(self.preprocess_ratings)
        self.negatives = self.negative_sampling(self.preprocess_ratings)

        random.seed(args.seed)
    
    def preprocess(self,ratings):
        users = list(ratings['user_id'].drop_duplicates())
        user2id = {w:i for i,w in enumerate(users)}

        items = list(ratings['item_id'].drop_duplicates())
        item2id = {w:i for i,w in enumerate(items)}

        ratings['user_id'] = ratings['user_id'].apply(lambda x : user2id[x])
        ratings['item_id'] = ratings['item_id'].apply(lambda x : item2id[x])
        ratings['rating'] = ratings['rating'].apply(lambda x : float(x>0))
        return ratings

    def leave_one_out(self,ratings):
        ratings['rank_latest'] = ratings.groupby(['user_id'])['timestamp'].rank(method='first',ascending=False)
        test = ratings.loc[ratings['rank_latest']==1]
        train = ratings.loc[ratings['rank_latest']>1]
        assert train['user_id'].nunique()==test['user_id'].nunique(), 'Not Match Train User with Test User'
        return train[['user_id', 'item_id', 'rating']], test[['user_id', 'item_id', 'rating']]

    def negative_sampling(self,ratings):
        interact_status = (ratings.groupby('user_id')['item_id'].apply(set).reset_index().rename(columns={'item_id':'interacted_items'}))
        interact_status['negative_items'] = interact_status['interacted_items'].apply(lambda x:self.items-x)
        interact_status['negative_samples'] = interact_status['negative_items'].apply(lambda x: random.sample(x,self.num_neg_test))
        return interact_status[['user_id','negative_items','negative_samples']]
    
    def get_train_data(self):
        users,items,ratings = [], [], []
        train_ratings = pd.merge(self.train_ratings,self.negatives[['user_id','negative_items']] , on='user_id')
        train_ratings['negatives'] = train_ratings['negative_items'].apply(lambda x: random.sample(x,self.num_neg_train))
        for row in train_ratings.itertuples():
            users.append(int(row.user_id))
            items.append(int(row.item_id))
            ratings.append(float(row.rating))
            for i in range(self.num_neg_train):
                users.append(int(row.user_id))
                items.append(int(row.negatives[i]))
                ratings.append(float(0))
        dataset = Rating_Dataset(user_list=users,item_list=items,rating_list=ratings)
        return torch.utils.data.DataLoader(dataset,batch_size=self.batch_size,shuffle=True , num_workers=4)
    
    def get_test_data(self):
        users,items,ratings = [], [], []
        test_ratings = pd.merge(self.test_ratings,self.negatives[['user_id','negative_samples']] , on='user_id')
        for row in test_ratings.itertuples():
            users.append(int(row.user_id))
            items.append(int(row.item_id))
            ratings.append(float(row.rating))
            for i in getattr(row,'negative_samples'):
                users.append(int(row.user_id))
                items.append(int(i))
                ratings.append(float(0))
        dataset = Rating_Dataset(user_list=users,item_list=items,rating_list=ratings)
        return torch.utils.data.DataLoader(dataset,batch_size=self.num_neg_test+1,shuffle=False , num_workers=4)

class Rating_Dataset(torch.utils.data.Dataset):
    def __init__(self,user_list , item_list , rating_list):
        super(Rating_Dataset,self).__init__()
        self.user_list = user_list
        self.item_list = item_list
        self.rating_list = rating_list
    
    def __len__(self):
        return len(self.user_list)
    
    def __getitem__(self,idx):
        user = self.user_list[idx]
        item = self.item_list[idx]
        rating = self.rating_list[idx]
        return (torch.tensor(user,dtype=torch.long),torch.tensor(item,dtype=torch.long),torch.tensor(rating,dtype=torch.float))

