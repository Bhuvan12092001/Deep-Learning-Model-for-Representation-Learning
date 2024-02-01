import torch
import torch.nn as nn

class NeuMF(nn.Module):
    def __init__(self,args,num_users,num_items):
        super(NeuMF,self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embed_size_mf = args.factor_num
        self.embed_size_mlp = int(args.layers[0]/2)
        self.layers = args.layers
        self.dropout = args.dropout

        self.embedding_user_mlp = nn.Embedding(self.num_users,self.embed_size_mlp)
        self.embedding_item_mlp = nn.Embedding(self.num_items,self.embed_size_mlp)

        self.embedding_user_mf = nn.Embedding(self.num_users,self.embed_size_mf)
        self.embedding_item_mf = nn.Embedding(self.num_items,self.embed_size_mf)

        self.fc_layers = nn.ModuleList()
        for idx, (in_size, out_size) in enumerate(zip(args.layers[:-1], args.layers[1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))
            self.fc_layers.append(nn.ReLU())
        
        self.output = nn.Linear(in_features=args.layers[-1] + self.embed_size_mf, out_features=1)
        self.logistic = nn.Sigmoid()
        self.init_weight()
    
    def init_weight(self):
        nn.init.normal_(self.embedding_user_mlp.weight , std=0.01)
        nn.init.normal_(self.embedding_user_mf.weight , std=0.01)
        nn.init.normal_(self.embedding_item_mlp.weight , std=0.01)
        nn.init.normal_(self.embedding_item_mf.weight , std=0.01)

        for m in self.fc_layers:
            if isinstance(m,nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        
        nn.init.xavier_uniform_(self.output.weight)
        
        for m in self.modules():
            if isinstance(m,nn.Linear) and m.bias is not None:
                m.bias.data.zero_()
    
    def forward(self,user_indices,item_indices):
        user_embedding_mlp = self.embedding_user_mlp(user_indices)
        item_embedding_mlp = self.embedding_item_mlp(item_indices)

        user_embedding_mf = self.embedding_user_mf(user_indices)
        item_embedding_mf = self.embedding_item_mf(item_indices)

        mlp_input = torch.cat([user_embedding_mlp , item_embedding_mlp] , dim=-1)
        mf_input = torch.mul(user_embedding_mf,item_embedding_mf)

        for idx , _ in enumerate(range(len(self.fc_layers))):
            mlp_input = self.fc_layers[idx](mlp_input)
        
        vector = torch.cat([mlp_input , mf_input] , dim=-1)
        logits = self.output(vector)
        rating = self.logistic(logits)
        return rating.squeeze()
