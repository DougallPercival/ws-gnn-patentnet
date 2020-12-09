# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 19:17:27 2020

@author: Paul Rev Oh
"""

import os
from utils import construct_graph_data

path = os.getcwd()
code_dir = os.path.join(path, 'src/')
data_dir = os.path.join(path, 'data/')

import torch
from torch_geometric.nn import SGConv
import torch.nn.functional as F

data, _, _ = construct_graph_data()

# num_classes = len(data.y.unique())

# conv = SGConv(in_channels=data.num_features, out_channels=num_classes,
#        K=1, cached=True)

# x  = data.x
# print("Shape before applying convoluton: ", x.shape)

# #x contains the node features, and edge_index encodes the structure of the graph
# x  = conv(x.float(), data.edge_index)
# print("Shape after applying convoluton: ", x.shape)

class SGNet(torch.nn.Module):
    def __init__(self, data, K=1):
        super().__init__()
        num_classes = len(data.y.unique())

        # Create a Simple convolutional layer with K neighbourhood 
        # "averaging" steps
        self.conv = SGConv(in_channels=data.num_features,
                            out_channels=64, 
                           K=K, cached=True)

        self.conv2 = SGConv(in_channels=64, 
                            out_channels=6, 
                            K=K)

    def forward(self, data):
        # Apply convolution to node features
        # F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv(data.x.float(), data.edge_index) # n,64
        x = F.relu(x)
        x = self.conv2(x, data.edge_index) # n,6

        # Compute log softmax.
        # Note: Negative log likelihood loss expects a log probability
        return F.log_softmax(x, dim=1) 
    
def train(model, data, optimizer):
    # Set the model.training attribute to True
    model.train() 
    
    # Reset the gradients of all the variables in a model
    optimizer.zero_grad() 
    
    # Get the output of the network. The output is a log probability of each
    log_softmax = model(data) 
    
    labels = data.y # Labels of each node
    
    # Use only the nodes specified by the train_mask to compute the loss.
    nll_loss = F.nll_loss(log_softmax[data.train_mask], labels[data.train_mask])
    
    #Computes the gradients of all model parameters used to compute the nll_loss
    #Note: These can be listed by looking at model.parameters()
    nll_loss.backward()
    
    # Finally, the optimizer looks at the gradients of the parameters 
    # and updates the parameters with the goal of minimizing the loss.
    optimizer.step() 
  

def compute_accuracy(model, data, mask):
    # Set the model.training attribute to False
    model.eval()
    logprob = model(data)
    _, y_pred = logprob[mask].max(dim=1)
    y_true=data.y[mask]
    acc = y_pred.eq(y_true).sum()/ mask.sum().float()
    return acc.item()

@torch.no_grad() # Decorator to deactivate autograd functionality  
def test(model, data):
    acc_train = compute_accuracy(model, data, data.train_mask)
    acc_val = compute_accuracy(model, data, data.val_mask)
    
    return acc_train, acc_val

# Create a model for the our dataset # CHECK THIS
model = SGNet(data, K=1)

# Create an Adam optimizer with learning rate and weight decay (i.e. L2 regularization)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.0)

for epoch in range(1, 5000):
    train(model, data, optimizer)
    if epoch %3 ==0:
      log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}'
      print(log.format(epoch, *test(model,data)))

model_dir = os.path.join(path, "model")
torch.save(model.state_dict(), os.path.join(model_dir, 'tfidf_conv_sd.pt'))
torch.save(optimizer.state_dict(), os.path.join(model_dir, 'tfidf_conv_opt.pt'))