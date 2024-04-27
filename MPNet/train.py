import torch
import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
from PIL import Image
import os.path
import random
from torch.autograd import Variable
import torch.nn as nn
import math
import cv2
from torchvision import datasets, transforms
import pandas as pd
import numpy as np
import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from torch.autograd import Variable 
import math
import csv
import MLP
from data_loader import dataset,targets

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.to(torch.device("cuda"))
    return x

def get_input(i,data,targets,bs):
    if i+bs<len(data):
        bi=data[i:i+bs]
        bt=targets[i:i+bs]
    else:
        bi=data[i:]
        bt=targets[i:]
    return torch.from_numpy(bi),torch.from_numpy(bt)

#dataset,targets= load_dataset()
#I have to yet write the architecture for the model
mlp = MLP(32, 2)

if torch.cuda.is_available():
    mlp = mlp.to(torch.device("cuda"))
    
criterion = nn.MSELoss()
optimizer = torch.optim.Adagrad(mlp.parameters())

total_loss=[]
sm=100 
#training the model for 1000 epochs
for epoch in range(1000):
    #print ("epoch" + str(epoch))
    avg_loss=0
    for i in range (0,len(dataset),100):
        mlp.zero_grad()
        bi,bt= get_input(i,dataset,targets,100)
        bi=to_var(bi)
        bt=to_var(bt)
        bo = mlp(bi)
        loss = criterion(bo,bt)
        #avg_loss=avg_loss+loss.data[0]
        loss.backward()
        optimizer.step()
        
    print(f'Epoch:{epoch+1}, Loss:{loss.item():.4f}')  
#     print "--average loss:"
#     print avg_loss/(len(dataset)/args.batch_size)
    total_loss.append(avg_loss/(len(dataset)/100))
    # Save the models
    if epoch==sm:
        model_path='model'+str(sm)+'.pkl'
        save_path = '/home/sj/Deep-RRT-Star-Implementation/weights/model_weights'
        torch.save(mlp.state_dict(),os.path.join(save_path , model_path))
        sm=sm+50 # save model after every 50 epochs from 100 epoch ownwards

            
#Following are the commands for saving the final model.            
model_path='/home/sj/Deep-RRT-Star-Implementation/weights/model_weights/final_model.pkl'
torch.save(mlp.state_dict() , model_path)
        
