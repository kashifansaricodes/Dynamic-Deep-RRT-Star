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

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()        
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), 
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), 
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64 , 128 , 3 , stride = 2 , padding = 1),
            nn.ReLU(),
            nn.Conv2d(128 , 256 , 8)
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 8),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

def data_loader(N = 200, NP = 200):
    
    
    autoencoder = Autoencoder()
    Q = autoencoder.encoder
    Q.load_state_dict(torch.load(r"C:\Users\Navdeep\Downloads\cae_encoder.pkl"))
    if torch.cuda.is_available():
        Q = Q.to(torch.device("cuda"))

    encoded_w_m=np.zeros((N,256),dtype=np.float32)
    for image_number in range(0,N):
#         print(image_number)
        image= cv2.imread(f'C:/Users/Navdeep/Downloads/images/images/{image_number}.jpg' , cv2.IMREAD_GRAYSCALE)
        to_tensor = transforms.ToTensor()
        torch_img = to_tensor(image)
        if torch.cuda.is_available():
            torch_img = torch_img.to(torch.device("cuda"))  # Move input tensor to GPU

        output=Q(torch_img)
        output = output.squeeze()
        output=output.data.cpu()
        output = output.numpy()
        encoded_w_m[image_number] = output
    
    
    
    example = pd.read_csv(r"C:\Users\Navdeep\Downloads\output_modified3.csv")
    #print(len(example))
    index1 = 0
    index2 = 0
    index3 = 1
    path_lengths=np.zeros((N,NP + 1),dtype=np.float32)
    prev_image_number = 189.0
    max_length=0
    while index1 < len(example) :
        row = example.iloc[index1]
        index_start = row['index_start']
        index_goal = row['index_goal']
        len_path = (index_goal - index_start) + 1
        image_number = row['image_number']
        index1 = index_goal + 1
        index1 = int(index1)
        if (image_number == prev_image_number):
            path_lengths[index2][0] = image_number
            path_lengths[index2][index3] = len_path
            index3 = index3 + 1
        else :
            index3 = 1
            index2 = index2 + 1
            path_lengths[index2][0] = image_number
            path_lengths[index2][index3] = len_path
            prev_image_number = image_number
            index3 = index3 + 1    
        if len_path > max_length :
            max_length = len_path
            
    
    max_length = int(max_length)
    
    
    path_lengths = path_lengths[:, 1:]
    
    
    
    paths=np.zeros((N,NP,max_length,2), dtype=np.float32)
    index = 0
    index1 = 0
    index2 = 0
    index3 = 0
    prev_image_number = 189.0
    while(index < len(example)) :
        row = example.iloc[index]
        index_start = int(row['index_start'])
        index_goal = int(row['index_goal'])
        image_number = row['image_number']
        index = index_goal + 1
        if (image_number == prev_image_number):
            index3 = 0
            for i in range(index_start , index_goal + 1):
                row = example.iloc[i]
                x = row['current_x']
                y = row['current_y']
                #print([x , y])
                paths[index1][index2][index3] = [x , y]
                #paths[index1][index2][index3][1] = y
                index3 = index3 + 1
            index2 = index2 + 1  
        else :
            index1 = index1 + 1
            index2 = 0
            index3 = 0
            for i in range(index_start , index_goal + 1):
                row = example.iloc[i]
                x = row['current_x']
                y = row['current_y']
                #print([x , y])
                paths[index1][index2][index3] = [x , y]
                #paths[index1][index2][index3][1] = y
                index3 = index3 + 1
            prev_image_number = image_number    
            index2 = index2 + 1
    
    
    
    
    dataset=[]
    targets=[]
    for i in range(200):
        for j in range(200):
            if path_lengths[i][j] > 0 :
                for m in range(0, int(path_lengths[i][j]-1)):
                    data=np.zeros(260,dtype=np.float32)
                    for k in range(0 , 256):
                        data[k] = encoded_w_m[i][k]
                    data[256] = paths[i][j][m][0]
                    data[257] = paths[i][j][m][1]
                    data[258] = paths[i][j][int(path_lengths[i][j]) - 1][0]
                    data[259] = paths[i][j][int(path_lengths[i][j]) - 1][1]
                    
                    targets.append(paths[i][j][m+1])
                    dataset.append(data)
                    
    #data = list(zip(dataset, targets))
    #random.shuffle(data)
#    dataset,targets=zip(*data)
    
    return np.asarray(dataset),np.asarray(targets)
    
    
    
dataset , targets = data_loader()


# print(max_length)

