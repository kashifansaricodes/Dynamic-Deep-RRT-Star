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
            nn.Conv2d(128 , 256 , 3 , stride = 2 , padding = 1),
            nn.ReLU(),
            nn.Conv2d(256 , 512 , 3 , stride = 2 , padding = 1),
            nn.ReLU(),
            nn.Conv2d(512 , 1024 , 2),
            nn.ReLU(),
            nn.Conv2d(1024 , 2048 , 1),
            nn.ReLU(),
            nn.Conv2d(2048 , 28 , 1)
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(28, 2048, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(2048, 1024, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 512, 2),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256,  3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
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

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    
def data_loader(environment_list , N = 101, NP = 201):
    
    dtypes = {col: 'float32' for col in pd.read_csv(file_path, nrows=1).columns}
    example = pd.read_csv(file_path, dtype=dtypes)

    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X = example.iloc[:, :2].values
    X = sc.fit_transform(X)
    
    autoencoder = Autoencoder()
    Q = autoencoder.encoder
    Q.load_state_dict(torch.load(r"/home/sj/Deep-RRT-Star-Implementation/weights/cae_encoder2.pkl"))
    if torch.cuda.is_available():
        Q = Q.to(torch.device("cuda"))

    encoded_w_m=np.zeros((N,28),dtype=np.float32)
    index = 0
    for env in environment_list :
        env = int(env)
#         print('env')
#         print(env)
#         print(image_number)
        image= cv2.imread(f'/home/sj/Deep-RRT-Star-Implementation/images/{env}.jpg' , cv2.IMREAD_GRAYSCALE)
        to_tensor = transforms.ToTensor()
        torch_img = to_tensor(image)
        if torch.cuda.is_available():
            torch_img = torch_img.to(torch.device("cuda"))  # Move input tensor to GPU

        output=Q(torch_img)
        output = output.squeeze()
        output=output.data.cpu()
        output = output.numpy()
        encoded_w_m[index] = output
        index = index + 1
    
    
    file_path = r"/home/sj/Deep-RRT-Star-Implementation/output.csv"
    dtypes = {col: 'float32' for col in pd.read_csv(file_path, nrows=1).columns}
    example = pd.read_csv(file_path, dtype=dtypes)
    #print(len(example))
    index1 = 0
    index2 = 0
    index3 = 1
    path_lengths=np.zeros((N,NP + 1),dtype=np.float32)
    prev_image_number = 0
    max_length=0
    while index1 < len(example) :
        row = example.iloc[index1]
        index_start = row['index_start']
        index_goal = row['index_goal']
        len_path = (index_goal - index_start) + 1
        image_number = row['image_number']
        index1 = index_goal + 1
        index1 = int(index1)
#         print('image number')
#         print(image_number)
        if (image_number == prev_image_number):
            path_lengths[index2][0] = image_number
            path_lengths[index2][index3] = len_path
            index3 = index3 + 1
        else :
#             print('image number')
#             print(prev_image_number)
#             print(index3 - 1)
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
    prev_image_number = 0
    while(index < len(example)) :
        row = example.iloc[index]
        # index_start = int(row['index_start'])
        # index_goal = int(row['index_goal'])
        index_start = X[index][0]
        image_number = X[index][1]
        index = index_goal + 1
        if (image_number == prev_image_number):
            index3 = 0
            for i in range(index_start , index_goal + 1):
                row = example.iloc[i]
                # x = row['current_x']
                # y = row['current_y']
                x = X[i][0]
                y = X[i][1]
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
                # x = row['current_x']
                # y = row['current_y']
                x = X[i][0]
                y = X[i][1]
                #print([x , y])
                paths[index1][index2][index3] = [x , y]
                #paths[index1][index2][index3][1] = y
                index3 = index3 + 1
            prev_image_number = image_number    
            index2 = index2 + 1
    
    
    
    
    dataset=[]
    targets=[]
    for i in range(N):
        for j in range(NP):
            if path_lengths[i][j] > 0 :
                for m in range(0, int(path_lengths[i][j]-1)):
                    data=np.zeros(32,dtype=np.float32)
                    for k in range(0 , 28):
                        data[k] = encoded_w_m[i][k]
                    data[28] = paths[i][j][m][0]
                    data[29] = paths[i][j][m][1]
                    data[30] = paths[i][j][int(path_lengths[i][j]) - 1][0]
                    data[31] = paths[i][j][int(path_lengths[i][j]) - 1][1]
                    
                    temp = [round(paths[i][j][m+1][0] , 6) , round(paths[i][j][m+1][1] , 6)]
                    temp = np.array(temp)
                    targets.append(temp)
                    dataset.append(data)
                    
    data = list(zip(dataset, targets))
    random.shuffle(data)
    dataset,targets=zip(*data)
    
    return np.asarray(dataset),np.asarray(targets)
    

def load_test_dataset(environment_list , N = 101, NP = 201):

    dtypes = {col: 'float32' for col in pd.read_csv(file_path, nrows=1).columns}
    example = pd.read_csv(file_path, dtype=dtypes)

    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X = example.iloc[:, :2].values
    X = sc.fit_transform(X)
    
    autoencoder = Autoencoder()
    Q = autoencoder.encoder
    Q.load_state_dict(torch.load(r"/home/sj/Deep-RRT-Star-Implementation/weights/cae_encoder2.pkl"))
    if torch.cuda.is_available():
        Q = Q.to(torch.device("cuda"))

    encoded_w_m=np.zeros((N,28),dtype=np.float32)
    index = 0
    for env in environment_list :
        env = int(env)
#         print('env')
#         print(env)
#         print(image_number)
        image= cv2.imread(f'/home/sj/Deep-RRT-Star-Implementation/images/{env}.jpg' , cv2.IMREAD_GRAYSCALE)
        to_tensor = transforms.ToTensor()
        torch_img = to_tensor(image)
        if torch.cuda.is_available():
            torch_img = torch_img.to(torch.device("cuda"))  # Move input tensor to GPU

        output=Q(torch_img)
        output = output.squeeze()
        output=output.data.cpu()
        output = output.numpy()
        encoded_w_m[index] = output
        index = index + 1
    
    
    file_path = r"/home/sj/Deep-RRT-Star-Implementation/output.csv"
    dtypes = {col: 'float32' for col in pd.read_csv(file_path, nrows=1).columns}
    example = pd.read_csv(file_path, dtype=dtypes)
    #print(len(example))
    index1 = 0
    index2 = 0
    index3 = 1
    path_lengths=np.zeros((N,NP + 1),dtype=np.float32)
    prev_image_number = 0
    max_length=0
    while index1 < len(example) :
        row = example.iloc[index1]
        index_start = row['index_start']
        index_goal = row['index_goal']
        len_path = (index_goal - index_start) + 1
        image_number = row['image_number']
        index1 = index_goal + 1
        index1 = int(index1)
#         print('image number')
#         print(image_number)
        if (image_number == prev_image_number):
            path_lengths[index2][0] = image_number
            path_lengths[index2][index3] = len_path
            index3 = index3 + 1
        else :
#             print('image number')
#             print(prev_image_number)
#             print(index3 - 1)
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
    prev_image_number = 0
    while(index < len(example)) :
        row = example.iloc[index]
        # index_start = int(row['index_start'])
        # index_goal = int(row['index_goal'])
        index_start = X[index][0]
        image_number = X[index][1]
        index = index_goal + 1
        if (image_number == prev_image_number):
            index3 = 0
            for i in range(index_start , index_goal + 1):
                row = example.iloc[i]
                # x = row['current_x']
                # y = row['current_y']
                x = X[i][0]
                y = X[i][1]
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
                # x = row['current_x']
                # y = row['current_y']
                x = X[i][0]
                y = X[i][1]
                #print([x , y])
                paths[index1][index2][index3] = [x , y]
                #paths[index1][index2][index3][1] = y
                index3 = index3 + 1
            prev_image_number = image_number    
            index2 = index2 + 1
    
    
    
    
    dataset=[]
    targets=[]
    for i in range(N):
        for j in range(NP):
            if path_lengths[i][j] > 0 :
                for m in range(0, int(path_lengths[i][j]-1)):
                    data=np.zeros(32,dtype=np.float32)
                    for k in range(0 , 28):
                        data[k] = encoded_w_m[i][k]
                    data[28] = paths[i][j][m][0]
                    data[29] = paths[i][j][m][1]
                    data[30] = paths[i][j][int(path_lengths[i][j]) - 1][0]
                    data[31] = paths[i][j][int(path_lengths[i][j]) - 1][1]
                    
                    temp = [round(paths[i][j][m+1][0] , 6) , round(paths[i][j][m+1][1] , 6)]
                    temp = np.array(temp)
                    targets.append(temp)
                    dataset.append(data)
                    
    data = list(zip(dataset, targets))
    random.shuffle(data)
    dataset,targets=zip(*data)
    
    return np.asarray(dataset),np.asarray(targets)

# print(max_length)