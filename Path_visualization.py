import argparse
import torch
import torch.nn as nn
import numpy as np
import os
# from MLP import MLP 
import pickle
# from data_loader import load_test_dataset, data_loader
from torch.autograd import Variable 
import math
import time
import pandas as pd
import pickle

with open(r"Scalers_and_weights\scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
    
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
    
import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable

class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self.fc = nn.Sequential(
		nn.Linear(input_size, 1280),nn.PReLU(),nn.Dropout(),
		nn.Linear(1280, 1024),nn.PReLU(),nn.Dropout(),
		nn.Linear(1024, 896),nn.PReLU(),nn.Dropout(),
		nn.Linear(896, 768),nn.PReLU(),nn.Dropout(),
		nn.Linear(768, 512),nn.PReLU(),nn.Dropout(),
		nn.Linear(512, 384),nn.PReLU(),nn.Dropout(),
		nn.Linear(384, 256),nn.PReLU(), nn.Dropout(),
		nn.Linear(256, 256),nn.PReLU(), nn.Dropout(),
		nn.Linear(256, 128),nn.PReLU(), nn.Dropout(),
		nn.Linear(128, 64),nn.PReLU(), nn.Dropout(),
		nn.Linear(64, 32),nn.PReLU(),
		nn.Linear(32, output_size))
        


    def forward(self, x):
        out = self.fc(x)
        return out
    
def distance(node, point):
        dist = np.sqrt((node[0] - point[0])**2 + (node[1] - point[1])**2)         
        return dist


def goalFound(point , goal):
        if distance(goal, point) <= 5:
            return True
        return False
    
def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.to(torch.device("cuda"))
    return x

import cv2
# from data_loader import Autoencoder
# from MLP import MLP
from torchvision import datasets, transforms
import torch
import numpy

# Encoding the image
autoencoder = Autoencoder()
Q = autoencoder.encoder
Q.load_state_dict(torch.load(r"Scalers_and_weights\model_982.pkl"))
if torch.cuda.is_available():
    Q = Q.to(torch.device("cuda"))
    
# Load trained model for path generation
mlp = MLP(32, 2)
mlp.load_state_dict(torch.load(r"Scalers_and_weights\model_190.pkl"))

if torch.cuda.is_available():
    mlp.cuda()

# Load test dataset
image_path = r"Datasets\images\152.jpg"
path = []

image = cv2.imread(image_path ,  cv2.IMREAD_GRAYSCALE)

encoded_w_m=np.zeros((1,28),dtype=np.float32)
to_tensor = transforms.ToTensor()
torch_img = to_tensor(image)


if torch.cuda.is_available():
    torch_img = torch_img.to(torch.device("cuda")) 

output=Q(torch_img)
output = output.squeeze()
output=output.data.cpu()
output = output.numpy()
encoded_w_m[0] = output

start_x = 20
start_y = 100
start = np.array([[start_x, start_y]])
start = scaler.transform(start)
goal_x = 40
goal_y = 30
goal = np.array([[goal_x, goal_y]])
goal = scaler.transform(goal)
# start = np.array([start_x, start_y])
# goal = np.array([goal_x, goal_y])
path.append(torch.tensor(np.array([start[0][0], start[0][1]])))
# path.append(start)


start1=torch.from_numpy(start)
goal1=torch.from_numpy(goal)
obs = torch.from_numpy(encoded_w_m)

start1 = start1.float()
goal1 = goal1.float()

start1 = start1.squeeze()
goal1 = goal1.squeeze()
obs = obs.squeeze()

# print(start1.shape)
# print(goal1.shape)
# print(obs.shape)
# data = torch.cat((obs,start1,goal1))
# print(data.shape)
# Implement the Network to generate the path
while True:
    data = torch.cat((obs,start1,goal1))
    data = to_var(data)
    current = mlp(data)
    current = current.data.cpu()
    #print(current.numpy())
    path.append(current)
    start1 = current
#     print(current)
    current = current.numpy()
    current = np.array([[current[0], current[1]]])
    current = scaler.inverse_transform(current)
#     print("current",current)
    if goalFound(numpy.array([current[0][0], current[0][1]]) , numpy.array([goal_x, goal_y])):
        path.append(goal1)
        break
path.append(goal1)

path_modified = []
for ele in (path):
    if not isinstance(ele, np.ndarray):
        ele = ele.numpy()
        ele = ele.tolist()
    else :
        ele = ele.tolist()
    path_modified.append(ele)

# print(path_modified)

path_modified = np.array(path_modified)
path_modified = scaler.inverse_transform(path_modified)

import matplotlib.pyplot as plt
import cv2
image = cv2.imread(r"Datasets\images\152.jpg" , cv2.IMREAD_GRAYSCALE)

plt.figure()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Image with Points')
# plt.scatter(path_modified[:, 0], path_modified[:, 1], color='blue', marker='o' , s = 7)
for i in range(len(path_modified)-1):
    plt.imshow(image , cmap = 'gray')
    print("Waypoint: ", path_modified[i][0], path_modified[i][1])
    plt.plot([path_modified[i][0], path_modified[i+1][0]], [path_modified[i][1], path_modified[i+1][1]],'ro', linestyle="--")
    plt.pause(0.01)