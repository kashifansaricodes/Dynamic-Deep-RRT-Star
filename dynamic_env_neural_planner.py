import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from torch.autograd import Variable 
import math
import time
import pandas as pd
import pickle
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import cv2
import torch
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
import time

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
    
# Load the scaler object from the saved file
with open(r"Scalers_and_weights\scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

start_x , start_y = 80 , 85

# path_list = [(current_x , current_y)]
goal_x , goal_y = 60 , 10

# index = 0
image_path = r"Datasets\images\60.jpg"
image = cv2.imread(image_path ,  cv2.IMREAD_GRAYSCALE)
path = []

start = np.array([[start_x, start_y]])
start = scaler.transform(start)

goal = np.array([[goal_x, goal_y]])
goal = scaler.transform(goal)

path.append(torch.tensor(np.array([start[0][0], start[0][1]])))

start1=torch.from_numpy(start)
goal1=torch.from_numpy(goal)

start1 = start1.float()
goal1 = goal1.float()

start1 = start1.squeeze()
goal1 = goal1.squeeze()

plt.figure()
# plt.imshow(image , cmap = 'gray')
plt.plot(start_x,start_y,'ro')
plt.plot(goal_x,goal_y,'bo')
# ax = plt.gca()
#     ax.add_patch(goalRegion)
plt.xlabel('X-axis $(m)$')
plt.ylabel('Y-axis $(m)$')   
time_start = time.time()
while True:
    plt.imshow(image , cmap = 'gray')
    encoded_w_m=np.zeros((1,28),dtype=np.float32)
    to_tensor = transforms.ToTensor()
    torch_img = to_tensor(image)
    if torch.cuda.is_available():
        torch_img = torch_img.to(torch.device("cuda")) 

    output=Q(torch_img)
    output = output.squeeze()
    output= output.data.cpu()
    output = output.numpy()
    encoded_w_m[0] = output

    obs = torch.from_numpy(encoded_w_m)
    obs = obs.squeeze()
    
    # temp = torch.from_numpy(start1)
    # temp = temp.squeeze()

    temp = start1

    data = torch.cat((obs,start1,goal1))

    # Implement the Network to generate the path 
    data = to_var(data)
    current = mlp(data)
    current = current.data.cpu()
    path.append(current)
    start1 = current
    x = current[0]
    y = current[1]

    unscaled_xy = current.numpy()
    unscaled_xy = np.array([[unscaled_xy[0], unscaled_xy[1]]])
    unscaled_xy = scaler.inverse_transform(unscaled_xy)

    temp = temp.numpy()
    temp = np.array([[temp[0], temp[1]]])
    temp = scaler.inverse_transform(temp)
    #path_list.append((x , y))
    plt.plot([unscaled_xy[0][0], temp[0][0]], [unscaled_xy[0][1] , temp[0][1]],'go', linestyle="--")  
    plt.pause(1)
    folder = r"Datasets\images"
    for filename in os.listdir(folder):
        image_path = os.path.join(folder, filename)
        image = cv2.imread(image_path , cv2.IMREAD_GRAYSCALE)
        pixel_value = image[int(unscaled_xy[0][1]) , int(unscaled_xy[0][0])]
        # print(pixel_value)
        if (pixel_value != 0):
            break
    
    checkcurrent = current.numpy()
    checkcurrent = np.array([[checkcurrent[0], checkcurrent[1]]])
    checkcurrent = scaler.inverse_transform(checkcurrent)
    # print("current",current)
    if goalFound(np.array([checkcurrent[0][0], checkcurrent[0][1]]) , np.array([goal_x, goal_y])):
        path.append(goal1)
        print(path)
        break

time_end = time.time()
time_taken = time_end - time_start
plt.title(f'Map after scanning, Run time: {time_taken} seconds')
print(time_taken)
plt.imshow(image , cmap = 'gray')
plt.pause(5)