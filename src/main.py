import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import progressbar
import time
from tqdm import tqdm
!pip install pytorch-model-summary
from pytorch_model_summary import summary
import os
import copy
from dataloader import *
from learner import *
from models import *

if torch.cuda.is_available():
  device = 'cuda:0'
else:
  device = 'cpu'

from torchvision import models

resnet_50 = models.resnet50(pretrained=True)
for param in resnet_50.parameters():
    param.requires_grad = True
#print(summary(resnet_50, torch.zeros((5, 3, 64, 64)), show_input=False))

print(torch.cuda.is_available())
model = Attn_Graph()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr = .0003)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
Learner_obj = Learner(datasets,model,criterion,optimizer,bs=50,num_workers=4,device=device)
epochs = 80
Learner_obj.fit(epochs=epochs)