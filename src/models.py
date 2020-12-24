import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from adj_mat import *

class Graph_Conv(nn.Module):
  def __init__(self,in_features,out_features):
    super(Graph_Conv, self).__init__()
    self.weight = Parameter(torch.FloatTensor(in_features, out_features))
    torch.nn.init.xavier_normal_(self.weight, gain=1.0)
#     torch.nn.init.normal_(self.weight, mean=0, std=1)

  def forward(self,x,A):      #x = b,d,nodes ; #A = b,nodes,nodes ; weight = d,d'
    #apply 1 graph convs(normal nn)
    # wei = torch.mm(x,self.weight)
    # lay = torch.mm(A,wei)     
    # lay = nn.functional.relu(A@(x.permute(0,2,1))@self.weight)
    lay = A@(x.permute(0,2,1))@self.weight
    lay = lay.permute(0,2,1)
    # if (self.weight.shape[0] == 256) & (self.weight.shape[1] == 256):
    #     print(self.weight)
    return lay  

class delft_block(nn.Module):
  def __init__(self,input_dimensions,k):
    super(delft_block,self).__init__()
    #self.conv1 = nn.Conv2d(input_dim, 512, kernel_size=kernel_size, padding=padding, stride=stride)
    #self.activate = nn.ReLU()
    self.k = k
    self.inp = input_dimensions
    #self.conv1 = nn.Conv2d(input_dimensions,512,kernel_size=3,padding=1)
    self.conv1 = nn.Conv2d(input_dimensions, 256, kernel_size=1, padding=0, stride=1)
    self.bn1 = nn.BatchNorm2d(256)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = nn.Conv2d(256,1, kernel_size=1, padding=0, stride=1)
    self.downsample = nn.Conv2d(input_dimensions, 1, kernel_size=1, padding=0, stride=1)
    self.bn = nn.BatchNorm2d(1)
  
  def forward(self,x):                                          #x = b,512,8,8
    b,c,h,w = x.size()
    residual = x
    out = self.conv1(x)                                         #out = b,1,8,8
    out = self.bn1(out)
    out = self.relu(out)
    out = self.conv2(out)
    residual = self.downsample(residual)
    resdiual = self.bn(residual)
    out += residual

    prob = nn.Softplus()(out)                                       
    x = x*prob                                                  #attention score multiplied by corresponding feature maps
    att = F.softmax(prob.view(b,-1),dim=1)                      #b,8*8(pixel wise attention scores flattened)
    #print(att.size())
    val,indices = torch.topk(att,self.k)                        # finding pixel indices with top k attention scores
    return x,indices

class mlp(nn.Module):
    def __init__(self,num_class):
        super(mlp,self).__init__()
        self.linear1 = nn.Linear(64,32)
        self.linear2 = nn.Linear(32,num_class)
        #self.linear3 = nn.Linear(32,num_class)
        #self.dropout = nn.Dropout(p=0.4)

    def forward(self,x):
        #x = self.dropout(F.relu(self.linear1(x)))
        #x = self.dropout(F.relu(self.linear2(x)))
        x = self.linear2(self.linear1(x))
        return x

class Attn_Graph(nn.Module):
    def __init__(self,num_class=10,k=120):
        super(Attn_Graph, self).__init__()
        
        self.k = k
        self.feature1 = nn.Sequential(*list(resnet_50.children())[:-5])
        # self.feature2 = nn.Sequential(*list(resnet_50.children())[:-5])
        # self.delft1 = delft_block(512,self.k)
        self.delft2 = delft_block(256,self.k)
        # self.projector = nn.Conv1d(in_channels = 256,out_channels=512,kernel_size = 1)
        #self.delft2 = delft_block(256,40)
        self.gcn1 = Graph_Conv(256,256)
        self.gcn2 = Graph_Conv(256,128)
        self.gcn3 = Graph_Conv(128,128)
        self.gcn4 = Graph_Conv(128,64)
        self.mlp = mlp(num_class)
        self.bn1 = nn.BatchNorm1d(num_features = 256)
        self.bn2 = nn.BatchNorm1d(num_features = 256)
        self.bn3 = nn.BatchNorm1d(num_features = 128)

        
    def forward(self,x):
        #print(x.shape)
        
        y1 = self.feature1(x)                               #b,512,56,56
        y5,ind = self.delft2(y1)                            #b,512,k

        #y2 = self.feature2(x)                     #b,256,16,16
        #y2 = self.projector(y2)                   #b,512,16,16
        #y2 = self.delft2(y2)                            #b,256,k
        #y2 = self.projector(y2)                   #b,512,k

        ind_x = ind//56
        ind_y = ind%56
        ind_new = torch.stack((ind_x,ind_y),dim=2)

        #y5 = torch.cat((y1,y2),dim=2)             #b,512,nodes(2k)
        y5 = nn.ReLU()(y5)
        #trans = y5.permute(0,2,1)                 #b,nodes,512
        A = get_dist_mat(ind_new)

        #Normalizing adjacency matrix
        #A = A + I
        # print(A)
        #dhat = torch.diag_embed(torch.pow(torch.sum(A,2),-0.5))
        # print(dhat)
        #A = dhat@A@dhat                           #b,nodes,nodes

        #Building adjacency matrix
        
        #A = torch.matmul(trans,y5)                #b,nodes,nodes
        # b,nodes,nodes = A.size()
        # I = torch.eye(nodes).to(device)
        # I = I.reshape((1, nodes, nodes))
        # I = I.repeat(b, 1, 1)

        b,c,h,w = y5.size()
        # print(indices.shape)
        _,k = ind.shape
        # ind_exp = indices.unsqueeze(-1).expand(b,self.k,self.inp)
        ind_exp = ind.unsqueeze(-1).expand(b,k,c)             
        l_perm = y5.permute(0,2,3,1)                                        #l_perm = b,56,56,512
        l_perm_r = l_perm.reshape(b,h*w,-1)                                 #l_perm = b,h*w,512
        feat = torch.gather(l_perm_r,1,ind_exp)                             #feat = b,k,512 using indices found earlier to extract feature maps[Should work now]
        
        # feat_n = feat.permute(0,2,1) #check?
        feat = feat.permute(0,2,1)

        #Normalizing adjacency matrix
        #A = A + I
        # print(A)
        dhat = torch.diag_embed(torch.pow(torch.sum(A,2),-0.5))
        # print(dhat)
        A = dhat@A@dhat                           #b,nodes,nodes
        #print(A.shape)
        #print(A)
        

        #Graph Convolutions
#         print(y5[0,:30])
        y5 = self.gcn1(feat,A)
#         print(y5[0,:,30])
        y5 = self.bn2(y5)
        y5 = nn.ReLU()(y5)
        y5 = self.gcn2(y5,A)
        y5 = nn.ReLU()(y5)
#         print(y5[0,:,30])
        y5 = self.gcn3(y5,A)                       #b,256,nodes
        y5 = nn.ReLU()(y5)
        y5 = self.gcn4(y5,A)
#         print(y5[0,:,30])
        y6 = torch.mean(y5,2)                      #b,256
        y6 = self.mlp(y6)                          #b,classes(k = #classes)

        return y6,ind