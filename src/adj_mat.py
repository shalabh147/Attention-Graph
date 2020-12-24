import torch

def get_dist_mat(coords_org):
    coords=coords_org.clone().detach()                      
    n=coords.shape[1]                                                             #coords_org = b,n,2
    tid=torch.eye(n)
    tid=tid.to(coords_org.device)
    tid.requires_grad=False

    dist_mat=coords.clone()                                                   #dist_mat = b,n,2
    dist_mat=dist_mat.unsqueeze(2)                                        #dist_mat = b,n,1,2
    coords=coords.unsqueeze(2)                                              #coords = b,n,1,2
    #print(coords.shape,n)
    for i in range(n-1):
        dist_mat=torch.cat([dist_mat,coords],dim=2)               #dist_mat = b,n,n,2
        
    #print(dist_mat.shape)
    #print(tid.shape)
    dist_mat = dist_mat.sub(coords.transpose(1,2))                #dist_mat = b,n,n,2
    dist_mat = torch.square(dist_mat)                           
    dist_mat = torch.sum(dist_mat,dim=3)                                #dist_mat = b,n,n
    dist_mat = torch.cuda.FloatTensor(dist_mat.float())
    dist_mat = torch.sqrt(dist_mat) + tid
    
    
    #dist_mat[dist_mat<=1]=1
    #dist_mat[torch.logical_and((dist_mat>1),(dist_mat<=10))]=0.5
    #dist_mat[torch.logical_and((dist_mat>10),(dist_mat<=40))]=0.25
    #dist_mat[dist_mat>40]=0
    
    
    dist_mat = 1/dist_mat
    #dist_mat = dist_mat - tid
    dist_mat = dist_mat/torch.max(dist_mat)
    dist_mat = dist_mat-tid*dist_mat
    dist_mat = dist_mat+tid*(torch.max(dist_mat))
    
    
    #dist_mat=dist_mat.detach()
    dist_mat.requires_grad=False
    
    return dist_mat
