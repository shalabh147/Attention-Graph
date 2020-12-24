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

class Learner():

    def __init__(self,datasets,model,criterion,optimizer,scheduler=None,bs=2,num_workers=2,device='cuda:0'):
        '''
        :param datasets: a dictionary containing the dataset classes with keys 'train' and 'valid'
        :param criterion: loss function
        :param device: torch.device()
        :param optimizer: no default value
        :param scheduler: default none
        :param bs: batch size for training
        :param num_workers: number of dataloader workers
        :return:
        '''
        self.datasets = datasets
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dataloaders = {}
        self.dataset_sizes = {}
        for key in self.datasets:
            shuffle = True if key=='train' else False
            self.dataloaders[key] = DataLoader(self.datasets[key],batch_size=bs,num_workers=num_workers,shuffle=shuffle)
            self.dataset_sizes[key] = len(self.datasets[key])

    def fit(self,tb_logs=None,epochs=10):
        '''
        :param tb_logs: a dictionary containing tensorboard log flag, path and comment
        :param epochs: number of epochs
        :return:
        '''
        # manually set image size for standard dataset class
        # imgsize = self.datasets['train'].img_size
        imgsize = self.datasets['train'][0][0].size()[2]

        if tb_logs is not None:
            logpath = tb_logs['path']
            logcomment = tb_logs['comment']
            tb = SummaryWriter(log_dir=logpath+f'/{logcomment}', comment=logcomment)

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_valid_loss=100.
        self.record_dict = {'train': {'loss': [], 'acc': []}, 'valid': {'loss': [], 'acc': []}}

        try:
          tqdm._instances.clear()
        except:
          pass

        for epoch in range(epochs):
            cnt = 0
            print(f'EPOCH : {epoch + 1}/{epochs}')
            for phase in self.dataloaders.keys():
                since = time.time()
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()
                running_loss = 0.
                running_acc = 0.
                #------------------Progress Bar------------------#
                # bari = 0
                # bar = progressbar.ProgressBar(maxval=self.dataset_sizes[phase], \
                # widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
                # bar.start()
                # print(bari)
                #------------------------------------------------#
                for inputs, targets in tqdm(self.dataloaders[phase]):
                    inputs = inputs.float().view(-1, 3, imgsize, imgsize).to(self.device)
                    #print(inputs)
                    # print()
                    # print()
                    #print(inputs.shape)
                    # define targets and push to device
                    targets = targets.to(self.device)
                    #print(targets.shape)
                    #targets = torch.unsqueeze(targets,-1)
                    #print(targets.shape)
                    with torch.set_grad_enabled(phase=='train'):
                        self.optimizer.zero_grad()
                        #outputs = self.model(inputs)
                        outputs,indices = self.model(inputs)

                        # visualisation block
                        ind_x = indices//56                                                  #b,k
                        ind_y = indices%56                                                   #b,k
                        # #-----------------Saliency Map--------------------#
                        if epoch%30 == 0:  
                            img = inputs[0].detach().to('cpu').numpy()
                            # img = inputs[0].detach().to('cpu').numpy()
                            img = np.transpose(img,(1,2,0))
                            indx = ind_x.detach()[0,:].to('cpu').numpy()
                            indy = ind_y.detach()[0,:].to('cpu').numpy()
                            for i in range(len(indx)):
                                x,y = indx[i],indy[i]
                                img[4*x+1,4*y+1,:] = [1,0,0]
                                img[4*x+1,4*y+2,:] = [1,0,0]
                                img[4*x+2,4*y+1,:] = [1,0,0]
                                img[4*x+2,4*y+2,:] = [1,0,0]
                            # plt.imshow(img)
                            cnt += 1

                            plt.imsave('images/{}_{}.jpg'.format(epoch,cnt), np.clip(img,0,1))
                            plt.close()
                        #-------------------------------------------------#

                        #preds = outputs
                        preds = torch.argmax(outputs,dim=1)
                        #print(preds.shape)
                        # current loss written for categorical cross entropy
                        loss = self.criterion(outputs, targets)
                        # print("LOss function",loss)
                        # print(outputs)
                        # print(targets)
                        acc = torch.mean((targets==preds).to(float)).item()
                        if phase == 'train':
                            loss.backward()
                            #plot_grad_flow(self.model.named_parameters())
                            self.optimizer.step()
                            # self.scheduler.step()
                        running_loss += loss.item() * inputs.size()[0]
                        running_acc += acc*inputs.size()[0]
        
                        # Check for scheduler first
                        # if phase == 'train' and not(scheduler is None):
                        #     scheduler.step(epoch_loss)
                    # bari+=1
                    # bar.update(bari)
                
                # bar.finish()
                epoch_loss = running_loss/self.dataset_sizes[phase]
                epoch_acc = running_acc/self.dataset_sizes[phase]

                if tb_logs is not None:
                    if phase == 'train':
                        tb.add_scalar('Train Loss', epoch_loss, epoch)
                        tb.add_scalar('Train Acc',epoch_acc,epoch)
                    else:
                        tb.add_scalar('Valid Loss', epoch_loss, epoch)
                        tb.add_scalar('Valid Acc', epoch_acc, epoch)

                # validation loss based schedulers (Reduce on plateau)
                # if phase == 'valid' and not (self.scheduler is None):  
                #     self.scheduler.step(epoch_loss)

                self.record_dict[phase]['loss'].append(epoch_loss)
                self.record_dict[phase]['acc'].append(epoch_acc)

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss,epoch_acc))
                time_elapsed = time.time() - since
                print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
                # deep copy the best model
                if phase == 'valid' and epoch_loss < best_valid_loss:
                    best_valid_loss = epoch_loss
                    best_model_wts = copy.deepcopy(self.model.state_dict())
                
                path_output = '/content/ImageNet/model.pth'
                torch.save(best_model_wts, path_output)

                if tb_logs is not None:
                    pass
                    # to monitor learning rate
                    # current_lr = self.optimizer.param_groups[0]['lr']
                    # tb.add_scalar('Learning Rate', current_lr, epoch)

                    # define figures well if we want to display on tensorboard
                    # if epoch%15==0:
                    #     train_fig = 
                    #     valid_fig = 
                    #     tb.add_figure('train figs',train_fig)
                    #     tb.add_figure('valid figs',valid_fig)
        print('Best valid loss: {:4f}'.format(best_valid_loss))
        
        plt.plot(self.record_dict['train']['loss'])
        plt.savefig('/content/ImageNet/trainingloss')
        
        plt.close()
        
        plt.plot(self.record_dict['train']['acc'])
        plt.savefig('/content/ImageNet/trainingacc')
        plt.close()
        
        
        plt.plot(self.record_dict['valid']['loss'])
        plt.savefig('/content/ImageNet/Validloss')
        
        plt.close()
        
        plt.plot(self.record_dict['valid']['acc'])
        plt.savefig('/content/ImageNet/Validacc')
        plt.close()
        
        

        # load best model weights
        
        self.model.load_state_dict(best_model_wts)
        if tb_logs is not None:
            tb.close()