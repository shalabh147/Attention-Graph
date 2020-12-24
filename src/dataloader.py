import torch
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as transforms

def dataloader():
	train_path = '../datasets/ImageNet/train/train/'
	#def train_dataloader():
	train_transform = transforms.Compose([transforms.Resize((224,224)),transforms.RandomHorizontalFlip(),transforms.RandomRotation(15),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
	# train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
	valid_transform = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
	train_data = torchvision.datasets.ImageFolder(root=train_path,transform=train_transform)
	#dataloader = torch.utils.data.DataLoader(data,batch_size=4,shuffle=True)
	#return dataloader

	test_path = '../datasets/ImageNet/train/val/'
	test_data = torchvision.datasets.ImageFolder(root = test_path,transform=valid_transform)

	datasets = {'train':train_data,'valid':test_data}
	return datasets