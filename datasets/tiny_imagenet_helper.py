import os,shutil
import numpy as np

path = './tiny-imagenet-200/'
os.chdir(path)
# os.listdir(path+'train')
for name in os.listdir(path+'train'):
	if not(os.path.exists(path+'test/'+name)):
		os.mkdir(path+'test/'+name)

for name in os.listdir(path+'train'):
	sz = int(len(os.listdir(path+'train/'+name+'/images/'))/5)
	# l = np.random.choice(lst,sz).copy()
	# print(len(l),name)
	for i in range(sz):
		# file = i
		lst = os.listdir(path+'train/'+name+'/images/')
		file = np.random.choice(lst)
		shutil.move(path+'train/'+name+'/images/'+file,path+'test/'+name+'/'+file)