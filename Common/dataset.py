import numpy as np
import os
from os.path import join as join
from os.path import abspath
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image
import random

class Dataset(data.Dataset):
    '''

    '''
    def __init__(self, root, data_dir='train', transform=transforms.ToTensor(), loader=None):
        assert data_dir in ('train', 'val', 'test', 'test2'), "data_dir must be 'train', 'val' or 'test'"
        self.root = join(root, data_dir)
        self.transforms = transform
        self.loader = loader
        self.image = []
        self.names = []
        self.lables = []
        self.dir = [] 
        self.files = []
        self.data_dir = data_dir

        for _, names, _ in os.walk(join(root, 'train')):
            for i, name in enumerate(names):
                self.names += [name]
        self.lable_map = {id:index for index,id in enumerate(self.names)}
        print(self.lable_map.items())


        if self.data_dir == 'train':       
            for dirs, names, files in os.walk(self.root):
                # print(len(files))
                if len(files) == 400:            
                    for file in files[0:380]:                
                        self.files.append(join(dirs,file))
                for i, name in enumerate(names):
                    self.lables += name
            # print(len(self.files))            
            self.lables = [self.lable_map[lable] for lable in self.lables]
            self.lables = sorted(self.lables * 380)
            # print('label length:',len(self.lables))           
            # print('image length:',len(self.image))
            self.image = self.files
            # for i in range(35000):
            #     print(i,self.lables[i], self.image[i])

        elif self.data_dir == 'val':
            for dirs, names, files in os.walk(join(root, 'train')):
                # print(len(files))
                if len(files) == 400:
                    for file in files[380:400]:
                        self.files.append(join(dirs, file))
                for i, name in enumerate(names):
                    self.lables += name
            # print(len(self.files))
            self.lables = [self.lable_map[lable] for lable in self.lables]
            self.lables = sorted(self.lables * 20)
            # print('label length:', len(self.lables))
            # for img in reversed(self.files):
            #     self.image.append(img)
            self.image = self.files
            # print('image length:', len(self.image))
            # for i in range(5000):
            #     print(i, self.lables[i], self.image[i])

        elif self.data_dir == 'test2':
            for dirs, _, files in os.walk(self.root):
                self.image += [join(dirs, file) for file in files]
                # self.files += [file for file in files]
            # for i in range(len(self.image)):
            #     print(self.image[i],self.files[i])

            

    def _loader(self, path):
        return Image.open(path).convert('L').resize((256,256)).rotate(random.randint(-80,80))

    def image_loader(self,path):
        return Image.open(path).convert('L').resize((256, 256))

    def __getitem__(self, index):
        if self.loader is None:
            self.loader = self._loader
        
        if self.data_dir =='test2':
            self.loader = self.image_loader
            imgs = self.image[index]
            imgs = self.transforms(self.loader(imgs))
            # files = self.files[index]
            # return imgs, files
            return imgs

        imgs = self.image[index]        
        imgs = self.transforms(self.loader(imgs))
        lables = self.lables[index]
        return imgs, lables
    
    def __len__(self):
        return len(self.image)


# '''
# test file
# '''
# def main():
#     test_datasets = Dataset("/data1/Charactor_Recognition_Competetion",'train')

# if __name__ == '__main__':
#     main()
    
