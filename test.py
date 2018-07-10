import numpy as np 
import csv
import os
from os.path import join as pjoin
from Common.net import resnet34
from Common.dataset import Dataset
from Common.save import restore
import torch
import torch.utils.data as data
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from PIL import Image

def test(model, label_maps, loader= None):
    model.eval()
    label = []
    files = []
    results = []
    # predictions = []
    labelmap = label_maps
    
    for dirs, _, imgs in os.walk(pjoin("/data1/Charactor_Recognition_Competetion", 'test2')):
        files += [img for img in imgs]
    # for t, (x, y) in enumerate(loader):
    for x in loader:        
        x_te = Variable(x.cuda(), volatile=True)
        scores = model(x_te)
        _, preds = scores.data.cpu().max(1)
        # files += y
        for pred in preds:
            # predictions += [pred]
            label += [labelmap[pred]]
    for i in range(len(label)):
        results +=[[files[i],label[i]]]
    
    print(results)
    with open('./predictions/test-label.csv', 'w', newline='') as f:
        title = [['filename', 'label']]
        writer = csv.writer(f)
        writer.writerows(title)
        writer.writerows(results)
    print('finished test') 


def check_top5_accuracy(model, label_maps, loader=None):
    print('Checking Top_5 accuracy on Test set')

    model.eval()
    labelname = []
    files = []
    results = []
    # predictions = []
    labelmap = label_maps

    for dirs, _, imgs in os.walk(pjoin("/data1/Charactor_Recognition_Competetion", 'test2')):
        files += [img for img in imgs]
    # for t, (x, y) in enumerate(loader):
    for x in loader:
        x_te = Variable(x.cuda(), volatile=True)
        scores = model(x_te)
        preds = scores.data.cpu().numpy()
        preds = np.argsort(preds, axis=1)
        predictions = preds[:,95:100]
        # print(predictions.shape)
        # print(predictions)
        
        # print('preds_shape is :',scores.shape)
        # print(preds)
        a,b = predictions.shape
        for i in range(a):
            label = []
            for j in range(b):                
                label.append(labelmap[predictions[i][j]])
            labelname.append(''.join(label))
    # print(label)
    for i in range(len(files)):
        results += [[files[i], labelname[i]]]
    print(results)
    with open('./predictions/test2-label-densenet.csv', 'w', newline='') as f:
        title = [['filename', 'label']]
        writer = csv.writer(f)
        writer.writerows(title)
        writer.writerows(results)
    print('finished test')

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    use_cuda = torch.cuda.is_available()
    test_datasets = Dataset("/data1/Charactor_Recognition_Competetion", 'test2')
    test_loader = data.DataLoader(test_datasets, batch_size=100)
    net = restore("./saved_nets/best_resenet.pkl")
    if  use_cuda:
        net.cuda()
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True
    

    labels = []
    for _, names, _ in os.walk(pjoin("/data1/Charactor_Recognition_Competetion", 'train')):
        for i, name in enumerate(names):
            labels += [name]
    label_map = {index: id for index, id in enumerate(labels)}
    print(label_map.items())
    test(net, label_map, test_loader)
    # check_top5_accuracy(net, label_map, test_loader)

if __name__ == '__main__':
    main()


