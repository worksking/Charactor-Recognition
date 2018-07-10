import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.autograd import Variable
from Common.net import resnet101
from Common.densenet import densenet161
from Common.dataset import Dataset
from Common.save import localtime, save
import torchvision.models as models
import torch.backends.cudnn as cudnn

def train(model, loss_fn, optimizer, lr_schedule, num_epochs=1, train_loader=None, val_loader=None):
    best_loss = 1
    best_val_acc = 0
    best_val_loss = 10
    for epoch in range(num_epochs):
        print('Starting epoch %d / %d' % (epoch + 1, num_epochs))
        num_correct = 0
        num_samples = 0
        loss = 0
        average_loss = 0
        model.train()
        for t, (x, y) in enumerate(train_loader):
            x_train = Variable(x.cuda())
            y_train = Variable(y.cuda())

            scores = model(x_train)
            loss = loss_fn(scores, y_train)            

            # reference https://discuss.pytorch.org/t/argmax-with-pytorch/1528
            _, preds = scores.data.cpu().max(1)

            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
            acc = float(num_correct) / num_samples
            # average_loss += loss
            # average_loss /= (t+1) 

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (t + 1) % 20 == 0:
                print('epoch = %d, t = %d, loss = %.4f, average_loss = %.4f, acc = %.4f%%' %
                      (epoch+1, t+1, loss.data[0], average_loss, 100 * acc ))
        val_acc,val_loss = check_accuracy(model, loss_fn, val_loader)
        # val_acc, val_loss = check_top5_accuracy(model, loss_fn, val_loader)
        lr_schedule.step(val_acc, epoch=epoch+1)

        if best_val_acc < val_acc:
            best_val_acc = val_acc
            print("best val_accuracy:", best_val_acc)
        # if best_val_loss > val_loss:
        #     best_val_loss = val_loss
        #     print("best validation loss is:", best_val_loss)            
            print("-------------------")
            print("saving net")
            # save(model, False, True)
            torch.save(model, f'./saved_nets/best_resenet.pkl')
            print("-------------------")
        print("current best validation accuracy is:",best_val_acc)
        # else:
        #     adjust_learning_rate(optimizer, epoch)
        

        # val_acc = check_top5_accuracy(model,val_loader)
        # lr_schedule.step(val_acc, epoch=epoch+1)
        # if val_acc > best_val_acc:
        #     best_val_acc = val_acc
        #     print("saving net.....")
        #     save(model, True, True)
        # #adjust_learning_rate(optimizer,epoch)
        # print('-------------------------------')
        # print("The best validation accuracy:%.4f%%" % (100 * best_val_acc))
        # print('-------------------------------')



def check_accuracy(model, loss_fn, loader):
    print("------------------------------------")
    print('Checking top1_accuracy on validation set')

    val_correct = 0
    val_samples = 0
    loss = 0
    # Put the model in test mode (the opposite of model.train(), essentially)
    model.eval()
    for t, (x, y) in enumerate(loader):
        # reference https://pytorch-cn.readthedocs.io/zh/latest/notes/autograd/
        x_val = Variable(x.cuda(), volatile=True)
        y_val = Variable(y.cuda(), volatile=True)

        scores = model(x_val.type(torch.cuda.FloatTensor))
        loss += loss_fn(scores, y_val).data[0]
        t = t+1
        _, preds = scores.data.cpu().max(1)
        val_correct += (preds == y).sum()
        val_samples += preds.size(0)
    val_acc = float(val_correct) / val_samples
    val_loss = loss/t
    print('val_loss:%.4f, Got %d / %d correct (%.4f%%)' % (val_loss, val_correct, val_samples, 100 * val_acc))
    print("-------------------------------------")
    return val_acc, val_loss

def check_top5_accuracy(model, loss_fn, loader=None):
    print("------------------------------------")
    print('Checking Top_5 accuracy on validation set')

    val_correct = np.int32(0)
    val_samples = 0
    loss = 0
    # Put the model in test mode (the opposite of model.train(), essentially)
    model.eval()
    for t, (x, y) in enumerate(loader):
        # reference https://pytorch-cn.readthedocs.io/zh/latest/notes/autograd/
        x_val = Variable(x.cuda(), volatile=True)
        y_val = Variable(y.cuda(), volatile=True)

        scores = model(x_val)
        loss += loss_fn(scores, y_val).data[0]
        t = t+1
        preds = scores.data.cpu().numpy()
        preds = np.argsort(preds, axis=1)
        predictions = preds[:, 95:100]
        # print(type(val_correct),type(predictions), type(y.numpy()))
        # print(predictions.shape, y.numpy().shape)
        val_correct += (predictions == np.reshape(y.numpy(),(-1,1))).sum()
        val_samples += preds.shape[0]
    val_acc = float(val_correct) / val_samples
    val_loss = loss/t
    print('val_loss:%.4f, Got %d / %d correct (%.4f%%)' %
          (val_loss, val_correct, val_samples, 100 * val_acc))
    print("------------------------------------")
    return val_acc, val_loss




#def adjust_learning_rate(optimizer, decay_rate=0.8):
#    for param_group in optimizer.param_groups:
#        param_group['lr'] = param_group['lr'] * decay_rate
def adjust_learning_rate(optimizer, num_epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    #lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr']*(0.5 ** (num_epoch //1.0))


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    use_cuda = torch.cuda.is_available()
    
    train_datasets = Dataset("/data1/Charactor_Recognition_Competetion")
    train_loader = data.DataLoader(train_datasets, batch_size=100, shuffle=True, num_workers=4)
    val_datasets = Dataset("/data1/Charactor_Recognition_Competetion", 'val')
    val_loader = data.DataLoader(val_datasets, batch_size=100, shuffle=True, num_workers=4)

    ############################################################################################  
    # net = resnet101()
    # densenet = models.densenet161(pretrained=True)
    net = densenet161()

    # #重置第一层卷积层
    # densenet.features = nn.Sequential(densenet.features

    # #提取fc层中固定的参数
    # fc_features =densenet.classifier.in_features
    # #修改类别为100
    # densenet.classifier = nn.Linear(fc_features*4*4, 100)

    # #读取参数
    # pretrained_dict = densenet.state_dict()
    # model_dict = net.state_dict()

    # # 将pretrained_dict里不属于model_dict的键剔除掉
    # pretrained_dict = {k: v for k,
    #                    v in pretrained_dict.items() if k in model_dict}

    # # 更新现有的model_dict
    # model_dict.update(pretrained_dict)

    # # 加载我们真正需要的state_dict
    # net.load_state_dict(model_dict)
    #############################################################################################
    if use_cuda:
        net.cuda()
        net = torch.nn.DataParallel(
            net, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True
    
    lr = 1e-3
    # optimizer = optim.SGD(params=net.parameters(), lr=0.01, momentum=0.9,weight_decay= 2e-5, nesterov=True)
    optimizer = optim.Adam(params=net.parameters(), lr=lr, weight_decay = 1e-10)
    lr_schedule = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',factor=0.3, verbose= True, patience=5)

    loss_fn = nn.CrossEntropyLoss()
    num_epochs = 300
    train(net, loss_fn, optimizer,lr_schedule, num_epochs=num_epochs, train_loader=train_loader, val_loader=val_loader)

if __name__ == '__main__':
    main()
