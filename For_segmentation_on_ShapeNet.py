import numpy as np
import torch
import torch.nn as nn
import os
import argparse
import matplotlib.pyplot as plt
from models import *
device = torch.device('cuda'if torch.cuda.is_available()else'cpu')
print(device)

#数据输入
class GetLoader(torch.utils.data.Dataset):
    def __init__(self,data_root,data_label):
        self.data = data_root
        self.label = data_label
    def __getitem__(self,index):
        data = self.data[index]
        label = self.label[index]
        return torch.tensor(data,dtype=torch.float32), torch.tensor(label,dtype=torch.float32)
    def __len__(self):
        return len(self.data)

#读取点云pts文件
def readLmk(fileName):
    if not os.path.exists(fileName):
        return 'error'
    else:
        with open(fileName) as file_obj:
            contents = file_obj.readlines() 
        Temp = list()
        for line in contents:
            TT = line.strip("\n")  
            TT_temp = TT.split(" ")
            TT_temp = np.array(TT_temp)
            Temp.append(TT_temp[np.newaxis,:])
        Temp = np.concatenate(Temp,axis=0)
        return Temp

def array_str2float(l):
    data = np.empty(shape=l.shape)
    for i in range(l.shape[0]):
        for j in range(l.shape[1]):
            data[i][j] = float(l[i][j])
    assert(data.shape==l.shape)
    return torch.tensor(data,dtype=torch.float32)
def array_str2int(l):
    len0 = len(l)
    data = np.empty(shape=l.shape)
    for i in range(len0):
        data[i] = int(l[i])
    assert(data.shape==l.shape)
    return torch.tensor(data,dtype=torch.float32)

type_name = {'areo':4, 'bag':2, 'cap':2, 'car':4, 'chair':4, 'ear_phone':3, 'guitar':3, 'knife':2, 'lamp':4, 'laptop':2, 'motor':6, 'mug':2,\
             'pistol':3 , 'rocket':3, 'skate_board':3, 'table':3}
def data_load(name):
    print('Loading Data... (type:{})'.format(name))
    with open('./shapenet_core/train_test_split/' + name + '_train.txt') as fp:
        data_train = fp.readlines()
        fp.close()
    with open('./shapenet_core/train_test_split/' + name + '_val.txt') as fp:
        data_val = fp.readlines()
        fp.close()
    with open('./shapenet_core/train_test_split/' + name + '_test.txt') as fp:
        data_test = fp.readlines()
        fp.close()

    train_point = []
    train_label = []
    val_point = []
    val_label = []
    test_point = []
    test_label = []

    for specific in data_train:
        spe = specific.split('\n')[0]
        train_point.append('./shapenet_core/' + name + '/points/' + spe + '.pts')
        train_label.append('./shapenet_core/' + name + '/points_label/' + spe + '.seg')

    for specific in data_val:
        spe = specific.split('\n')[0]
        val_point.append('./shapenet_core/' + name + '/points/' + spe + '.pts')
        val_label.append('./shapenet_core/' + name + '/points_label/' + spe + '.seg')

    for specific in data_test:
        spe = specific.split('\n')[0]
        test_point.append('./shapenet_core/' + name + '/points/' + spe + '.pts')
        test_label.append('./shapenet_core/' + name + '/points_label/' + spe + '.seg')

    train_path = [train_point, train_label]
    val_path = [val_point, val_label]
    test_path = [test_point, test_label]
    if len(test_label) == 0:
        print('Data Error!')
        raise RuntimeError('Data Error!')
    print('Loading Successfully! (type:{})'.format(name))
    print('Train data num: {} Val data num: {} Test data num: {}'.format(len(train_point), len(val_point), len(test_point)))
    return train_path, val_path, test_path, type_name[name]

#计算每个类别的交集和并集大小
def intersection_and_union(predictions, labels, num_classes):
    # 初始化每个类别的交集和并集大小

    intersections = torch.zeros(num_classes, dtype=torch.int)
    unions_gt = torch.zeros(num_classes, dtype=torch.int)
    unions_pred = torch.zeros(num_classes, dtype=torch.int)
    num_points = labels.shape[0]
    # 计算每个类别的交集和并集大小
    for i in range(num_points):
        gt = labels[i]
        pred = predictions[i]
        intersections[gt] += int(gt == pred)
        unions_gt[gt] += 1
        unions_pred[pred] += 1

    # 计算每个类别的 Point IoU
    ious = []
    for i in range(num_classes):
        intersection = intersections[i]
        union = unions_gt[i] + unions_pred[i] - intersection
        iou_class = intersection / union if union > 0 else 1
        ious.append(iou_class)
        
    # 计算所有类别的平均 Point IoU
    mean_iou = sum(ious) / num_classes
    return mean_iou

#以下train和test针对的是点云数目不一定相同的点云，但点云数目至少大于FPS点数和KNN点数
def train(net, pointcloud, pointlabel, criterion, optimizer):
    net.train()
    net.zero_grad()
    output, r = net(pointcloud.to(device))
    labels = pointlabel.long().squeeze().to(device)
    loss1 = criterion[0](output, labels)
    loss2 = criterion[1](r, torch.eye(3).to(device))
    all_loss = loss1+0.3*loss2
    all_loss.backward()
    optimizer.step()
    return loss1.item()+0.3*loss2.item()

#以下train和test针对的是点云数目不一定相同的点云，但点云数目至少大于FPS点数和KNN点数
def train_test(net, pointcloud, pointlabel, criterion, optimizer, part_num):
    net.train()
    net.zero_grad()
    output, r = net(pointcloud.to(device))
    labels = pointlabel.long().squeeze().to(device)
    loss1 = criterion[0](output, labels)
    loss2 = criterion[1](r, torch.eye(3).to(device))
    all_loss = loss1+0.3*loss2
    all_loss.backward()
    optimizer.step()
    pre = torch.argmax(output, dim=1)
    mean_iou = intersection_and_union(pre.cpu(), labels.cpu(), part_num)
    mean_acc = np.sum(pre.cpu().numpy()==labels.cpu().numpy().squeeze())/len(pre)
    return loss1.item()+0.3*loss2.item(), mean_iou, mean_acc

def test(net, pointcloud, pointlabel, part_num):
    net.eval()
    with torch.no_grad():
        output, r = net(pointcloud.to(device))
        labels = pointlabel.long().squeeze().to(device)
        loss1 = criterion[0](output, labels)
        loss2 = criterion[1](r, torch.eye(3).to(device))

    pre = torch.argmax(output, dim=1).squeeze()
    mean_iou = intersection_and_union(pre.cpu(), labels.cpu(), part_num)
    mean_acc = np.sum(pre.cpu().numpy()==labels.cpu().numpy().squeeze())/len(pre)
    return loss1.item()+0.3*loss2.item(), mean_iou, mean_acc

def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=10, help='Train Epoches')
    parser.add_argument('--name', type=str, default='cap', help='Train type name')
    parser.add_argument('--date', type=str, default='0410', help='Train data(eg. 1110)')
    parser.add_argument('--device', type=str, default='cpu', help='Train device(only one)')
    parser.add_argument('--save_epoch', type=int, default=10,
                        help='how many epoch to save once')
    parser.add_argument('--pretrained', type=str,
                        default='', help='pretrained model')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--optimizer', type=str,
                        default='adam', help='Optimizer: adma/SGD')
    return parser


if __name__ == '__main__':
    parser = get_argparse()
    opt = parser.parse_args()
    if opt.device != 'cpu':
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.device
        device = 'cuda'
    print(opt.device)
    train_path, test_path, val_path, parnum = data_load(opt.name)
    if opt.pretrained == '':
        mygcn = AMGCN_seg_EdgeConv(parnum).to(device)
    else:
        mygcn = AMGCN_seg_EdgeConv(parnum).to(device)
        mygcn.load_state_dict(torch.load(opt.pretrained))
    if opt.optimizer == 'adam':
        optimizer = torch.optim.Adam(mygcn.parameters(), lr=opt.lr)
    elif opt.optimizer == 'SGD':
        optimizer = torch.optim.SGD(mygcn.parameters(), lr=opt.lr, momentum=0.9, dampening=0.5, weight_decay=0.01,
                                    nesterov=False)
    criterion = []
    criterion.append(torch.nn.CrossEntropyLoss().to(device))
    criterion.append(torch.nn.MSELoss().to(device))
    accuracy_best = 0
    miou_best = 0
    epoch_best = 0
    train_loss_plot = []
    test_loss_plot = []
    for epoch in range(opt.epoch):
        print('Epoch:{}'.format(epoch))
        print('Train...')
        train_loss = 0
        accuracy = 0
        miou = 0
        for i in range(len(train_path[0])):
            pointcloud = array_str2float(readLmk(train_path[0][i]))
            pointlabel = array_str2int(readLmk(train_path[1][i])) - 1
            if epoch % 10 == 0:
                loss, iou, acc = train_test(mygcn, pointcloud, pointlabel, criterion, optimizer, parnum)
                train_loss = train_loss + loss/len(train_path[0])
                accuracy = accuracy + acc
                miou = miou + iou
            if epoch % 10 != 0:
                loss = train(mygcn, pointcloud, pointlabel, criterion, optimizer)
                train_loss = train_loss + loss/len(train_path[0])
        if epoch % 10 == 0:
            print('{} Epoch: {} Train loss: {} mIOU: {} Accuracy: {}'.\
                  format(opt.name, epoch, train_loss, miou / len(train_path[0]), accuracy / len(train_path[0])))
        else:
            print('{} Epoch: {} Train loss: {}'.format(opt.name, epoch, train_loss))

        print('{} Epoch: {} Train loss: {}'.format(opt.name, epoch, train_loss))
        if epoch % 2 != 0:
            continue
        print('Val...')
        test_loss = 0
        accuracy = 0
        miou = 0
        for i in range(len(val_path[0])):
            pointcloud = array_str2float(readLmk(val_path[0][i]))
            pointlabel = array_str2int(readLmk(val_path[1][i])) - 1
            loss, iou, acc = test(mygcn, pointcloud, pointlabel, parnum)
            test_loss = test_loss+loss/len(val_path[0])
            accuracy = accuracy + acc
            miou = miou + iou

        accuracy = accuracy / len(val_path[0])
        miou = miou / len(val_path[0])
        if accuracy > accuracy_best:
            accuracy_best = accuracy
        if miou > miou_best:
            miou_best = miou
            epoch_best = epoch
            torch.save(mygcn.state_dict(), './Final_model/Shapenet_' + opt.name + '_EdgeConv.pth')
            print('Model saved!')
        print('{} Epoch: {} Val loss: {}'.format(opt.name, epoch, test_loss))
        print('{} Epoch: {} Accuracy: {} mIOU: {}'.format(opt.name, epoch, accuracy, miou))
        print('{} Best Epoch: {} Best Accuracy: {}Best mIOU: {}'.format(opt.name, epoch_best, accuracy_best, miou_best))
        train_loss_plot.append(train_loss)
        test_loss_plot.append(test_loss)
        if epoch % opt.save_epoch == 0:
            torch.save(mygcn.state_dict(), './Final_model/Shapenet_' + opt.name + '_EdgeConv.pth')
            print('Model saved!')
        if (epoch+1) != opt.epoch:
            continue

        print('============Test============')
        test_loss = 0
        accuracy = 0
        miou = 0
        for i in range(len(test_path[0])):
            pointcloud = array_str2float(readLmk(test_path[0][i]))
            pointlabel = array_str2int(readLmk(test_path[1][i])) - 1
            loss, iou, acc = test(mygcn, pointcloud, pointlabel, parnum)
            test_loss = test_loss+loss/len(test_path[0])
            accuracy = accuracy + acc
            miou = miou + iou

        accuracy = accuracy / len(test_path[0])
        miou = miou / len(test_path[0])
        print('{} Epoch: {} Test loss: {}'.format(opt.name, epoch, test_loss))
        print('{} Epoch: {} Accuracy: {}'.format(opt.name, epoch, accuracy))
        print('{} Epoch: {} mIOU: {}'.format(opt.name, epoch, miou))
        # 保存模型
        torch.save(mygcn.state_dict(), './Final_model/Shapenet_' + opt.name + '_EdgeConv.pth')
        print('Model saved!')
        
# %%
