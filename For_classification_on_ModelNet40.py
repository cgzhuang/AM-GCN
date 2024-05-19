import numpy as np
import torch
import torch.nn as nn
import random
import open3d as o3d
from models import *
device = torch.device('cuda'if torch.cuda.is_available()else'cpu')
print(device)


#数据输入
class_to_num = {'airplane': 0, 'bathtub': 1, 'bed': 2, 'bench': 3, 'bookshelf': 4, \
                'bottle': 5, 'bowl': 6, 'car': 7, 'chair': 8, 'cone': 9, 'cup': 10, \
                'curtain': 11, 'desk': 12, 'door': 13, 'dresser': 14, 'flower_pot': 15, \
                'glass_box': 16, 'guitar': 17, 'keyboard': 18, 'lamp': 19, 'laptop': 20, \
                'mantel': 21, 'monitor': 22, 'night_stand': 23, 'person': 24, 'piano': 25, \
                'plant': 26, 'radio': 27, 'range_hood': 28, 'sink': 29, 'sofa': 30, \
                'stairs': 31, 'stool': 32, 'table': 33, 'tent': 34, 'toilet': 35, \
                'tv_stand': 36, 'vase': 37, 'wardrobe': 38, 'xbox': 39}

def fps(points, sample_num):
    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = o3d.utility.Vector3dVector(points)
    sampled_pointcloud = o3d.geometry.PointCloud.farthest_point_down_sample(pointcloud, sample_num)
    sampled_points = np.asarray(sampled_pointcloud.points)
    return sampled_points

def load_pointcloud_labels(point_path):
    batch = len(point_path)
    sample_points = 4096
    labels = np.zeros((batch, 1))
    pointcloud = np.zeros((batch, sample_points, 3))
    for i in range(batch):
        points = np.loadtxt(point_path[i], delimiter=',', dtype=np.float32)
        points = points[:, :3]
        points = fps(points, sample_points)
        pointcloud[i] = points
        # print(point_path[i])
        label = point_path[i].split('/')[-2]
        labels[i] = class_to_num[label]
    return torch.tensor(pointcloud, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)

def load_path(path):
    root = 'modelnet40_normal_resampled/'
    point_path = []
    for p in path:
        types = p.split('0')[0][:-1]
        name = p.split('\n')[0]
        point_path.append(root + types + '/' + name + '.txt')
    return point_path

def train(net, criterion, optimizer, train_path_raw):
    net.train()
    all_train_loss =  0
    train_path = load_path(train_path_raw)
    pointcloud, labels = load_pointcloud_labels(train_path) # (B, N, 3), (B, 1)
    net.zero_grad()
    all_loss = 0
    for i in range(len(labels)):
        # 数据归一化
        min_value = torch.min(pointcloud[i])
        max_value = torch.max(pointcloud[i])
        pointcloud[i] = (pointcloud[i] - min_value) / (max_value - min_value)
        output, r = net(pointcloud[i].float().to(device))
        output = output.float().to(device)
        label = labels[i].long().view(-1,).to(device)
        loss1 = criterion[0](output, label)
        loss2 = criterion[1](r, torch.eye(3).to(device))
        loss = loss1+0.1*loss2
        all_loss += loss
        all_train_loss += loss1.item()+0.1*loss2.item()
        # print('loss1: %f, loss2: %f'%(loss1.item(), 0.1*loss2.item()))
    all_loss.backward()
    optimizer.step()
    return all_train_loss

def test(net,criterion,test_path_raw, Wrong_test):
    net.eval()
    test_loss = 0
    all_correct = 0
    number_all = 0
    test_path = load_path(test_path_raw)
    pointcloud,labels = load_pointcloud_labels(test_path)
    B, N, c = pointcloud.shape
    for i in range(B):
        # 数据归一化
        min_value = torch.min(pointcloud[i])
        max_value = torch.max(pointcloud[i])
        pointcloud[i] = (pointcloud[i] - min_value) / (max_value - min_value)
        output, r = net(pointcloud[i].float().to(device))
        label = labels[i].long().view(-1,).to(device)
        loss1 = criterion[0](output, label)
        loss2 = criterion[1](r, torch.eye(3).to(device))
        output = output.cpu()
        pre = torch.argmax(output, dim = 1).numpy()
        label = label.cpu().numpy()
        if pre != label:
            Wrong_test.append(test_path_raw[i])
        all_correct = all_correct+np.sum(pre==label)
        number_all = number_all+len(label)
        test_loss = test_loss+loss1.item()+0.1*loss2.item()
        # print('loss1: %f, loss2: %f'%(loss1.item(), 0.1*loss2.item()))
    return test_loss, all_correct/number_all, Wrong_test

batch_size = 4
epochs = 100
best_acc = 0


if __name__ == '__main__':
    #分类数据
    print(' ==== Loading data... ===== ')
    train_path = 'modelnet40_normal_resampled/modelnet40_train.txt'
    test_path = 'modelnet40_normal_resampled/modelnet40_test.txt'
    with open (train_path, 'r') as f:
        train_data = f.readlines()
    random.shuffle(train_data)
    with open (test_path, 'r') as f:
        test_data = f.readlines()
    random.shuffle(test_data)
    print(' ==== Load data successfully! ===== ')
    AMG = AMGCN_cls_GraphConv(40).to(device)
    # The first loss function is the cross-entropy loss function, for classification tasks,
    # and the second loss function is the MSE loss function, for T-Net.
    criterion = [torch.nn.CrossEntropyLoss().to(device), torch.nn.MSELoss().to(device)]
    optimizer = torch.optim.Adam(AMG.parameters(), lr=0.0001)
    for i in range(epochs):
        print(' ==== Training... ===== ')
        train_loss_all = 0
        for j in range(len(train_data)//batch_size):
            train_loss = train(AMG, criterion, optimizer, train_data[j*batch_size:(j+1)*batch_size])
            train_loss_all += train_loss
        print('epoch: %d, train_loss: %f'%(i, train_loss_all/(j+1)))
        test_loss_all = 0
        acc_all = 0
        Wrong_test = []
        print(' ==== Testing... ===== ')
        for j in range(len(test_data)//batch_size):
            test_loss, acc, Wrong_test = test(AMG, criterion, test_data[j*batch_size:(j+1)*batch_size], Wrong_test)
            test_loss_all += test_loss
            acc_all += acc
        np.savetxt('tmp/epoch_'+str(i)+'_Wrong_test.txt', Wrong_test, fmt='%s')
        print('epoch: %d, test_loss: %f, acc: %f'%(i, test_loss_all/(j+1), acc_all/(j+1)))
        if acc_all/(j+1) > best_acc:
            best_acc = acc_all/(j+1)
                

