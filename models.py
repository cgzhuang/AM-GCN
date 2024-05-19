import torch
import torch.nn as nn
from dgl.nn.pytorch import GraphConv
from dgl.nn.pytorch import GATConv
from dgl.nn.pytorch import EdgeConv
from dgl.nn.pytorch import GATv2Conv
from utils import *

# T-net, for learning the transformation matrix
class T_net(nn.Module):
    def __init__(self):
        super(T_net, self).__init__()
        self.fc1 = nn.Linear(3, 64)
        self.fc2 = nn.Linear(64, 1024)

        self.fc3 = nn.Linear(1024, 256)
        self.fc4 = nn.Linear(256, 9)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        self.maxpool = nn.MaxPool1d(x.shape[0])
        x = self.maxpool(x.transpose(1, 0)).transpose(1, 0) # (1, 1024)
        x = self.relu(self.fc3(x))
        x = self.fc4(x).view(3, 3)
        q,r = torch.linalg.qr(x)
        return q, r

# AMGCN for classification based on DGL.GraphConv
# If you want to change GraphConv to EdgeConv, you can change the GraphConv to EdgeConv
class AMGCN_cls_GraphConv(nn.Module):
    def __init__(self, cls_num, in_dim=3, hidden_dim1=16, hidden_dim2=64):
        super(AMGCN_cls_GraphConv, self).__init__()
        self.tnet = T_net()
        self.Gconv1 = GraphConv(in_dim, hidden_dim2)
        self.Gconv2 = GraphConv(hidden_dim2, hidden_dim2)
        self.Gconv3 = GraphConv(hidden_dim2, hidden_dim2)
        self.Gconv4 = GraphConv(hidden_dim2, hidden_dim2)
        self.Gconv5 = GraphConv(hidden_dim2, hidden_dim2)

        self.Gconv1xy = GraphConv(in_dim, hidden_dim1)
        self.Gconv2xy = GraphConv(hidden_dim1, hidden_dim2)

        self.Gconv1yz = GraphConv(in_dim, hidden_dim1)
        self.Gconv2yz = GraphConv(hidden_dim1, hidden_dim2)

        self.Gconv1xz = GraphConv(in_dim, hidden_dim1)
        self.Gconv2xz = GraphConv(hidden_dim1, hidden_dim2)

        self.fc0 = nn.Linear(hidden_dim2*3, 1024)
        self.drop0 = nn.Dropout(0.4)
    
        self.gatv2conv = GATv2Conv(1024+hidden_dim2*6, 256, num_heads=4)

        self.fc1 = nn.Linear(1024, 256)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(256, 64)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(64, cls_num)

        self.relu = nn.ReLU()

    def forward(self, pointcloud):
        # pointcloud: (N, 3)
        t_matrix, r = self.tnet(pointcloud)
        x = torch.matmul(pointcloud, t_matrix)
        g_batch = convert_pointcloud_to_dgl_graph_k(x)

        h0 = self.Gconv1(g_batch[0], x) # (N, 64)
        h0 = self.Gconv2(g_batch[0], h0) # (N, 64)
        g0 = create_graph_with1channel_k(h0)

        h1 = self.Gconv3(g0, h0) # (N, 64)
        h1 = self.Gconv4(g0, h1) # (N, 64)
        g1 = create_graph_with1channel_k(h1)

        h2 = self.Gconv5(g1, h1) # (N, 64)

        h = torch.cat((h0, h1, h2), dim=1) # (N, 64*3)
        h_ = self.fc0(h) # (N, 1024)
        self.maxpool1 = nn.MaxPool1d(x.shape[0])
        h_ = self.maxpool1(h_.transpose(1, 0)).transpose(1, 0)  # (1, 1024)
        h_ = h_.repeat(x.shape[0], 1)   # (N, 1024)

        hxy = self.Gconv1xy(g_batch[1], x) # (N, 16)
        gxy = create_graph_with1channel_k(hxy, k=50)
        hxy_ = self.Gconv2xy(gxy, hxy) # (N, 64)

        hyz = self.Gconv1yz(g_batch[2], x) # (N, 16)
        gyz = create_graph_with1channel_k(hyz, k=50)
        hyz_ = self.Gconv2yz(gyz, hyz) # (N, 64)

        hxz = self.Gconv1xz(g_batch[3], x) # (N, 16)
        gxz = create_graph_with1channel_k(hxz, k=50)
        hxz_ = self.Gconv2xz(gxz, hxz) # (N, 64)

        h = torch.cat((h, h_, hxy_, hyz_, hxz_), dim=1) # (N, 1024+64*4)
        h = self.gatv2conv(g_batch[0], h).view(pointcloud.shape[0], -1) # (N, 256*4)

        self.maxpool2 = nn.MaxPool1d(x.shape[0])
        h = self.maxpool2(h.transpose(1, 0)).transpose(1, 0)  # (1, 256*4)

        h = self.relu(self.fc1(h)) # (1, 256)
        h = self.drop1(h)
        h = self.relu(self.fc2(h)) # (1, 64)
        h = self.drop2(h)
        h = self.fc3(h) # (1, cls_num)
        return h, r
    
# AMGCN for part segmentation based on DGL.EdgeConv
# If you want to change EdgeConv to GraphConv, you can change the EdgeConv to GraphConv
class AMGCN_seg_EdgeConv(nn.Module):
    def __init__(self, par_num, in_dim=3, hidden_dim1=16, hidden_dim2=64):
        super(AMGCN_seg_EdgeConv, self).__init__()
        self.tnet = T_net()
        self.Gconv1 = EdgeConv(in_dim, hidden_dim2)
        self.Gconv2 = EdgeConv(hidden_dim2, hidden_dim2)
        self.Gconv3 = EdgeConv(hidden_dim2, hidden_dim2)
        self.Gconv4 = EdgeConv(hidden_dim2, hidden_dim2)
        self.Gconv5 = EdgeConv(hidden_dim2, hidden_dim2)

        self.Econv1xy = EdgeConv(in_dim, hidden_dim1)
        self.Econv2xy = EdgeConv(hidden_dim1, hidden_dim2)

        self.Econv1yz = EdgeConv(in_dim, hidden_dim1)
        self.Econv2yz = EdgeConv(hidden_dim1, hidden_dim2)

        self.Econv1xz = EdgeConv(in_dim, hidden_dim1)
        self.Econv2xz = EdgeConv(hidden_dim1, hidden_dim2)
        
        
        self.fc0 = nn.Linear(hidden_dim2*3, 1024)
        self.drop0 = nn.Dropout(0.4)

        self.gatv2conv = GATv2Conv(1024+hidden_dim2*6, 256, num_heads=4)

        self.fc1 = nn.Linear(1024, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(256, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, par_num)

        self.relu = nn.ReLU()

    def forward(self, pointcloud):
        # pointcloud: (N, 3)
        t_matrix, r = self.tnet(pointcloud)
        x = torch.matmul(pointcloud, t_matrix)
        g_batch = convert_pointcloud_to_dgl_graph_k(x)

        h0 = self.Gconv1(g_batch[0], x) # (N, 64)
        h0 = self.Gconv2(g_batch[0], h0) # (N, 64)
        g0 = create_graph_with1channel_k(h0, 30)

        h1 = self.Gconv3(g0, h0) # (N, 64)
        h1 = self.Gconv4(g0, h1) # (N, 64)
        g1 = create_graph_with1channel_k(h1, 30)

        h2 = self.Gconv5(g1, h1) # (N, 64)

        h = torch.cat((h0, h1, h2), dim=1) # (N, 64*3)
        h_ = self.fc0(h) # (N, 1024)
        self.maxpool1 = nn.MaxPool1d(x.shape[0])
        h_ = self.maxpool1(h_.transpose(1, 0)).transpose(1, 0)  # (1, 1024)
        h_ = h_.repeat(x.shape[0], 1) # (N, 1024)

        hxy = self.Econv1xy(g_batch[1], x) # (N, 16)
        gxy = create_graph_with1channel_k(hxy, k=100)
        hxy_ = self.Econv2xy(gxy, hxy) # (N, 64)

        hyz = self.Econv1yz(g_batch[2], x) # (N, 16)
        gyz = create_graph_with1channel_k(hyz, k=100)
        hyz_ = self.Econv2yz(gyz, hyz) # (N, 64)

        hxz = self.Econv1xz(g_batch[3], x) # (N, 16)
        gxz = create_graph_with1channel_k(hxz, k=100)
        hxz_ = self.Econv2xz(gxz, hxz) # (N, 64)

        h = torch.cat((h, h_, hxy_, hyz_, hxz_), dim=1) # (N, 1024+64*6)
        h = self.gatv2conv(g_batch[0], h).view(pointcloud.shape[0], -1) # (N, 256*4)

        h = self.relu(self.bn1(self.fc1(h))) # (N, 256)
        h = self.drop1(h)
        h = self.relu(self.bn2(self.fc2(h))) # (N, 256)
        h = self.drop2(h)
        h = self.fc3(h) # (N, par_num)

        return h, r

# For the model for scene segmentation, you can refer to the model for part segmentation

if __name__ == '__main__':
    
    model_cls = AMGCN_cls_GraphConv(40)
    model_seg = AMGCN_seg_EdgeConv(5)

    tmp_data = torch.randn(2048, 3)

    out_cls, r_cls = model_cls(tmp_data)
    print(out_cls.shape, r_cls.shape)
    out_seg, r_seg = model_seg(tmp_data)
    print(out_seg.shape, r_seg.shape)