import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import random
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 点云数据转为图数据 Convert point cloud to graph data
# Construct the graph with the k nearest neighbors
def create_graph_with1channel_k(point, k=30):
    g = dgl.DGLGraph().to(device)
    num_points = point.shape[0]
    point_distances = torch.cdist(point, point).to(device)
    _, nearest_indices = torch.topk(point_distances, k, largest=False)
    src_nodes = torch.arange(num_points).repeat_interleave(k).to(device)
    dst_nodes = nearest_indices.flatten().to(device)
    g.add_edges(src_nodes, dst_nodes)
    g = dgl.add_self_loop(g)
    return g

# Convert point cloud to graph data
# Construct the graph with the k nearest neighbors
# For every projection of the point cloud and the point cloud itself
def convert_pointcloud_to_dgl_graph_k(pointcloud):
    """
    input:
        pointcloud (torch.Tensor): Point cloud to convert. Shape (N, 3)
    Returns:
        g_batch (dgl.DGLGraph): Batched graph.
    """
    # B, N, c = pointcloud.shape
    gxy = create_graph_with1channel_k(pointcloud[:, :2], k=50)
    gyz = create_graph_with1channel_k(pointcloud[:, 1:], k=50)
    gxz = create_graph_with1channel_k(pointcloud[:, [0, 2]], k=50)
    gxyz5 = create_graph_with1channel_k(pointcloud) 
    # gxyz10 = create_graph_with1channel(pointcloud, dist_thresh=0.1) 
    # gxyz15 = create_graph_with1channel(pointcloud, dist_thresh=1)
    g_batch = [gxyz5, gxy, gyz, gxz]
    return g_batch 


#点云数据转为图数据 Convert point cloud to graph data
# Construct the graph with the distance threshold
def create_graph_with1channel(point, dist_thresh=0.1):
    """
    input:
        point (torch.Tensor): Point cloud to convert. Shape (B, N, 1)/(B, N, 2)/(B, N, 3).
        pointcloud (torch.Tensor): Point cloud to convert. Shape (B, N, 3).
    Returns:
        g (dgl.DGLGraph): Graph.
    """
    # compute pairwise distances
    if len(point.shape) == 1:
        point = point.view(-1, 1)
    dist_mat = torch.cdist(point, point)
    # create edge indices
    edges = torch.where(dist_mat <= dist_thresh)
    # create DGL graph
    g = dgl.graph(edges)
    g = dgl.add_self_loop(g)
    return g

# Convert point cloud to graph data
# Construct the graph with the distance threshold
# For every projection of the point cloud and the point cloud itself
def convert_pointcloud_to_dgl_graph(pointcloud):
    """
    input:
        pointcloud (torch.Tensor): Point cloud to convert. Shape (N, 3).
    Returns:
        g_batch (dgl.DGLGraph): Batched graph.
    """
    # B, N, c = pointcloud.shape
    gxy = create_graph_with1channel(pointcloud[:, :2], dist_thresh=0.005)
    gyz = create_graph_with1channel(pointcloud[:, 1:], dist_thresh=0.005)
    gxz = create_graph_with1channel(pointcloud[:, [0, 2]], dist_thresh=0.005)
    gxyz5 = create_graph_with1channel(pointcloud, dist_thresh=0.01) 
    # gxyz10 = create_graph_with1channel(pointcloud, dist_thresh=0.1) 
    # gxyz15 = create_graph_with1channel(pointcloud, dist_thresh=1)
    g_batch = [gxyz5, gxy, gyz, gxz]
    return g_batch #返回图数据和batch_size


