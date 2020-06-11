import numpy as np
import torch.nn.functional as F
import torch
from network import BrainNet

# inputs are random data with each entry taken from normal distribution

# n points in 'dim' dimensions which are labelled by by halfspace
def random_halfspace_data(dim, n, b = 0): 
    vec = 2 * (np.random.rand(dim) - 0.5) 
    pts = 2 * (np.random.rand(n, dim) - 0.5)
    labels = np.sign(np.dot(pts, vec) + b)
    return pts, labels == 1
    
# Same as random_halfspace_data. Flipped label with prob. p.
def random_halfspace_error_data(dim, n, p):
    pts, labels = random_halfspace_data(dim, n)
    for i in range(len(labels)): 
        if np.random.uniform(low = 0, high = 1) < p:
            labels[i] = 1 - labels[i]
    return pts, labels

# 1st layer: k relu with random weights. 
# 2nd layer: sum of outputs of first layer
def layer_relu_data(dim, n, k):
      
    pts = 2 * (torch.rand(n, dim) - 0.5)
    weights = 2 * (torch.rand(dim, k) - 0.5)
    
    out1 = F.relu(torch.matmul(pts, weights))
    w1 = 2 * (torch.rand(k, 2) - 0.5)
    out2 = F.softmax(torch.matmul(out1, w1))
    
    return np.array(pts), np.array(np.argmax(out2,axis=1))

def brainnet_data(n, dim, labels, num_v = 20, p = .15, cap = 5, rounds = 1): 
    pts = 2 * (torch.rand(n, dim) - 0.5)
    pts = pts.double()
    net = BrainNet(dim, labels, num_v = num_v, p = p, cap = cap, rounds = rounds, full_gd = True, outlayer_connected = True)
    
    with torch.no_grad():
        out = net(pts)
    return np.array(pts), np.array(np.argmax(out,axis=1))
