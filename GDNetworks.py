from torch import nn
import torch.nn.functional as F

class Regression(nn.Module): 
    def __init__(self, input_sz, hidden_sz, out_sz):
        super().__init__()
        self.hidden = nn.Linear(input_sz, hidden_sz)   # hidden layer
        self.predict = nn.Linear(hidden_sz, out_sz)   # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))      # activation function for hidden layer
        x = self.predict(x)             # linear output
        return x


class SingleLayer(nn.Module):
    def __init__(self, input_sz, m):
        super().__init__()
        
        self.m = m # num. of classes
        self.layer = nn.Linear(input_sz, self.m)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):        
        return self.softmax(self.layer(x))
    

class ReluNetwork(nn.Module):
    def __init__(self, input_sz, width):
        super().__init__()
        
        self.m = 2 # num. of classes
        self.width = width 
        
        self.layer1 = nn.Linear(input_sz, width)
        self.layer2 = nn.Linear(width, self.m)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        out1 = F.relu(self.layer1(x))
        out2 = self.softmax(self.layer2(out1))
        return out2
    
class DeepNetwork(nn.Module):
    def __init__(self, input_sz, width, depth):
        super().__init__()
        
        self.m = 2 # num. of classes
        self.width = width
        self.depth = depth
        
        self.layers = []
        self.layers.append(nn.Linear(input_sz, width))
        for i in range(depth - 1): 
            self.layers.append(nn.Linear(width, width))        
        self.layers.append(nn.Linear(width, self.m))
        
        self.layers = nn.ModuleList(self.layers)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        for i in range(self.depth): 
            x = F.relu(self.layers[i](x))
            
        return self.softmax(x)
