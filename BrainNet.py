import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class BrainNet(nn.Module):
    # n = # of features
    # m = # of labels/classes
    # num_v = number of verices in graph
    # p = prob. of an edge in graph
    def __init__(self, n = 10, m = 2, num_v = 100, p = 0.5, cap = 50, rounds = 0, input_rule = False, full_gd = False, outlayer_connected = True, gd_input = True, gd_output = False):
        super().__init__()

        self.cap = cap
        self.rounds = rounds

        self.n = n
        self.m = m
        self.num_v = num_v
            
        self.activated = torch.zeros(self.num_v).long()

        out_p = p
        if outlayer_connected: 
            out_p = 1

        # defines the graph
        self.input_layer = self.generate_input_layer(n, num_v, p)
        self.graph = self.generate_random_graph(num_v, p)
        self.output_layer = self.generate_output_layer(m, num_v, out_p)
        
        if full_gd: 
            self.graph_weights = nn.Parameter((torch.rand(num_v, num_v)) - 0.5)
            self.output_weights = nn.Parameter((torch.rand(m, num_v)) - 0.5)
            self.graph_bias = nn.Parameter((torch.rand(num_v)) - 0.5)
            self.output_bias = nn.Parameter((torch.rand(m)) - 0.5)
            self.input_weights = nn.Parameter((torch.rand(num_v, n)) - 0.5)
        else: 
            if gd_input:
                self.input_weights = nn.Parameter((torch.randn(num_v, n)))
            else:
                self.input_weights = torch.randn(num_v, n) 
            
            if gd_output:
                self.output_weights = nn.Parameter(torch.randn(m, num_v))
            else: 
                self.output_weights = torch.randn(m, num_v)
            
            self.graph_weights = torch.randn(num_v, num_v)
            self.graph_bias = torch.zeros(num_v)
            self.output_bias = torch.zeros(m)
        
    # Define the forward pass here.
    def forward_pass(self, x):
        self.activated_rounds = []
        self.activated = torch.zeros(self.num_v)

        self.input = x.clone()
        x = self.feed_input(x)
        for _ in range(self.rounds):
            x = self.step_once_graph(x).T
        out = self.get_output(x.T)

        return out

    '''
        The weights are reset at the beginning of each outer level epoch.
    '''
    def reset_weights(self, additive = False, input_rule = False, output_rule = False):
        if additive:
            self.graph_weights = torch.zeros(self.num_v, self.num_v)
            self.graph_bias = torch.zeros(self.num_v)
        else: 
            self.graph_weights = torch.ones(self.num_v, self.num_v)
            self.graph_bias = torch.ones(self.num_v)
        
        if input_rule:
            self.input_weights = torch.zeros(self.num_v, self.n)             
        if output_rule:
            if additive: 
              self.output_weights = torch.zeros(self.m, self.num_v)
            else: 
              self.output_weights = torch.ones(self.m, self.num_v)

    def random_weights(self):
        self.graph_weights = torch.rand(self.num_v, self.num_v)
        self.graph_bias = torch.rand(self.num_v)
        self.output_weights = torch.rand(self.m, self.num_v)
        
    def feed_input(self, x):
        res = torch.mm((self.input_weights * self.input_layer), x.T)
        res = res + self.graph_bias[:, None]
        res = F.relu(res)
        return self.get_cap(res.T).T

    def get_output(self, x):
        a =  torch.mm(x, (self.output_layer * self.output_weights).T)
        res = a + self.output_bias
        #return res
        return F.softmax(res, dim=1)

    def step_once_graph(self, x):
        input_ = torch.mm(self.input_weights * self.input_layer, self.input.T)

        res = torch.mm((self.graph * self.graph_weights), (x + input_)).T + self.graph_bias

        r = F.relu(res)
        return self.get_cap(r)

    # only top nodes will fire.
    def get_cap(self, x):
        
        if self.cap != -1:
            topk, indices = torch.topk(x, self.cap, axis = 1)
            res = torch.zeros_like(x)
            res = res.scatter(1, indices, topk)

            try:
                activated = torch.zeros_like(x).scatter(1, indices, 1)
                self.activated = 2 * self.activated + activated.squeeze()
                self.activated_rounds.append(activated.squeeze())
            except Exception as e:
                pass
            
            return res
        else: # don't use cap. activate what is greater than 0 after ReLu 
            try: 
              indices_ = (x[0] > 0).nonzero().T
              activated_ = torch.zeros_like(x).scatter(1, indices_, 1)
              self.activated = 2 * self.activated + activated_.squeeze()
            except Exception as e:
                pass

            return x
        

    # Generate random unweighted directed graph with n nodes
    # each edge appears with probability p
    def generate_random_graph(self, n, p):
        adj = torch.rand(n, n).double()
        adj[adj <= 1-p] = 0
        adj[adj > 1-p] = 1

        for i in range(n):
            adj[i][i] = 0

        return adj

    # n input variables.
    # Graph has m nodes
    def generate_input_layer(self, n, m, p):
        adj = torch.rand(m, n).double()

        adj[adj <= 1-p] = 0
        adj[adj > 1-p] = 1

        return adj

    # n labels
    # graph has m nodes
    def generate_output_layer(self, n, m, p):
        adj = torch.rand(n, m).double()
        
        adj[adj <= 1-p] = 0
        adj[adj > 1-p] = 1

        return adj


from torch.autograd import Variable

class BrainNetSequence(BrainNet): 
    def __init__(self, n, m, num_v, p, cap, rounds, input_rule = False, full_gd = False, outlayer_connected = True, gd_output_only = False, gd_input = True, gd_output = False):
        super().__init__(n, m, num_v, p, cap, rounds, input_rule=input_rule, full_gd=full_gd, outlayer_connected=True, gd_output_only=gd_output_only, gd_input=gd_input, gd_output=gd_output)
        self.vocab_size = n;

    def forward(self, inp, hidden): 
        self.activated = torch.zeros(self.num_v)

        inp = inp.double()
        input_ = torch.mv(self.input_weights * self.input_layer, inp)        

        for _ in range(self.rounds):
            hidden_ = torch.mv((self.graph * self.graph_weights), hidden.double())
            hidden = F.relu(input_ + hidden_ + self.graph_bias)
            hidden = self.get_cap(hidden)

        output = torch.mv(self.output_layer * self.output_weights, hidden) + self.output_bias
        return F.softmax(output, dim=0), hidden
    
            
    def initHidden(self):
        return Variable(torch.zeros(self.num_v))

    # only top nodes will fire.
    def get_cap(self, x):
        topk, indices = torch.topk(x, self.cap, axis = 0)
        res = torch.zeros_like(x)
        res = res.scatter(0, indices, topk)

        try:
            activated = torch.zeros_like(x).scatter(1, indices, 1)
            self.activated = 2 * self.activated + activated.squeeze()
        except Exception as e:
            pass

        return res
