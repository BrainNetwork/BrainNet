import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.utils import shuffle
from BrainNet import BrainNet

class Options: 
    def __init__(
            self, 
            use_input_rule = False,
            use_output_rule = False,
            use_graph_rule = False, 
            gd_graph_rule = False,
            gd_output_rule = False,
            gd_input = False,
            gd_output = False,
            additive_rule = True):
        self.gd_output = gd_output
        self.gd_input = gd_input
        self.use_graph_rule = use_graph_rule 
        self.use_input_rule = use_input_rule
        self.gd_graph_rule = gd_graph_rule 
        self.use_output_rule = use_output_rule
        self.gd_output_rule = gd_output_rule
        self.additive_rule = additive_rule
    
class UpdateScheme: 
    def __init__(
            self, 
            cross_entropy_loss = True, 
            mse_loss = False, 
            update_misclassified = True, 
            update_all_edges = False):
        self.cross_entropy_loss = cross_entropy_loss 
        self.mse_loss = mse_loss 
        self.update_misclassified = update_misclassified 
        self.update_all_edges = update_all_edges

# do not instantiate this class directly. Use something from the network.py file. ex. LocalNet.
class LocalNetBase(BrainNet):
    '''
        n = # of features
        m = # of possible labels
        num_v = # of nodes in graph
        p = probability that an edge exists in the graph
        cap = choose top 'cap' nodes which fire
        rounds = # of times the graph 'fires'           
    '''
    def __init__(self, n, m, num_v, p, cap, rounds, options = Options(), update_scheme = UpdateScheme()):
        super().__init__(   n = n, 
                            m = m, 
                            num_v = num_v, 
                            p = p, 
                            cap = cap, 
                            rounds = rounds, 
                            gd_input = options.gd_input, 
                            gd_output = options.gd_output)

        self.options = options
        self.update_scheme = update_scheme
        
        self.single_rules = torch.randn((rounds, 4))

        self.rule = torch.randn((2**(rounds+1))**2)
        self.input_rule = torch.randn(2**(rounds + 1))
        self.output_rule = torch.zeros((2**(rounds + 1)) * 2)
        self.step_sz = 0.01

    def get_rnn_rule(self): 
        return self.rule.clone().detach().view(2 ** (self.rounds + 1), 2 ** (self.rounds + 1))
    
    def get_output_rule(self): 
        return self.output_rule.clone().detach().view(2 ** (self.rounds + 1), 2)

    def set_rnn_rule(self, rule):
        self.rule = torch.tensor(rule).flatten().double()

    def set_output_rule(self, rule): 
        self.output_rule = rule.clone().detach().flatten().double()

    def copy_graph(self, net, input_layer = False, graph = False, output_layer = False):
        if input_layer: 
            self.input_layer = net.input_layer
        if graph:
            self.graph = net.graph
        if output_layer: 
            self.output_layer = net.output_layer
            
    def update_weights(self, probs, label):
        prob = probs[0]
        prediction = torch.argmax(prob)

        if self.update_scheme.update_misclassified and prediction == label: 
            return 

        def mult(a, b): 
            a *= 1 + self.step_sz * b
        def add(a, b): 
            a += self.step_sz * b

        if self.options.additive_rule:
            update_func = add 
        else: 
            update_func = mult 


        # update graph weights
        if self.options.use_graph_rule: 
            a1 = self.activated.repeat(1,self.num_v).view(-1, self.num_v)
            a2 = self.activated.view(-1, 1).repeat(1, self.num_v).view(self.num_v, self.num_v)
            act = (2 ** (self.rounds + 1)) * a1 + a2
                 
            act *= self.graph
            act = act.long()

            update_func(self.graph_weights, self.rule[act])

        # update input weights 
        if self.options.use_input_rule:
            input_act = self.activated.repeat(1, self.n).view(-1, self.n)
            input_act *= self.input_layer
            input_act = input_act.long() 

            update_func(self.input_weights, self.input_rule[input_act])

        #update output weights
        if self.options.use_output_rule:
            if self.update_scheme.update_all_edges:
                update_func(self.output_weights[label], self.output_rule[2 * self.activated.long()])
                for j in range(len(prob)):
                    if j != label:
                        update_func(self.output_weights[j], self.output_rule[2 * self.activated.long() + 1])
            else:
                update_func(self.output_weights[prediction], self.output_rule[2 * self.activated.long() + 1])
                update_func(self.output_weights[label], self.output_rule[2 * self.activated.long()])

    def forward(self, inputs, labels, epochs, batch, continue_ = False):
        if continue_ == False:
            self.reset_weights(additive = self.options.additive_rule, input_rule = self.options.use_input_rule, output_rule = self.options.use_output_rule)
            self.double()

        if self.update_scheme.mse_loss: 
            criterion = nn.MSELoss()
        elif self.update_scheme.cross_entropy_loss:
            criterion = nn.CrossEntropyLoss()

        self.output_updates = torch.zeros(self.m, self.num_v)
        
        for epoch in range(1, epochs + 1):
            for x,ell in zip(inputs,labels):
                outputs = self.forward_pass(x.unsqueeze(0))
                self.update_weights(outputs, ell)       
        outputs = self.forward_pass(inputs)
        
        if self.update_scheme.mse_loss:
            target = torch.zeros_like(outputs)
            target = target.scatter_(1, labels.unsqueeze(1), 1)
            loss = criterion(outputs, target) 
        else: 
            loss = criterion(outputs, labels) 

        return loss

