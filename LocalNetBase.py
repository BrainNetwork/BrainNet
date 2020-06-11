import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.utils import shuffle
from BrainNet import BrainNet

class Options: 
    def __init__(self, 
                 num_graphs = 1, 
                 rule = None, 
                 use_input_rule = False,
                 use_bias_rule = False, 
                 use_output_rule = False,
                 gd_graph_rule = True,
                 gd_output_rule = False,
                 additive_rule = True):
        self.rule = rule
        self.num_graphs = num_graphs
        self.use_input_rule = use_input_rule
        self.use_bias_rule = use_bias_rule
        self.gd_graph_rule = gd_graph_rule 
        self.use_output_rule = use_output_rule
        self.gd_output_rule = gd_output_rule
        self.additive_rule = additive_rule
    

# do not instantiate this class directly. Use something from the BrainNetVariants file. ex. LocalNet.
class LocalNetBase(nn.Module):
    '''
        n = # of features
        m = # of possible labels
        num_v = # of nodes in graph
        p = probability that an edge exists in the graph
        cap = choose top 'cap' nodes which fire
        rounds = # of times the graph 'fires'
        num_graphs = # of graphs we train on. Same rule across all graphs, but separate GD on input layer
        rules: if None, we apply GD on rule and on input layer. 
                If rules is specified, we use that fixed rule, and only run GD on input layer
        use_bias_rule/use_output_rule: True/False whether to update network bias and output weights with a fixed rule. 
                If False, bias remains 0, and output weights are all equal to 1.
            
    '''
    def __init__(self, n, m, num_v, p, cap, rounds, options = Options()):
        super().__init__()
        self.cap = cap
        self.rounds = rounds

        self.n = n
        self.m = m
        self.p = p
        self.num_v = num_v
        self.options = options
        
        self.cur_batch = 0
        
        self.accuracies = []
        
        self.network = []
        for _ in range(self.options.num_graphs):
            self.network.append(BrainNet(n, 
                                         m, 
                                         num_v = num_v, 
                                         p = p, 
                                         cap = cap, 
                                         rounds = rounds, 
                                         input_rule = options.use_input_rule))
        
        self.rule = torch.randn((2**(rounds+1))**2)
        self.bias_rule = torch.randn(2**(rounds + 1))
        self.input_rule = torch.randn(2**(rounds + 1))
        self.output_rule = torch.zeros((2**(rounds + 1)) * 2)
        self.step_sz = 0.01

    def copy_rule(self, net): 
        self.rule = net.rule.data
        self.bias_rule = net.rule.bias_rule.data
        if self.options.use_input_rule:
            self.input_rule = net.rule.input_rule.data


    def copy_graph(self, net, input_layer = False, graph = False, output_layer = False):
        if input_layer: 
            self.network[0].input_layer = net.network[0].input_layer
        if graph:
            self.network[0].graph = net.network[0].graph
        if output_layer: 
            self.network[0].output_layer = net.network[0].output_layer
            
    '''
        Given input 'x' and corresponding correct label 'label'.
        Updates weights of each graph 
    '''
    def update_weights(self, probs, label):
        for i in range(self.options.num_graphs):
            prob = probs[i][0]
            prediction = torch.argmax(prob)
            
            if prediction != label: 
                output_rule = self.output_rule
            else:
=               return

            a1 = self.network[i].activated.repeat(1,self.num_v).view(-1, self.num_v)
            a2 = self.network[i].activated.view(-1, 1).repeat(1, self.num_v).view(self.num_v, self.num_v)
            act = (2 ** (self.rounds + 1)) * a1 + a2
                 
            act *= self.network[i].graph
            act = act.long()
 
            # update graph weights
            if self.options.additive_rule:
                self.network[i].graph_weights += self.step_sz * self.rule[act]
            else: 
                self.network[i].graph_weights *= (1 + self.step_sz * self.rule[act])

            # update input weights 
            if self.options.use_input_rule:
                input_act = self.network[i].activated.repeat(1, self.n).view(-1, self.n)
                input_act *= self.network[i].input_layer
                input_act = input_act.long() 

                self.network[i].input_weights += self.step_sz * self.input_rule[input_act]

            # update bias                 
            if self.options.use_bias_rule:
                if self.options.additive_rule:
                    self.network[i].graph_bias += self.step_sz * self.bias_rule[self.network[i].activated.long()]
                else: 
                    self.network[i].graph_bias *= (1 + self.step_sz * self.bias_rule[self.network[i].activated.long()])

            #update output weights
            if self.options.use_output_rule:
                if self.options.additive_rule:
                    self.network[0].output_weights[1 - label] += self.step_sz * self.output_rule[2 * self.network[i].activated.long() + 1]
                    self.network[0].output_weights[label] += self.step_sz * self.output_rule[2 * self.network[i].activated.long()]    
                else: 
                    self.network[0].output_weights[prediction] *= (1 + self.step_sz * self.output_rule[2 * self.network[i].activated.long() + 1])
                    self.network[0].output_weights[label] *= (1 + self.step_sz * self.output_rule[2 * self.network[i].activated.long()])    
    '''
        Given training data X and labels y
        Run for 'epochs'
        For each x in X, update weights using current rule.
        print error at each batch.
    '''
    def forward(self, inputs, labels, epochs, batch, continue_ = False):
        if continue_ == False:
          for i in range(self.options.num_graphs):
              self.network[i].reset_weights(additive = self.options.additive_rule, input_rule = self.options.use_input_rule, output_rule = self.options.use_output_rule)
              self.network[i].double()
        torch.set_printoptions(precision=3)

        sz = len(inputs)
        num_batches = sz//batch

        #criterion = nn.MSELoss()
        criterion = nn.CrossEntropyLoss()

        self.output_updates = torch.zeros(self.m, self.num_v)
        
        running_loss = []
        for epoch in range(1, epochs + 1):
            for x,ell in zip(inputs,labels):
                outputs = self.network[0](x.unsqueeze(0))
                self.update_weights([outputs], ell)
        
        cur_losses = torch.zeros(self.options.num_graphs)
        for i in range(self.options.num_graphs):
            outputs = self.network[i](inputs)

            # For MSE Loss
            #target = torch.zeros_like(outputs)
            #target = target.scatter_(1, labels.unsqueeze(1), 1)
            #loss = criterion(outputs, target) 

            loss = criterion(outputs, labels) 
            cur_losses[i] = loss

        return torch.mean(cur_losses)

    '''
     Evaluate data X with correct labels y using model
    '''
    def evaluate(self, X, y, model):
        
        ac = [0] * self.m
        total = [0] * self.m
        with torch.no_grad():

            correct = 0

            outputs = model(X)
            b = np.argmax(outputs, axis = 1)

            for i in range(len(b)):
                total[y[i]] += 1
                if b[i] == y[i]:
                    ac[y[i]] += 1

            correct = torch.sum(y == b)
            acc = correct*1.0/ sum(total)
            print(correct, sum(total), acc)
            
            self.accuracies.append([ac[i] / total[i] for i in range(len(ac))])
            
            for i in range(self.m):
                print("Acc of class", i, ":{0:.4f}".format(ac[i] / total[i]))
        return acc
