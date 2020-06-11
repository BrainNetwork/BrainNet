import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from LocalNetBase import * 
from BrainNet import BrainNetSequence
from GDNetworks import Regression

class LocalNet(LocalNetBase): 
    def __init__(self, n, m, num_v, p, cap, rounds, options = Options()):
        super().__init__(n, m, num_v, p, cap, rounds, options = options)
    
        if options.rule is None:
            if self.options.gd_graph_rule:
                self.rule = nn.Parameter(self.rule)
            if self.options.use_bias_rule:
              self.bias_rule = nn.Parameter(self.bias_rule)
            if self.options.use_input_rule:
                self.input_rule = nn.Parameter(self.input_rule)
            if self.options.gd_output_rule: 
                self.output_rule = nn.Parameter(self.output_rule)
        else: 
            self.rule = self.options.rule.rule.data
            if self.options.use_bias_rule:
              self.bias_rule = self.options.rule.bias_rule.data
            if self.options.use_input_rule:
                self.input_rule = self.options.rule.input_rule.data  

# use 2x2 rule for every pair of consecutive rounds. 
class LocalSingleRules(LocalNetBase):
    def __init__(self, n, m, num_v, p, cap, rounds, single_rules, output_rule):
        super().__init__(   n, 
                            m, 
                            num_v, 
                            p, 
                            cap, 
                            rounds, 
                            options = Options(
                                gd_graph_rule = False,
                                gd_output_rule = False,
                                use_output_rule = True,
                                gd_input=False, 
                                additive_rule=False
                            ))

        self.first_rule = torch.nn.Parameter(torch.randn(4))
        self.single_rules = single_rules
        self.output_rule = output_rule 

    def update_weights(self, probs, label):
        for i in range(self.options.num_graphs):
            prob = probs[i][0]
            prediction = torch.argmax(prob)
             
            if prediction == label: return  
               
            # update outputs 
            if self.options.additive_rule:
                self.network[0].output_weights[1 - label] += self.step_sz * self.output_rule[2 * self.network[i].activated_rounds[-1].long() + 1]
                self.network[0].output_weights[label] += self.step_sz * self.output_rule[2 * self.network[i].activated_rounds[-1].long()]    
            else: 
                self.network[0].output_weights[1 - label] *= (1 + self.step_sz * self.output_rule[2 * self.network[i].activated_rounds[-1].long() + 1])
                self.network[0].output_weights[label] *= (1 + self.step_sz * self.output_rule[2 * self.network[i].activated_rounds[-1].long()])

            # update graph weights
            for k in range(self.rounds):
                a1 = self.network[i].activated_rounds[k].repeat(1,self.num_v).view(-1, self.num_v)
                a2 = self.network[i].activated_rounds[k].view(-1, 1).repeat(1, self.num_v).view(self.num_v, self.num_v)
                act = 2 * a1 + a2
    
                act *= self.network[i].graph
                act = act.long()

                if k == 0: 
                    if self.options.additive_rule:
                        self.network[i].graph_weights += self.step_sz * self.first_rule[act]
                    else: 
                        self.network[i].graph_weights *= (1 + self.step_sz * self.first_rule[act])
                else: 
                    if self.options.additive_rule:
                        self.network[i].graph_weights += self.step_sz * self.single_rules[k - 1][act]
                    else:
                        self.network[i].graph_weights *= (1 + self.step_sz * self.single_rules[k - 1][act])




class HebbianRule(LocalNetBase): 
    # step = amount we change weight of edge per update
    def __init__(self, n, m, num_v, p, cap, rounds, step = 1):
        super().__init__(n, m, num_v, p, cap, rounds, 
                        options = Options(
                            num_graphs = 1, 
                            rule = None, 
                            use_input_rule = False,
                            use_bias_rule = False, 
                            use_output_rule = False, 
                            additive_rule = True,
                        ))

        n = 2 ** (rounds + 1)
        self.rule = torch.zeros(n ** 2)
        
        for i in range(n): 
            for k in range(n): 
                bits_i = self.convert_to_bits(i)
                bits_k = self.convert_to_bits(k)

                for j in range(rounds): 
                    if bits_i[j] == 1 and bits_k[j + 1] == 1: 
                        self.rule[i * n + k] += step

    def convert_to_bits(self, x): 
        arr = []
        for i in range(self.rounds + 1): 
            arr.append(x % 2)
            x //= 2
        return arr[::-1] # activation of first round = most significant bit = last entry of array.


# Approximate Rule with a neural net. 
class LocalNetRuleModel(LocalNetBase): 
    def __init__(self, n, m, num_v, p, cap, rounds, options = Options()):
        super().__init__(n, m, num_v, p, cap, rounds, options = options)
            
        self.rule_sz = 4 ** (rounds + 1)
        self.rule_bits = 2 * (rounds + 1) # = log (rule_sz)
        self.conv = [None] * self.rule_sz

        self.edges = (self.network[0].graph > 0).nonzero()
        self.output_edges = (self.network[0].output_layer > 0).nonzero()
        self.input_edges = (self.network[0].input_layer > 0).nonzero()

        # We train this model 
        self.rule_model = Regression(self.rule_bits, 10, 1) 
        self.rule_model.double()

    def update_weights(self, x, label):
        self.generate_rule()
        super().update_weights(x, label)

    def forward(self, X, y, epochs, batch):
        self.rule = torch.zeros(self.rule_sz)
        return super().forward(X, y, epochs, batch)

    def generate_rule(self):
        for activation_seq in range(self.rule_sz): 
            seq = self.convert_to_bits(activation_seq)
            self.rule[activation_seq] = self.rule_model(seq)
            
    def convert_to_bits(self, x): 
        if self.conv[x] is None: 
            xx = x
            arr = []
            for i in range(self.rule_bits): 
                arr.append(x % 2)
                x //= 2
            self.conv[xx] = torch.tensor(arr).double()
            return self.conv[x]
        else: 
            return self.conv[x]