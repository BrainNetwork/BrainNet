import torch
import torch.nn as nn
from LocalNetBase import * 

class LocalNet(LocalNetBase): 
    def __init__(self, n, m, num_v, p, cap, rounds, options = Options(), update_scheme = UpdateScheme()):
        super().__init__(n, m, num_v, p, cap, rounds, options = options, update_scheme = update_scheme)

        if self.options.gd_graph_rule:
            self.rule = nn.Parameter(self.rule)
        if self.options.use_input_rule:
            self.input_rule = nn.Parameter(self.input_rule)
        if self.options.gd_output_rule: 
            self.output_rule = nn.Parameter(self.output_rule)


