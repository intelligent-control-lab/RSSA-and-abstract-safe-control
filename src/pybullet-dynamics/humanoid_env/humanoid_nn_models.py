from turtle import forward
from numpy import single
import torch as th
import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(
        self, 
        hidden_dim, num_layer, 
        use_bn=True, use_residual=True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layer = num_layer
        self.use_bn = use_bn
        self.use_residual = use_residual
        
    def generate_network(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        # self.fc = nn.ModuleList()
        # self.fc.append(nn.Linear(self.input_dim, self.hidden_dim))
        # if self.use_bn:
        #     self.fc.append(nn.BatchNorm1d(self.hidden_dim))
        # self.fc.append(nn.ReLU())
        # for _ in range(self.num_layer - 2):
        #     self.fc.append(nn.Linear(self.hidden_dim, self.hidden_dim))
        #     if self.use_bn:
        #         self.fc.append(nn.BatchNorm1d(self.hidden_dim))
        #     self.fc.append(nn.ReLU())
        # self.fc.append(nn.Linear(self.hidden_dim, self.output_dim))
        # self.fc = nn.Sequential(*self.fc)
        
        self.fc = nn.ModuleDict()
        self.fc['first_layer'] = self.single_block(input_dim=self.input_dim, output_dim=self.hidden_dim)
        for i in range(self.num_layer - 2):
            self.fc['residual_layer_' + str(i)] = self.single_block(input_dim=self.hidden_dim, output_dim=self.hidden_dim)
        self.fc['last_layer'] =  nn.Linear(self.hidden_dim, self.output_dim)
        
    def single_block(self, input_dim, output_dim):
        if self.use_bn:
            return nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.BatchNorm1d(output_dim),
                nn.ReLU(),
            )
        else:
            return nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.ReLU(),
            )
    
    def forward(self, x):
        x = self.fc['first_layer'](x)
        for i in range(self.num_layer - 2):
            if self.use_residual:
                residual = x
                x = self.fc['residual_layer_' + str(i)](x)
                x = x + residual
            else:
                x = self.fc['residual_layer_' + str(i)](x)
        x = self.fc['last_layer'](x)
        return x
        
        
class HumanoidVModel(BaseModel):
    def __init__(
        self, 
        hidden_dim=512, num_layer=4, 
        use_bn=True, use_residual=True,
        x_dim=108, u_dim=21, 
        if_preprocess=False, 
    ):
        super().__init__(hidden_dim, num_layer, use_bn, use_residual)
        self.x_dim = x_dim
        self.u_dim = u_dim
        self.if_preprocess = if_preprocess
        output_dim = 1 # ddot_pos_z
        if self.if_preprocess:  # concatenate cos(x), sin(x) to (x, u)
            input_dim = self.x_dim + self.u_dim + 2 * self.x_dim
        else:
            input_dim = self.x_dim + self.u_dim
        self.generate_network(input_dim, output_dim)
    
    def forward(self, x: th.Tensor, u: th.Tensor):
        input = th.cat((x, u), dim=-1)
        if self.if_preprocess:
            input = th.cat((input, th.cos(x), th.sin(x)), dim=-1)
        output = super().forward(input)
        return output.squeeze_()   
    
    
class HumanoidZModel(BaseModel):
    def __init__(
        self, 
        hidden_dim=512, num_layer=4, 
        use_bn=True, use_residual=True,
        x_dim=108,
        if_preprocess=False, 
    ):
        super().__init__(hidden_dim, num_layer, use_bn, use_residual)       
        self.x_dim = x_dim
        self.if_preprocess = if_preprocess
        output_dim = 2 # [pos_z, dot_pos_z]
        if self.if_preprocess:
            input_dim = 3 * self.x_dim  # concatenate cos(x), sin(x) to x
        else: 
            input_dim = self.x_dim
        self.generate_network(input_dim, output_dim)
    
    def forward(self, x: th.Tensor):
        input = x
        if self.if_preprocess:
            input = th.cat((input, th.cos(x), th.sin(x)), dim=-1)
        output = super().forward(input)
        return output      
    