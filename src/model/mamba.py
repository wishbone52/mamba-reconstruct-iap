import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba2
from .RM_physics_modules import PhysicsCalc

class CBR(nn.Module):
    """
    A Conv-BatchNorm-ReLU block used as a basic building block for 1D convolutional feature extraction.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(CBR, self).__init__()
        
        padding = (dilation * (kernel_size - 1)) // 2  
        
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, padding=padding, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU()
            
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

class Mamba_model(nn.Module):
    """
    Mamba-based model for iap prediction with optional physics-guided auxiliary output.
    """
    
    def __init__(self, seq_len=8000, n_layers=4,dim=512,d_state=32, expand=2, physics=True,use_cond=False):
        """
        Initialize the Mamba model.

        Args:
            seq_len (int, optional): Length of input sequence. 
            n_layers (int, optional): Number of Mamba2 layers. 
            dim (int, optional): Model dimension. 
            d_state (int, optional): State dimension for Mamba2. 
            expand (int, optional): Expansion factor for Mamba2. 
            physics (bool, optional): Whether to enable physics head. 
            use_cond (bool, optional): Whether to use conditioning in physics module. 
        """
        super().__init__()

        self.physics = physics
        self.use_cond = use_cond

        self.in_conv1 = CBR(1, dim // 4, kernel_size=1)
        self.in_conv2 = CBR(1, dim // 4, kernel_size=5)
        self.in_conv3 = CBR(1, dim // 4, kernel_size=9)
        self.in_conv4 = CBR(1, dim // 4, kernel_size=13)
        
        self.input_norm = nn.LayerNorm(dim)

        self.mamba_layers = nn.ModuleList([
            Mamba2(d_model=dim, d_state=d_state, d_conv=4, expand=expand, layer_idx=i, rmsnorm=True) 
            for i in range(n_layers)
        ])


        self.out_proj = nn.Linear(dim, 1)
        self.tanh = nn.Tanh()

        if self.physics:
            # v_out encoder
            self.v_out_enc = nn.Sequential(
                CBR(in_channels=1, out_channels=32, kernel_size=11, stride=5, dilation=1),
                CBR(in_channels=32, out_channels=64, kernel_size=11, stride=5, dilation=1),
                CBR(in_channels=64, out_channels=96, kernel_size=11, stride=5, dilation=1),
                CBR(in_channels=96, out_channels=128, kernel_size=11, stride=5, dilation=1),
                nn.AdaptiveAvgPool1d(1),
            )
            
            # b encoder
            self.b_enc = nn.Sequential(
                nn.Flatten(),
                nn.BatchNorm1d(num_features=128),
                nn.Linear(128, 8),
                nn.ReLU(inplace=True),
            )
            
            # Physics calculation module
            self.physics_calc_module = PhysicsCalc(input_dim=8)  
    
    def forward(self,x):

        x = x.unsqueeze(1)
        x = torch.cat([self.in_conv1(x), self.in_conv2(x), self.in_conv3(x), self.in_conv4(x)],dim=1)

        x = x.transpose(1,2)
        y = self.input_norm(x)

        for mamba_layer in self.mamba_layers:
            y = mamba_layer(y)  

        out = self.tanh(self.out_proj(y))
        out = out.squeeze(-1)

        dv_out = None
        if self.physics and self.training:
            v_out = out.unsqueeze(1)
            bk = self.v_out_enc(v_out).squeeze(-1)
            b = self.b_enc(bk)
            
            dv_out = self.physics_calc_module(v_out,b,self.use_cond,'noq')[0]
            dv_out = dv_out.squeeze(1)

        return out, dv_out

        

