import torch
import torch.nn as nn
import torch.nn.functional as F

def gaussian_kernel(size: int, std: float, device='cuda'):
    """
    Generate a 1D Gaussian kernel for smoothing or filtering.
    """
    x = torch.linspace(-size // 2 + 1, size // 2, size, device=device)
    kernel = torch.exp(-x**2 / (2 * std**2))
    kernel = kernel / torch.sum(kernel)
    return kernel

def smooth_predictions(predictions, kernel_size=7001, std=30.0):
    """
    Apply Gaussian smoothing to 1D predictions using a convolution filter.
    """
    assert kernel_size % 2 == 1, "Kernel size must be odd."
    
    device = predictions.device
    
    gaussian_filter = gaussian_kernel(kernel_size, std, device)
    gaussian_filter = gaussian_filter.view(1, 1, kernel_size)
    
   
    if predictions.dim() == 3:
        if predictions.shape[1] == 1:  # (batch_size, 1, seq_len)
            pass
        elif predictions.shape[2] == 1:  # (batch_size, seq_len, 1)
            predictions = predictions.transpose(1, 2)
        else:
            raise ValueError("Input should have shape (batch_size, 1, seq_len) or (batch_size, seq_len, 1)")
    
    padding = kernel_size // 2
    smoothed_predictions = F.conv1d(
        predictions, 
        gaussian_filter, 
        padding=padding,
        stride=1
    )
     
    return smoothed_predictions

class PhysicsCalc(nn.Module):
    """
    Physics constraint calculation module inspired by the phenomenological Rogers–McCulloch cardiac action potential model.
    """
    
    def __init__(self, input_dim):
        super(PhysicsCalc, self).__init__()
        self.k_linear = nn.Linear(input_dim, 1)
        self.a_linear = nn.Linear(input_dim, 1)
        self.epsilon_linear = nn.Linear(input_dim, 1)
        self.gamma_linear = nn.Linear(input_dim, 1)
        
        
        nn.init.normal_(self.k_linear.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.a_linear.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.epsilon_linear.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.gamma_linear.weight, mean=0.0, std=0.02)
        
    def forward(self, v_out, b, use_cond, name):
        """
        Args: 
        - v_out: (batch_size, 1, 8000)
        - b: feature tensor (batch_size, feature_dim)
        """
        batch_size, _, seq_len = v_out.shape
        
        # 1. Physical parameter extraction
        t_out = torch.ones_like(v_out) * 1.6
        t_ = t_out

        # Extract parameter k from feature b
        k_out = self.k_linear(b)
        k_out = F.relu(k_out)
        k_out = torch.clamp(k_out, min=0.1, max=1.0)
        k = k_out

        # Extract parameter a from feature b
        a_out = self.a_linear(b)
        a_out = F.relu(a_out)
        a_out = torch.clamp(a_out, min=0.001, max=0.1)
        a = a_out

        # Extract parameters ε from feature b
        epsilon_out = self.epsilon_linear(b)
        epsilon_out = F.relu(epsilon_out)
        epsilon_out = torch.clamp(epsilon_out, min=0.005, max=0.1)
        epsilon = epsilon_out

        # Extract parameter γ from feature b
        gamma_out = self.gamma_linear(b)
        gamma_out = F.relu(gamma_out)
        gamma_out = torch.clamp(gamma_out, min=0.5, max=2.0)
        gamma = gamma_out

        # 2. Dimensional adjustment
        a_out = a_out.unsqueeze(1)  # (batch_size, 1, 1)
        k_out = k_out.unsqueeze(1)  # (batch_size, 1, 1)
        epsilon_out = epsilon_out.unsqueeze(1)  # (batch_size, 1, 1)
        gamma_out = gamma_out.unsqueeze(1)  # (batch_size, 1, 1)

        # Broadcast to the correct shape
        a_out = a_out.expand(-1, 1, seq_len)  # (batch_size, 1, seq_len)
        k_out = k_out.expand(-1, 1, seq_len)  # (batch_size, 1, seq_len)
        epsilon_out = epsilon_out.expand(-1, 1, seq_len)  # (batch_size, 1, seq_len)
        gamma_out = gamma_out.expand(-1, 1, seq_len)  # (batch_size, 1, seq_len)
        t_out = t_out.expand(-1, 1, seq_len)  # (batch_size, 1, seq_len)

        # 3. Differential calculation
        ones = torch.ones_like(v_out)
        v_out_smooth = smooth_predictions(v_out)

        # Calculate the first derivative (central difference).
        dv_l = F.pad(v_out_smooth, (1, 0), mode='constant', value=0)  # 左padding
        dv_r = F.pad(v_out_smooth, (0, 1), mode='constant', value=0)  # 右padding
        
        dv_t = (dv_r[:, :, 1:] - dv_l[:, :, :-1]) / (2 * (t_out / 7999))
        
        # Calculate the second derivative
        dv2_t = (dv_r[:, :, 1:] + dv_l[:, :, :-1] - 2 * v_out_smooth) / ((t_out / 7999) ** 2)

        # 4. Construction of physical equations
        a_v = a_out * v_out_smooth

        # Handling division by zero
        eps_0 = 1e-3
        v_out_noisy = torch.where(v_out_smooth == 0, 
                                 torch.ones_like(v_out_smooth) * eps_0, 
                                 v_out_smooth)
        inv_v = v_out_noisy ** -1
        inv_v = torch.where(torch.isnan(inv_v), torch.zeros_like(inv_v), inv_v)

        # Construct left-hand items
        term_l1 = v_out_smooth * k_out * dv_t * (ones - 2 * v_out_smooth + a_out)
        term_l2 = inv_v * (dv_t ** 2)
        term_l3 = dv2_t
        term_lt = 1000 * term_l1 + term_l2 - term_l3

        # Constructing step function and right-hand side terms
        term_r1 = v_out_smooth * (v_out_smooth - gamma_out * k_out * 1000 * (ones - v_out_smooth) * (v_out_smooth - a_out) )
        term_r2 = gamma_out * dv_t
        term_rt = epsilon_out * (term_r1 + term_r2)

        # Complete physical equation residuals
        dv_out = term_lt - term_rt

        # 5. cond
        if use_cond:
            print('cond-True')
            dv_out = torch.where(v_out <= 0, torch.zeros_like(dv_out), dv_out)

        dv_out = torch.cat([dv_out[:, :, 100:950], dv_out[:, :, 1050:-100]], dim=2)
        
        eps = 1e-5
        dv_out = torch.log(dv_out ** 2 + eps)

        return dv_out, dv_t, dv2_t, a, k * 1000, epsilon, gamma, t_, dv_l, dv_r, t_out

