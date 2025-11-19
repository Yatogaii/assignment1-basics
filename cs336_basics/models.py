import torch
from math import sqrt
from einops import einsum,rearrange, reduce, repeat
from torch.nn.modules.module import Module

class LinearModule(torch.nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype


        sigma=2/(in_features+out_features)
        std_dev = sqrt(sigma)
        self.weights = torch.empty((out_features, in_features), device=device, dtype=dtype)
        
        torch.nn.init.trunc_normal_(
            tensor=self.weights,
            mean=0,
            std=std_dev,
            a=-3*std_dev,
            b=3*std_dev,
        )
        
        self.weights = torch.nn.Parameter(self.weights)
        assert self.weights.size() == torch.Size([out_features, in_features])
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return einsum(self.weights, x, "d_out d_in, ... d_in -> ... d_out")
    
class EmbeddingModel(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        """
        Parameter:
            num_embeddings = vocab_size
            embedding_dim = model_dim 
        """
        super().__init__()

        weights = torch.empty(num_embeddings, embedding_dim)
        torch.nn.init.trunc_normal_(
            tensor=weights,
            mean=0,
            std=1,
            a=-3,
            b=3,
        )
        self.weights = weights

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weights[token_ids]

class RMSNormModel(torch.nn.Module):
    def __init__(self, d_model: int, eps:float =1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype

        self.gain = torch.nn.Parameter(torch.ones(d_model))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor: # x -> (batch_size, sequence_length, d_model)
        in_dtpye = x.dtype
        x = x.to(torch.float32)

        exp_sum = 0

        rms_a = torch.sqrt(reduce(x**2, "... d -> ... 1", "mean") + self.eps)
        # Equal to: rms_a = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) # rms_a -> (batch_size, sequence_length, 1)
        x /= rms_a # PyTorch will broadcase (batch_size,sequence_length, 1) to div x

        x *= self.gain

        return x.to(in_dtpye)

class SwiGLU(torch.nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = torch.nn.Parameter(torch.rand((d_ff, d_model)))
        self.w2 = torch.nn.Parameter(torch.rand((d_model,d_ff)))
        self.w3 = torch.nn.Parameter(torch.rand((d_ff, d_model)))

    def forward(self, x: torch.Tensor) -> torch.Tensor: # x -> (batch_size, seq_length, d_model)
        silu_res = run_SiLU(einsum(self.w1, x, "d_ff d_model, ... d_model -> ... d_ff"))
        glu_res = einsum(self.w3, x, "d_ff d_model, ... d_model -> ... d_ff")

        return einsum(self.w2, silu_res*glu_res, "d_model d_ff, ... d_ff -> ... d_model")
    
    
def run_SiLU(in_features: torch.Tensor):
    return in_features * torch.sigmoid(in_features)

class RoPEModel(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        half = d_k // 2
        dim_idx = torch.arange(half)
        exponent = (2* dim_idx).float() / float(d_k) # (2k-2)/d, shape=(half,)
        self.freq = 1.0 / (theta ** exponent) # shape = (half,)

        positions = torch.arange(max_seq_len)
        angles = einsum(self.freq, positions,"half, seq_len -> seq_len half")
        cos = angles.cos()
        sin = angles.sin()
        
        self.register_buffer("cos_table", cos, persistent=False)
        self.register_buffer("sin_table", sin, persistent=False)
        

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        x: (..., seq_len, d_k)
        token_positions: (..., seq_len)
        """
        x_even = x[..., 0: : 2]
        x_odd = x[..., 1: : 2]
        
        cos_pos = self.cos_table[token_positions]
        sin_pos = self.sin_table[token_positions]

        x_rot_even = x_even * cos_pos - x_odd * sin_pos
        x_rot_odd = x_even * sin_pos + x_odd * cos_pos

        x_rot_2d = torch.cat([x_rot_even, x_rot_odd], dim = -1)

        return rearrange([x_rot_even, x_rot_odd], 'two ... half -> ... (half two)')