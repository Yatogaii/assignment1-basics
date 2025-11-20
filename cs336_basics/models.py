import torch
from math import sqrt
from einops import einsum,rearrange, reduce, repeat
from torch.nn.modules.module import Module
from .nn_utils import run_softmax
class LinearModule(torch.nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype


        sigma=2/(in_features+out_features)
        std_dev = sqrt(sigma)
        self.weight = torch.empty((out_features, in_features), device=device, dtype=dtype)
        
        torch.nn.init.trunc_normal_(
            tensor=self.weight,
            mean=0,
            std=std_dev,
            a=-3*std_dev,
            b=3*std_dev,
        )
        
        self.weight = torch.nn.Parameter(self.weight)
        assert self.weight.size() == torch.Size([out_features, in_features])
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return einsum(self.weight, x, "d_out d_in, ... d_in -> ... d_out")
    
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

        self.weight = torch.nn.Parameter(torch.ones(d_model))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor: # x -> (batch_size, sequence_length, d_model)
        in_dtpye = x.dtype
        x = x.to(torch.float32)

        exp_sum = 0

        rms_a = torch.sqrt(reduce(x**2, "... d -> ... 1", "mean") + self.eps)
        # Equal to: rms_a = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) # rms_a -> (batch_size, sequence_length, 1)
        x /= rms_a # PyTorch will broadcase (batch_size,sequence_length, 1) to div x

        x *= self.weight

        return x.to(in_dtpye)

class SwiGLU(torch.nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()
        self.w1 = LinearModule(d_model,d_ff, device=device, dtype=dtype)
        self.w2 = LinearModule(d_ff,d_model, device=device, dtype=dtype)
        self.w3 = LinearModule(d_model,d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor: # x -> (batch_size, seq_length, d_model)
        a1 = self.w1(x)
        silu = run_SiLU(a1)
        return self.w2(silu * self.w3(x))
    
    
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
        assert x.size()[-1] % 2 == 0
        x_even = x[..., 0:: 2]
        x_odd = x[..., 1:: 2]
        
        cos_pos = self.cos_table[token_positions]
        sin_pos = self.sin_table[token_positions]

        x_rot_even = x_even * cos_pos - x_odd * sin_pos
        x_rot_odd = x_even * sin_pos + x_odd * cos_pos

        x_rot_2d = torch.cat([x_rot_even, x_rot_odd], dim = -1)

        return rearrange([x_rot_even, x_rot_odd], 'two ... half -> ... (half two)')
    
def run_scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask=None) -> torch.Tensor:
    """
    Params:
        Q, K -> (batch_size, ..., seq_len, d_k) 
        V -> (batch_size, ..., seq_len, d_v)
        Mask(opt) -> Bool(seq_len, seql_len)
    Return:
        tensor -> (batch_size, ..., d_v)
    """
    # 计算相关性
    qk_similarity = einsum(Q, K, "... q_len d_k, ... k_len d_k -> ... q_len k_len")
    # 缩放
    d_k = Q.size(dim=-1)
    scaled_score = qk_similarity / sqrt(d_k)
    # Masking
    if mask is not None:
        scaled_score = scaled_score.masked_fill(~mask, float(-1e9)) # 使用 ~ 来取反整个 mask，因为masked_fill默认是填充True的值。

    softmax_score = run_softmax(scaled_score, dimension=-1)

    return einsum(softmax_score, V, "... q_len k_len, ... k_len d_v -> ... q_len d_v")

class MultiHeadAttentionModel(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, device=None, dtype=None):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads

        assert d_model % num_heads == 0
        self.d_head = self.d_v = d_model // num_heads

        self.wqkv =LinearModule(d_model, d_model*3, device=device, dtype=dtype)

        self.output_proj = LinearModule(d_model, d_model, device=device,dtype=dtype)
        

    def forward(self, x:torch.Tensor, rope: RoPEModel|None=None, positions=None) -> torch.Tensor:
        seq_len = x.size()[-2]
        QKV = self.wqkv.forward(x)

        Q, K, V = QKV.split(self.d_model, dim=-1)
        Q = rearrange(Q, "b s (h d) -> b h s d", h=self.num_heads)
        K = rearrange(K, "b s (h d) -> b h s d", h=self.num_heads)
        V = rearrange(V, "b s (h d) -> b h s d", h=self.num_heads)

        if positions == None:
            positions = torch.arange(seq_len, device=x.device)

        if rope is not None:
            Q = rope.forward(Q,positions)
            K = rope.forward(K,positions)

        mask = torch.ones([seq_len, seq_len], dtype=torch.bool, device=x.device)
        mask = ~torch.triu(mask, diagonal=1)

        y = run_scaled_dot_product_attention(Q, K, V, mask)

        combined_output = rearrange(y, "b h s hd -> b s (h hd)", h=self.num_heads)

        return self.output_proj.forward(combined_output)

class TransformerBlockModel(torch.nn.Module):
    def __init__(self, d_model, num_heads, d_ff, rope) -> None:
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads

        self.rope = rope
        

        self.attn = MultiHeadAttentionModel(d_model, num_heads)

        self.ln1 = RMSNormModel(d_model)

        self.ln2 = RMSNormModel(d_model)

        self.ffn = SwiGLU(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn.forward(self.ln1.forward(x), self.rope)
        x = x + self.ffn.forward(self.ln2.forward(x))
        return x