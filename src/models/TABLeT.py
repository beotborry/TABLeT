import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)
    
class SwiGLU_MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=True)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

def apply_rope(x, seq_len):
    """
    Apply rotary position embedding (RoPE) to Q or K.
    x: (B, num_heads, seq_len, head_dim)
    """
    B, H, N, D = x.shape
    half = D // 2
    theta = 10000 ** (-torch.arange(0, half, dtype=torch.float32, device=x.device) / half)
    seq_idx = torch.arange(N, device=x.device).float()
    freqs = torch.einsum("n , d -> n d", seq_idx, theta)  # (N, half)
    sin, cos = freqs.sin(), freqs.cos()
    sin = sin[None, None, :, :].repeat(B, H, 1, 1)  # (B, H, N, half)
    cos = cos[None, None, :, :].repeat(B, H, 1, 1)

    x1, x2 = x[..., :half], x[..., half:]
    x_rot = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
    return x_rot


class MultiheadGQA(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads, dropout=0.0, use_rope=True, qk_norm=False):
        super().__init__()
        assert d_model % num_heads == 0
        assert num_heads % num_kv_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_heads
        self.kv_repeat = num_heads // num_kv_heads
        self.use_rope = use_rope

        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, self.head_dim * num_kv_heads, bias=True)
        self.v_proj = nn.Linear(d_model, self.head_dim * num_kv_heads, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)
        self.dropout = dropout
        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

    def forward(self, x, attn_mask=None):
        B, N, D = x.shape

        if self.qk_norm:
            q = self.q_norm(self.q_proj(x).view(B, N, self.num_heads, self.head_dim)).transpose(1, 2)  # (B, Hq, N, Dh)
            k = self.k_norm(self.k_proj(x).view(B, N, self.num_kv_heads, self.head_dim)).transpose(1, 2)
        else:
            q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, Hq, N, Dh)
            k = self.k_proj(x).view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)
            
        v = self.v_proj(x).view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if self.use_rope:
            q = apply_rope(q, N)
            k = apply_rope(k, N)

        # Expand keys and values to match query heads
        k = k.repeat_interleave(self.kv_repeat, dim=1)
        v = v.repeat_interleave(self.kv_repeat, dim=1)

        if attn_mask is not None:
            additive_mask = (attn_mask.bool()).unsqueeze(1).unsqueeze(2)  # [B, 1, 1, N]
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=additive_mask,
                dropout_p=self.dropout,
                is_causal=False
            )
        else:
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout,
                is_causal=False
            )
        attn_output = attn_output.transpose(1, 2).reshape(B, N, D)

        return self.out_proj(attn_output)


class TransformerEncoderLayerGQA(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads, d_ff, dropout, hidden_act='relu', norm_type='ln', mlp_type='vanilla', qk_norm=False):
        super().__init__()
        self.self_attn = MultiheadGQA(d_model, num_heads, num_kv_heads, dropout, qk_norm)
        if norm_type == 'ln':
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
        elif norm_type == 'rms':
            self.norm1 = RMSNorm(d_model)
            self.norm2 = RMSNorm(d_model)
        if mlp_type == 'vanilla':
            self.hidden_act = nn.ReLU() if hidden_act == 'relu' else nn.GELU() if hidden_act == 'gelu' else nn.SiLU()
            self.ff = nn.Sequential(
                nn.Linear(d_model, d_ff),
                self.hidden_act,
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
                nn.Dropout(dropout)
            )
        elif mlp_type == 'swiglu':
            self.ff = SwiGLU_MLP(d_model, d_ff)

    def forward(self, x, attn_mask=None):
        x = x + self.self_attn(self.norm1(x), attn_mask=attn_mask)
        x = x + self.ff(self.norm2(x))
        return x


class TABLeT(nn.Module):
    def __init__(self, input_feature_size, num_tokens, num_classes, num_layers=6, num_heads=8, num_kv_heads=2, d_model=512, d_ff=2048, dropout=0.0,
                 feature_dropout=False, hidden_act='relu',
                 norm_type='ln', mlp_type='vanilla', qk_norm=False, input_norm=False, out_norm=False, feature_norm=False, tablet_pretrained_weight_path=None):
        super().__init__()
        self.input_feature_size = input_feature_size
        self.num_tokens = num_tokens
        self.num_classes = num_classes
        self.feature_dropout = feature_dropout
        self.tablet_pretrained_weight_path = tablet_pretrained_weight_path
        self.feature_projection = nn.Linear(input_feature_size, d_model)
        if norm_type == 'ln':
            self.feature_norm = nn.LayerNorm(input_feature_size) if feature_norm else None
            self.input_norm = nn.LayerNorm(d_model) if input_norm else None
            self.out_norm = nn.LayerNorm(d_model) if out_norm else None
        elif norm_type == 'rms':
            self.feature_norm = RMSNorm(input_feature_size) if feature_norm else None
            self.input_norm = RMSNorm(d_model) if input_norm else None
            self.out_norm = RMSNorm(d_model) if out_norm else None

        self.input_dropout = nn.Dropout(dropout)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model), requires_grad=True)
        self.layers = nn.ModuleList([
            TransformerEncoderLayerGQA(d_model, num_heads, num_kv_heads, d_ff, dropout, hidden_act, norm_type=norm_type, mlp_type=mlp_type, qk_norm=qk_norm)
            for _ in range(num_layers)
        ])

        self.fc = nn.Linear(d_model, num_classes)
        
        if self.tablet_pretrained_weight_path is not None:
            state_dict = torch.load(self.tablet_pretrained_weight_path)
            state_dict['model_state_dict'].pop('mask_token')
            state_dict['model_state_dict'].pop('decoder.weight')
            state_dict['model_state_dict'].pop('decoder.bias')
            
            self.load_state_dict(state_dict['model_state_dict'], strict=True)
            print(f"Loaded tablet pretrained weights from {self.tablet_pretrained_weight_path}")
            del state_dict

    def forward(self, x, **kwargs):
        B, T, A, N, C, H, W = x.shape # batch, num_input_frames, axis=3, num_slices, channels, height, width
        assert A == 3
        num_input_frames = T
        x = x.permute(0,1,2,5,6,3,4)  # (B, T, A, H, W, N, C)
        x = x.reshape(B, T, A, H, W, N//32, C*32)
        x[:,:,0,:,:,:,:] = x[:,:,0,:,:,:,:].permute(0,1,4,2,3,5).clone()  # Bring back axis 0 to (B, T, A, N//32, H, W, C*32)
        x[:,:,1,:,:,:,:] = x[:,:,1,:,:,:,:].permute(0,1,2,4,3,5).clone() # Align axis 1 to axis 0
        # no need to align axis 2, it is already aligned
        x = torch.cat([x[:,:,i,:,:,:,:] for i in range(3)], dim = 5)  # Concatenate along the channel dimension
        x = x.reshape(B, T*H*W*N//32, C*32*3) 
        
        if self.feature_norm is not None: 
            x = self.feature_norm(x)

        x = self.feature_projection(x)
        if self.feature_dropout: x = self.input_dropout(x)
        
        attn_mask = kwargs.get('attn_mask', None)
        if attn_mask is not None:
            attn_mask = torch.repeat_interleave(attn_mask, self.num_tokens // num_input_frames, dim=1).bool()  # (B, L)
            valid_lengths = attn_mask.sum(dim=1)  # (B,)
            B, L, D = x.shape

            cls_token = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls_token, x], dim=1)
            cls_mask = torch.ones((B, 1), dtype=torch.bool, device=x.device)
            attn_mask = torch.cat([cls_mask, attn_mask], dim=1)

        else:
            cls_token = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls_token, x], dim=1)

        if self.input_norm is not None: x = self.input_norm(x)

        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask)

        if self.out_norm is not None: x = self.out_norm(x)
            
        x = x[:, 0]
        return self.fc(x).squeeze()