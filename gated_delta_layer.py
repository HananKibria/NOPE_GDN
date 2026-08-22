from typing import Tuple
from main import nn,torch,F
from nope_gdn import FLA_AVAILABLE

class RMSNormGated(nn.Module):
    """
    RMSNorm with sigmoid gating — matches fla's FusedRMSNormGated.

    Computes: RMSNorm(x) * sigmoid(gate)
    """
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    
    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        #RMSNorm: x / sqrt(mean(x^2) + eps) * weight
        rms = x.float().pow(2).mean(-1, keepdim = True).add(self.eps).rsqrt()
        normed = (x.float() * rms).to(x.dtype) * self.weight
        
        return normed * gate.sigmoid()

class GatedDeltaLayer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int = 4,
                 head_dim: int = 128, chunk_size: int =64,
                 channel_wise_decay: bool = True, decay_low_rank: int = None, 
                 allow_neg_eigval: bool = False, decay_target_dt: float = None,
                 a_init_range: Tuple[float, float] = (1.0, 16.0)):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.chunk_size = chunk_size
        self.channel_wise_decay = channel_wise_decay
        self.allow_neg_eigval = allow_neg_eigval
        self.scale = head_dim ** -0.5
        total_dim = num_heads * head_dim
        
        # Q, K, V projections
        self.q_proj = nn.Linear(hidden_size, total_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, total_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, total_dim, bias=False)
        
        if channel_wise_decay:
            # KDA: channel-wise log-space decay 
            rank= decay_low_rank or head_dim
            self.f_proj = nn.Sequential(
                nn.Linear(hidden_size, rank, bias=False),
                nn.Linear(rank, total_dim, bias=False)
            )   
            
            a_lo, a_hi = a_init_range
            self.A_log = nn.Parameter(
                torch.log(torch.empty(num_heads, dtype=torch.float32).uniform_(a_lo, a_hi))
            )
            self.A_log._no_weight_decay = True
            
            if decay_target_dt is not None:
                _dt_init =torch.zeros(total_dim,  dtype= torch.float32)
            else:
                _dt_init = torch.full(
                    (total_dim,), math.log(math.expm1(decay_target_dt)),
                    dtype = torch.float32)
            self.dt_bias = nn.Parameter(_dt_init)
            self.dt_bias._no_weight_decay = True
            
            if decay_target_dt is not None:
                nn.init.normal_(self.f_proj[-1].weight, std=1e-3)
        else:
            # Original GDN: scalar α per head 
            self.a_proj = nn.Linear(hidden_size, num_heads, bias=False)
         
        # Write gate β 
        self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)
        
        # Short casual convolutions
        self.q_conv1d = nn.Conv1d(total_dim, total_dim, kernel_size = 4,
                                  padding = 3, groups = total_dim)
        self.k_conv1d = nn.Conv1d(total_dim, total_dim, kernel_size = 4,
                                  padding = 3, groups = total_dim)
        self.v_conv1d = nn.Conv1d(total_dim, total_dim, kernel_size = 4,
                                    padding = 3, groups = total_dim)
        
        self.g_proj = nn.Sequential(
            nn.Linear(hidden_size, head_dim, bias = False),
            nn.Linear(head_dim, total_dim, bias = False)
        )
        
        self.o_norm = RMSNormGated(head_dim, eps = 1e-5)
        self.o_proj = nn.Linear(total_dim, hidden_size, bias = False)
        
        self.silu = nn.SiLU()
        
        def _short_conv(self, q, k, v, L, conv_state=None):
            def _one(conv, xin, st):
                xt =xin.transpose(1, 2)
                Kc = conv.kernel_size[0]
                if st is None:
                    st = xt.new_zeros(xt.shape[0], xt.shape[1], Kc - 1)
                inp = torch.cat([st, xt], dim = -1)
                out = F.conv1d(inp, conv.weight, conv.bias,
                               padding = 0, groups = conv.groups)
                return out.transpose(1, 2), inp[..., -(Kc - 1):]
            qs, ks, vs = [None, None, None] if conv_state is None else conv_state
            q, qs = _one(self.q_conv1d, q, qs)
            k, ks = _one(self.k_conv1d, k, ks)
            v, vs = _one(self.v_conv1d, v, vs)
            return q, k, v, (qs, ks, vs)
        
        def forward(self, x: torch.Tensor, state = None, use_parallel = None):
            output, new_state, _  = self._core(x, state , None, use_parallel)
            return output, new_state
        
        def forward_chunk(self, x, state = None, conv_state = None, use_parallel = None):
            return self._core(x, state, conv_state, use_parallel)
        
        def _core(self, x, state, conv_state, use_parallel):
            B, L, _ = x.shape
            if use_parallel is None:
                use_parallel = self.training or FLA_AVAILABLE 
            
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)
            
            q, k, v, new_conv_state = self._short_conv(q, k , v, L, conv_state)
            
            q = F.normalize(self.silu(q), p = 2, dim = -1) * self.scale
            k = F.normalize(self.silu(k), p = 2, dim = -1)
            v = self.silu(v)
            
            if self.channel_wise_decay:
                g = self.f_proj(x)
                g = g.view(B, L, self.num_heads, self.head_dim)
                g = g.float()
                g = g + self.dt_bias.view(self.num_heads, self.head_dim)
                g = -self.A_log.exp().view(1, 1, self.num_heads, 1) * F.softplus(g)
            
            else:
                alpha = torch.sigmoid(self.a_proj(x))
            
            beta = torch.sigmoid(self.b_proj(x))
            if self.allow_neg_eigval:
                beta = beta * 2.0
            
            q = q.view(B, L, self.num_heads, self.head_dim)
            k = k.view(B, L, self.num_heads, self.head_dim)
            v = v.view(B, L, self.num_heads, self.head_dim)
            
            if use_parallel:
                if self.channel_wise_decay:
                    output, new_state = self._chunkwise_channelwise(
                        q, k, v, g, beta, state
                    )
                else:
                    output, new_state = self._chunkwise(
                        q, k, v, alpha, beta, state
                    )
            else:
                if self.channel_wise_decay:
                    output, new_state = self._sequential(
                        q, k, v, g, beta, state, mode = 'kda'
                    )
                else:
                    output, new_state = self._sequential(
                        q, k, v, alpha, beta, state, mode = 'gdn'
                    )
        def _sequential(self, q, k, v, gate_or_alpha, beta, state, mode = 'kda'):
            B, L, H, D = q.shape
            if state is None:
                state = torch.zeros(B, H, D, D, device = q.device, dtype = q.dtype)
            I = torch.eye(D, device = q.device, dtype = q.dtype)
            outputs = []
            for t in range(L):
                q_t, k_t, v_t = q[:, t], k[:, t], v[:, t]
                b_t = beta[:, t, :, None, None]
                k_outer = k_t.unsqueeze(-1) @ k_t.unsqueeze(-2)
                householder = I - b_t * k_outer
                
                if mode == 'kda':
                    g_t = gate_or_alpha[:, t]
                    decay = g_t.exp()
                    state = (state * decay.unsqueeze(-2)) @ householder
                else:
                    a_t = gate_or_alpha[:, t, :,  None, None]
                    state = state @ (a_t * householder)
                    
                state = state + b_t * (v_t.unsqueeze(-1) @ k_t.unsqueeze(-2))
                outputs.append(state @ q_t.unsqueeze(-1)).squeeze(-1)
                
            return torch.stack(outputs, dim = 1), state
        
        def _chunkwise(self, q, k, v, alpha, beta, state):
            B, L, H, D = q.shape
            C = self.chunk_size
            
            pad = (C - L % C) % C
            if pad > 0:
                q = F.pad(q, (0, 0, 0, 0, 0, pad))
                k =F.pad(k, (0, 0, 0, 0, 0, pad))
                v = F.pad(v, (0, 0, 0, 0, 0, pad))
                alpha = F.pad(alpha, (0, 0, 0, pad), value = 1.0)
                beta = F.pad(beta, (0, 0, 0, pad), value = 0.0)
                
            L_pad = q.shape[1]
            nc =L_pad // C
            if state is None:
                state = torch. zeros(B, H, D, D, device = q.device, dtype = q.dtype)
            
            q = q.view(B, nc, C, H, D).permute(0, 3, 1, 2, 4)
            k = k.view(B, nc, C, H, D).permute(0, 3, 1, 2, 4)
            v = v.view(B, nc, C, H, D).permute(0, 3, 1, 2, 4)
            alpha = alpha.view(B, nc, C, H).permute(0, 3, 1, 2)
            beta = beta.view(B, nc, C, H).permute(0, 3, 1, 2)
            
            chunks_out = []
            for c in range(nc):
                out_c, state = self._process_chunk(
                    q[:, : , c], k[:, :, c], v[:, :, c],
                    alpha[:, :, c], beta[:, :, c], state
                )
                chunks_out.append(out_c)
            
            output = torch.cat(chunks_out, dim = 2).permute(0, 2, 1, 3)
            return (output[:, :L] if pad>0 else output), state
        
        def _chunkwise_channelwise(self, q, k, v, g, beta, state):
            B, L, H, D = q.shape
            
            if FLA_AVAILABLE:
                compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else q.dtype
                q, k, v = q.to(compute_dtype), k.to(compute_dtype), v.to(compute_dtype)
                g, beta =g.to(compute_dtype), beta.to(compute_dtype)
                output, final_state = chunk_kda(
                    q, k, v, g, beta,
                    scale = 1.0,
                    initial_state = state,
                    output_final_state = True,
                )
                if final_state is not None:
                    return output, final_state
                return output, state
            
            # Fallback
            C = self.chunk_size
            
            orig_dtype = q.dtype
            q, k, v = q.float(), k.float(), v.float()
            g = g.float()
            beta = beta.float()
            
            pad = (C - L % C) % C
            if pad > 0:
                q = F.pad(q, (0, 0, 0, 0, 0, pad))
                k = F.pad(k, (0, 0, 0, 0, 0, pad))
                v = F.pad(v, (0, 0, 0, 0, 0, pad))
                g = F.pad(g, (0, 0, 0, 0, 0, pad), value = 0.0)
                beta = F.pad(beta, (0, 0, 0, pad), value = 0.0)
                
            L_pad = q.shape[1]
            nc = L_pad // C
            if state is None:
                state = torch.zeros(B, H, D, D, device = q.device, dtype = q.dtype)
            
            q = q.view(B,nc, C, H, D).permute(0, 3, 1, 2, 4)
            k = k.view(B, nc, C, H, D).permute(0, 3, 1, 2, 4)
            v = v.view(B, nc, C, H, D).permute(0, 3, 1, 2, 4)
            g = g.view(B, nc, C, H, D).permute(0, 3, 1, 2, 4)
            beta = beta.view(B, nc, C, H).permute(0, 3, 1, 2)
            
            I = torch.eye(D, device = q.device, dtype = q.dtype)
            chunks_out = []
            
            for c in range(nc):
                q_c, k_c, v_c = q[:, :, c], k[:, :, c], v[:, :, c]
                g_c = g[:, :, c]
                b_c = beta[:, :, c]
                
                outputs_c = []
                for r in range(C):
                    q_r, k_r, v_r = q_c[:, :, r], k_c[:, :, r], v_c[:, :, r]
                    g_r = g_c[:, :, r]
                    b_r = b_c[:, :, r, None, None]
                    
                    state = state * g_r.exp().unsqueeze(-2)
                    k_outer = k_r.unsqueeze(-1) @ k_r.unsqueeze(-2)
                    state = state @ (I - b_r * k_outer) + \
                            b_r * (v_r.unsqueeze(-1) @ k_r.unsqueeze(-2))
                    outputs_c.append(state @ q_r.unsqueeze(-1).squeeze(-1))
                
                chunks_out.append(torch.stack(outputs_c, dim = 2))
            
            output = torch.cat(chunks_out, dim = 2).permute(0, 2, 1, 3)
            output = output[:, :L] if pad > 0 else output
            return output.to(orig_dtype), state
        
        def process_chunk(self, q, k, v, alpha, beta, state):
            """ Single chunk WY core with log space decay (scalar α only) """
            B, H, C, D =q.shape
            dev, dt = q.device, q.dtype
            
            # Stage1 Cumlative decay in log space
            log_alpha = torch.log(alpha.clamp(min=1e-6))
            log_gamma = torch.cumsum(log_alpha, dim = 1)
            gamma = torch.exp(log_gamma)
            gamma_C = gamma[:, : , -1]
            
            log_Gamma = log_gamma.unsqueeze(-1) - log_gamma.unsqueeze(-2)
            causal = torch.tril(torch.ones(C, C, device=dev, dtype=dt))
            Gamma = torch.exp(log_Gamma) * causal
            
            #Stage2: Gated WY -U_g
            KKT = k @ k.transpose(-1, -2)
            beta_diag = torch.diag_embed(beta)
            I_CC = torch.eye(C, device=dev, dtype=dt)
            
            
            L_g = torch.tril(beta_diag @ (Gamma * KKT), diagonal = -1)
            T_g = torch.linalg.solve_triangular(I_CC + L_g, beta_diag, upper = False)
            U_g = T_g @ v
            
            #Stage3 : Un-gated WY -> W
            L_ug = torch.tril(beta_diag @ KKT, diagonal = -1)
            T_ug = torch.linalg.solve_triangular(I_CC + L_ug, beta_diag, upper = False)
            W = T_ug @ k
            
            #Stage4L Output
            g_exp = gamma.unsqueeze(-1)
            ST = state.transpose(-1, -2)
            
            QKT_causal = (q @ k.transpose(-1, -2)) * causal
            Q_corrected = q - QKT_causal @ W
            inter = (Q_corrected * g_exp) @ ST
            
            QKT_gamma = (q @ k.transpose(-1, -2)) * Gamma
            intra = QKT_gamma @ U_g
            output = inter + intra
            
            #Stage5: state update - log space forward decay
            w_left = W * g_exp
            delta = U_g - w_left @ ST
            log_gamma_C = log_gamma[:, :, -1:]
            log_fwd = log_gamma_C - log_gamma
            k_right = k * torch.exp(log_fwd).unsqueeze(-1)
            
            new_state = state * gamma_C[:, :, None, None] + \
                delta.unsqueeze(-1) @ k_right
            
            return output, new_state
            