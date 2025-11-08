import math, torch
import torch.nn as nn

def sinusoidal_positional_encoding(T, d_model, device):
    pos = torch.arange(0, T, dtype=torch.float, device=device).unsqueeze(1)
    i = torch.arange(0, d_model, 2, dtype=torch.float, device=device)
    angles = pos / (10000 ** (i / d_model))
    pe = torch.zeros(T, d_model, device=device)
    pe[:, 0::2] = torch.sin(angles)
    pe[:, 1::2] = torch.cos(angles)
    return pe  # [T, d_model]

def causal_mask(T, device):
    return torch.tril(torch.ones(T, T, device=device, dtype=torch.bool)).unsqueeze(0).unsqueeze(1)

def make_aligned_causal_cross_mask(T_dec: int, T_enc: int, device):
    dec_pos = torch.arange(T_dec, device=device).unsqueeze(1)   # [T_dec,1]
    enc_pos = torch.arange(T_enc, device=device).unsqueeze(0)   # [1,T_enc]
    m = (enc_pos <= (dec_pos - 1))                              # [T_dec,T_enc] 布尔
    if T_enc > 0:
        no_any_true = ~m.any(dim=1)                             # 那些整行全 False 的位置（通常是 t=0）
        m[no_any_true, 0] = True                                # 兜底，至少允许看 encoder 第 0 位
    return m.unsqueeze(0).unsqueeze(1)                          # [1,1,T_dec,T_enc]

# ------------------ Core Blocks ------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, q_in, k_in, v_in, attn_mask=None):
        B, Tq, C = q_in.shape
        Tk = k_in.size(1)

        q = self.q_proj(q_in).view(B, Tq, self.n_head, self.d_head).transpose(1, 2)  # [B,h,Tq,d]
        k = self.k_proj(k_in).view(B, Tk, self.n_head, self.d_head).transpose(1, 2)  # [B,h,Tk,d]
        v = self.v_proj(v_in).view(B, Tk, self.n_head, self.d_head).transpose(1, 2)  # [B,h,Tk,d]

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)                     # [B,h,Tq,Tk]
        if attn_mask is not None:
            att = att.masked_fill(attn_mask == 0, float('-inf'))
        att = torch.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = att @ v                                                                  # [B,h,Tq,d]
        y = y.transpose(1, 2).contiguous().view(B, Tq, C)
        return self.resid_drop(self.out_proj(y))

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )
    def forward(self, x): return self.net(x)

class EncoderBlock(nn.Module):
    def __init__(self, d_model, n_head, d_ff, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.mha = MultiHeadAttention(d_model, n_head, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)
    def forward(self, x, src_mask=None):
        x = x + self.mha(self.ln1(x), self.ln1(x), self.ln1(x), src_mask)  # 通常 encoder 不加因果 mask
        x = x + self.ffn(self.ln2(x))
        return x

class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_head, d_ff, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.self_mha  = MultiHeadAttention(d_model, n_head, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.cross_mha = MultiHeadAttention(d_model, n_head, dropout)
        self.ln3 = nn.LayerNorm(d_model)
        self.ffn       = FeedForward(d_model, d_ff, dropout)
    def forward(self, x, enc_out, tgt_mask=None, cross_mask=None):
        x = x + self.self_mha(self.ln1(x), self.ln1(x), self.ln1(x), tgt_mask)
        x = x + self.cross_mha(self.ln2(x), enc_out, enc_out, cross_mask)
        x = x + self.ffn(self.ln3(x))
        return x

class TinyTransformerSeq2Seq(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_head=4, n_layer=2, d_ff=512,
                 block_size=64, dropout=0.1, use_pe=True):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.d_model = d_model
        self.use_pe = use_pe

        self.src_emb = nn.Embedding(vocab_size, d_model)
        self.tgt_emb = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(dropout)

        self.encoder = nn.ModuleList([EncoderBlock(d_model, n_head, d_ff, dropout) for _ in range(n_layer)])
        self.decoder = nn.ModuleList([DecoderBlock(d_model, n_head, d_ff, dropout) for _ in range(n_layer)])

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, src_ids, tgt_in_ids):
        B, Ts = src_ids.shape
        Tt = tgt_in_ids.size(1)
        device = src_ids.device

        src = self.src_emb(src_ids)
        if self.use_pe:
            src = src + sinusoidal_positional_encoding(Ts, self.d_model, device).unsqueeze(0)
        src = self.drop(src)
        src_mask = None
        for blk in self.encoder:
            src = blk(src, src_mask)

        tgt = self.tgt_emb(tgt_in_ids)
        if self.use_pe:
            tgt = tgt + sinusoidal_positional_encoding(Tt, self.d_model, device).unsqueeze(0)
        tgt = self.drop(tgt)

        tgt_mask   = causal_mask(Tt, device)                        # [1,1,Tt,Tt]
        cross_mask = make_aligned_causal_cross_mask(Tt, Ts, device) # [1,1,Tt,Ts]

        for blk in self.decoder:
            tgt = blk(tgt, src, tgt_mask=tgt_mask, cross_mask=cross_mask)

        out = self.ln_f(tgt)
        logits = self.head(out)  # [B, Tt, V]
        return logits

    @torch.no_grad()
    def generate(self, src_ids, start_id, max_new_tokens=200, temperature=1.0, top_k=50):
        self.eval()
        device = src_ids.device
        B, _ = src_ids.shape
        tgt = torch.full((B, 1), start_id, dtype=torch.long, device=device)
        for _ in range(max_new_tokens):
            logits = self.forward(src_ids, tgt)[:, -1, :] / max(1e-6, temperature)
            if top_k is not None:
                v, ix = torch.topk(logits, k=min(top_k, logits.size(-1)), dim=-1)
                probs = torch.softmax(v, dim=-1)
                next_id = ix.gather(-1, torch.multinomial(probs, 1))
            else:
                probs = torch.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, 1)
            tgt = torch.cat([tgt, next_id], dim=1)
        return tgt
