import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=1,
                                   out_channels=d_model,
                                   kernel_size=3,
                                   padding=padding,
                                   padding_mode='circular')
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class DataEmbedding(nn.Module):
    def __init__(self, s_in, t_in, d_model, scale, dropout=0.1):
        super(DataEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.scale = scale
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.token_embedding = TokenEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.s_proj = nn.Conv1d(in_channels=s_in,
                                out_channels=t_in,
                                kernel_size=3,
                                padding=padding,
                                padding_mode='circular')
        self.scale_proj = nn.Conv1d(in_channels=t_in,
                                    out_channels=scale,
                                    kernel_size=3,
                                    padding=padding,
                                    padding_mode='circular')

    def forward(self, x):
        B, T, S = x.shape
        x = self.s_proj(x.permute(0, 2, 1))
        x = self.scale_proj(x)
        x = self.scale_proj(x.permute(0, 2, 1))
        x = x.view(B*self.scale, self.scale, 1)
        x = self.token_embedding(x) + self.position_embedding(x)
        x = x.view(B, self.scale, self.scale, -1)
        return self.dropout(x)


class TriangularCausalMask():
    def __init__(self, B, L, l, device="cpu"):
        mask_shape = [B, 1, L, l]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask

class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, avgPool_dim=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / math.sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class LinAttention(nn.Module):
    def __init__(self, mask_flag=True, avgPool_dim=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(LinAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.avgPool = nn.AdaptiveAvgPool1d(avgPool_dim)
        self.lin_dim = avgPool_dim

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / math.sqrt(E)

        keys = keys.view(B, S, -1).transpose(-1, 1)  # (B, S, H, D) --> (B, D, H, S)
        keys = self.avgPool(keys).transpose(-1, 1).view(B, self.lin_dim, H, D)
        values = values.view(B, S, -1).transpose(-1, 1)
        values = self.avgPool(values).transpose(-1, 1).view(B, self.lin_dim, H, E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, self.lin_dim, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads=8,
                 d_keys=None, d_values=None, mix=False):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads * 2, d_model)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values, attn_mask):
        B, L, _, _ = queries.shape
        _, S, _, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(queries, keys, values, attn_mask)
        if self.mix:
            out = out.transpose(2, 1).contiguous()
        out = out.view(B, L, -1)

        return self.out_projection(out)

class STAttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads=8,
                 d_keys=None, d_values=None, mix=False):
        super(STAttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads * 2, d_model)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values, attn_mask):
        B, T, S, dm = queries.shape
        qt = queries.view(B*S, T, dm)
        qs = queries.view(B*T, S, dm)
        kt = keys.view(B*S, T, dm)
        ks = keys.view(B*T, S, dm)
        vt = values.view(B*S, T, dm)
        vs = values.view(B*T, S, dm)
        H = self.n_heads //2

        qt = self.query_projection(qt).view(B*S, T, H, -1)
        qs = self.query_projection(qs).view(B*T, S, H, -1)
        kt = self.query_projection(kt).view(B*S, T, H, -1)
        ks = self.query_projection(ks).view(B*T, S, H, -1)
        vt = self.query_projection(vt).view(B*S, T, H, -1)
        vs = self.query_projection(vs).view(B*T, S, H, -1)

        out_t, attn_t = self.inner_attention(qt, kt, vt, attn_mask)
        out_s, attn_s = self.inner_attention(qs, ks, vs, attn_mask)
        out_t = out_t.view(B, S, T, -1)
        out_s = out_s.view(B, S, T, -1)
        out = torch.cat([out_t, out_s], dim=-1)

        if self.mix:
            out = out.transpose(2, 1).contiguous()
        out = out.view(B, S*T, -1)
        out = self.out_projection(out)
        out = out.view(B, T, S, dm)
        return out


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        B, T, S, dm = x.shape

        new_x = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = y.view(B * S, T, dm)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        y = y.view(B, T, S, dm)
        return self.norm2(x + y)


class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        for attn_layer in self.attn_layers:
            x = attn_layer(x, attn_mask=attn_mask)
        if self.norm is not None:
            x = self.norm(x)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        B, T, S, dm = x.shape
        x = x + self.dropout(self.self_attention(x, x, x, attn_mask=x_mask))
        x = self.norm1(x)

        x = x + self.dropout(self.cross_attention(x, cross, cross,attn_mask=cross_mask))
        y = x = self.norm2(x)
        y = y.view(B * S, T, dm)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        y = y.view(B, T, S, dm)
        return self.norm3(x + y)

class Decoder(nn.Module):
    """spatial-temporal linear Transformer"""
    def __init__(self, layers, norm_layer=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
        if self.norm is not None:
            x = self.norm(x)
        x = x.mean(dim=2)
        return x


class STLT(nn.Module):
    def __init__(self, enc_s, enc_t, out_s, out_t, scale=16, avgPool_dim=8,
                 d_model=32, n_heads=16, e_layers=3, d_layers=2, d_ff=64,
                 dropout=0.1, activation='gelu',
                 output_attention=False):
        super(STLT, self).__init__()
        self.pred_len = out_t
        self.output_attention = output_attention

        # Encoding
        self.enc_embedding = DataEmbedding(s_in=enc_s, t_in=enc_t, d_model=d_model, scale=scale)
        self.dec_embedding = DataEmbedding(s_in=enc_s, t_in=enc_t, d_model=d_model, scale=scale)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    STAttentionLayer(LinAttention(False, avgPool_dim, attention_dropout=dropout, output_attention=output_attention),
                                   d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    STAttentionLayer(LinAttention(True, avgPool_dim, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads, mix=False),
                    STAttentionLayer(LinAttention(False, avgPool_dim, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.conv1d = nn.Conv1d(in_channels=scale, out_channels=out_t, kernel_size=1, padding_mode='replicate')
        self.projection = nn.Linear(d_model, out_s, bias=True)

    def forward(self, x_enc, x_dec, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        enc_out = self.enc_embedding(x_enc)
        enc_out = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)

        dec_out = self.conv1d(dec_out)
        dec_out = self.projection(dec_out)

        return dec_out[: ,-self.pred_len: ,:] # [B, L, D]

class Dataset(data.Dataset):
    def __init__(self, seqs, y):
        super(Dataset, self).__init__()
        assert len(seqs) == len(y)
        self.seqs = torch.from_numpy(seqs).float()
        self.y = torch.from_numpy(y).float()
        self.len = len(seqs)

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        return self.seqs[item].cuda(), self.y[item].cuda()