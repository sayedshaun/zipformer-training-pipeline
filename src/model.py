"""Zipformer, built from scratch, module by module, in one file.

See ARCHITECTURE.md for the full architectural reference this follows.
Sections below are ordered bottom-up: helpers -> feature extraction ->
subsampling -> positional encoding -> BiasNorm/Swoosh -> attention (shared
weights + non-linear attention + MHSA) -> feed-forward -> convolution ->
bypass -> Zipformer block -> encoder -> CTC head -> RNNT prediction/joint ->
the top-level model that wires it all together.

Feature extraction, subsampling, positional encoding, and the CTC/RNNT heads
are unchanged from the sibling FastConformer repo's model.py - only the
encoder block internals (BiasNorm instead of LayerNorm, Swoosh instead of
Swish, shared-attention-weight NLA+MHSA instead of a single MHSA, and a
learnable bypass instead of a plain residual) differ, per ARCHITECTURE.md.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def lengths_to_padding_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """True where a position is padding. Shape: (batch, max_len)."""
    arange = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    return arange >= lengths.unsqueeze(1)


def conv_out_length(
    lengths: torch.Tensor, kernel_size: int, stride: int, padding: int
) -> torch.Tensor:
    return (
        torch.div(lengths + 2 * padding - kernel_size, stride, rounding_mode="floor")
        + 1
    )


def _hz_to_mel(hz: float) -> float:
    return 2595.0 * math.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: torch.Tensor) -> torch.Tensor:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def build_mel_filterbank(n_mels: int, n_fft: int, sample_rate: int) -> torch.Tensor:
    """Triangular mel filterbank, shape (n_mels, n_fft // 2 + 1)."""
    n_freq_bins = n_fft // 2 + 1
    mel_min, mel_max = _hz_to_mel(0.0), _hz_to_mel(sample_rate / 2)
    mel_points = torch.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = _mel_to_hz(mel_points)
    bin_points = torch.floor((n_fft + 1) * hz_points / sample_rate).long()

    filterbank = torch.zeros(n_mels, n_freq_bins)
    for m in range(1, n_mels + 1):
        left, center, right = (
            bin_points[m - 1].item(),
            bin_points[m].item(),
            bin_points[m + 1].item(),
        )
        if center > left:
            k = torch.arange(left, center)
            filterbank[m - 1, k] = (k - left).float() / (center - left)
        if right > center:
            k = torch.arange(center, right)
            filterbank[m - 1, k] = (right - k).float() / (right - center)
    return filterbank


class LogMelFeatureExtractor(nn.Module):
    """Module 1: log-mel filterbank feature extraction. Waveform -> (B, T, n_mels)."""

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 400,
        hop_length: int = 160,
        win_length: int = 400,
        n_mels: int = 80,
        log_eps: float = 1e-5,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.log_eps = log_eps
        self.register_buffer("window", torch.hann_window(win_length))
        self.register_buffer(
            "mel_filterbank", build_mel_filterbank(n_mels, n_fft, sample_rate)
        )

    def forward(self, waveform: torch.Tensor, waveform_lengths: torch.Tensor):
        """waveform: (B, T_samples). Returns (features (B, T_frames, n_mels), feature_lengths)."""
        stft = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            return_complex=True,
        )
        power_spectrum = stft.abs() ** 2  # (B, n_freq_bins, T_frames)
        mel_spectrum = torch.einsum("mf,bft->bmt", self.mel_filterbank, power_spectrum)
        log_mel = torch.log(mel_spectrum.clamp(min=self.log_eps))
        features = log_mel.transpose(1, 2)  # (B, T_frames, n_mels)

        # center=True pads n_fft//2 on each side, matching conv_out_length with that padding.
        feature_lengths = conv_out_length(
            waveform_lengths,
            kernel_size=self.win_length,
            stride=self.hop_length,
            padding=self.n_fft // 2,
        )
        return features, feature_lengths


class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, stride: int = 2):
        super().__init__()
        padding = kernel_size // 2
        self.depthwise = nn.Conv2d(
            channels, channels, kernel_size, stride=stride, padding=padding, groups=channels,
        )
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.pointwise(self.depthwise(x)))


class ConvSubsampling(nn.Module):
    """Module 2: 8x time subsampling via depthwise-separable convolutions.
    (B, T, n_mels) -> (B, T // 8, d_model), with matching length reduction."""

    def __init__(self, d_model: int, n_mels: int = 80, kernel_size: int = 3):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = 2
        self.padding = kernel_size // 2

        self.first_conv = nn.Conv2d(1, d_model, kernel_size, stride=self.stride, padding=self.padding)
        self.first_activation = nn.ReLU()
        self.dw_conv1 = DepthwiseSeparableConv2d(d_model, kernel_size, self.stride)
        self.dw_conv2 = DepthwiseSeparableConv2d(d_model, kernel_size, self.stride)

        freq_out = n_mels
        for _ in range(3):
            freq_out = (freq_out + 2 * self.padding - kernel_size) // self.stride + 1
        self.out_proj = nn.Linear(d_model * freq_out, d_model)

    def forward(self, features: torch.Tensor, feature_lengths: torch.Tensor):
        x = features.unsqueeze(1)  # (B, 1, T, n_mels)
        x = self.first_activation(self.first_conv(x))
        x = self.dw_conv1(x)
        x = self.dw_conv2(x)  # (B, d_model, T // 8, n_mels // 8)

        b, c, t, f = x.shape
        x = x.permute(0, 2, 1, 3).reshape(b, t, c * f)
        x = self.out_proj(x)  # (B, T // 8, d_model)

        out_lengths = feature_lengths
        for _ in range(3):
            out_lengths = conv_out_length(out_lengths, self.kernel_size, self.stride, self.padding)
        return x, out_lengths


class RelPositionalEncoding(nn.Module):
    """Module 3: relative positional encoding (Transformer-XL style).
    Not added to x directly - passed alongside x into the attention weights
    module, which uses it to compute relative position bias."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        self.max_len = 0
        self.register_buffer("pe", torch.empty(0), persistent=False)
        self._build_pe(max_len, torch.device("cpu"), torch.float32)

    def _build_pe(self, max_len: int, device: torch.device, dtype: torch.dtype):
        positions = torch.arange(
            max_len - 1, -max_len, -1, dtype=torch.float32, device=device
        ).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32, device=device)
            * (-math.log(10000.0) / self.d_model)
        )
        pe = torch.zeros(2 * max_len - 1, self.d_model, device=device)
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)
        self.pe = pe.unsqueeze(0).to(dtype)
        self.max_len = max_len

    def forward(self, x: torch.Tensor):
        """x: (B, T, d_model). Returns (x, pos_emb) where pos_emb has shape (1, 2T-1, d_model).

        x is passed through untouched: unlike absolute positional encoding, the
        embedding is never added to x (AttentionWeights projects `pos_emb`
        separately), so there is nothing for a sqrt(d_model) input scale to pair
        with - it would only inflate the first block's residual branch.
        """
        t = x.size(1)
        # Grown lazily rather than at construction, so a longer-than-expected
        # utterance rebuilds `pe` on x's device instead of leaving it on the CPU.
        if t > self.max_len or self.pe.device != x.device:
            self._build_pe(max(t, self.max_len), x.device, self.pe.dtype)
        center = self.max_len - 1
        pos_emb = self.pe[:, center - (t - 1) : center + t]
        return x, pos_emb


class BiasNorm(nn.Module):
    """Module 4a: BiasNorm (ARCHITECTURE.md section 2) - replaces LayerNorm.
    Keeps x's own magnitude/mean; only uses a learned per-channel bias to
    estimate the scale to divide by."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.log_scale = nn.Parameter(torch.zeros(1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., d_model)."""
        rms = (x - self.bias).pow(2).mean(dim=-1, keepdim=True).clamp(min=self.eps).sqrt()
        return x / rms * self.log_scale.exp()


def swoosh_r(x: torch.Tensor) -> torch.Tensor:
    """Module 4b: SwooshR (ARCHITECTURE.md section 3) - used in feed-forward modules."""
    return F.softplus(x - 1.0) - 0.08 * x - 0.313


def swoosh_l(x: torch.Tensor) -> torch.Tensor:
    """Module 4b: SwooshL (ARCHITECTURE.md section 3) - used in convolution modules."""
    return F.softplus(x - 4.0) - 0.08 * x - 0.035


def rel_shift(x: torch.Tensor) -> torch.Tensor:
    """x: (B, H, T, 2T-1) -> (B, H, T, T), aligning relative offsets to absolute positions."""
    b, h, t1, t2 = x.size()
    zero_pad = torch.zeros((b, h, t1, 1), device=x.device, dtype=x.dtype)
    x_padded = torch.cat([zero_pad, x], dim=-1)
    x_padded = x_padded.view(b, h, t2 + 1, t1)
    return x_padded[:, :, 1:].view_as(x)[:, :, :, : t2 // 2 + 1]


class AttentionWeights(nn.Module):
    """Module 5a: computes shared (B, H, T, T) attention weights from Q/K only
    (relative-position bias, Transformer-XL style) - reused by both the
    Non-Linear Attention module and the MHSA module in the same block, so the
    O(T^2) QK^T cost is paid once instead of twice (ARCHITECTURE.md section 4)."""

    def __init__(self, d_model: int, n_heads: int, attn_head_dim: int = 24, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.attn_head_dim = attn_head_dim
        attn_dim = n_heads * attn_head_dim

        self.norm = BiasNorm(d_model)
        self.linear_q = nn.Linear(d_model, attn_dim)
        self.linear_k = nn.Linear(d_model, attn_dim)
        self.linear_pos = nn.Linear(d_model, attn_dim, bias=False)

        self.pos_bias_u = nn.Parameter(torch.zeros(n_heads, attn_head_dim))
        self.pos_bias_v = nn.Parameter(torch.zeros(n_heads, attn_head_dim))
        nn.init.xavier_uniform_(self.pos_bias_u)
        nn.init.xavier_uniform_(self.pos_bias_v)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, pos_emb: torch.Tensor, padding_mask: torch.Tensor = None):
        """x: (B, T, d_model), pos_emb: (1, 2T-1, d_model). Returns weights (B, H, T, T)."""
        b, t, _ = x.shape
        x = self.norm(x)

        q = self.linear_q(x).view(b, t, self.n_heads, self.attn_head_dim).transpose(1, 2)
        k = self.linear_k(x).view(b, t, self.n_heads, self.attn_head_dim).transpose(1, 2)
        p = self.linear_pos(pos_emb).view(1, -1, self.n_heads, self.attn_head_dim).transpose(1, 2)

        q_with_bias_u = q + self.pos_bias_u.unsqueeze(0).unsqueeze(2)
        q_with_bias_v = q + self.pos_bias_v.unsqueeze(0).unsqueeze(2)

        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))
        matrix_bd = rel_shift(torch.matmul(q_with_bias_v, p.transpose(-2, -1)))
        scores = (matrix_ac + matrix_bd) / math.sqrt(self.attn_head_dim)

        if padding_mask is not None:
            mask = padding_mask.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, T)
            scores = scores.masked_fill(mask, float("-inf"))

        return self.dropout(torch.softmax(scores, dim=-1))


class NonLinearAttention(nn.Module):
    """Module 5b: applies pre-computed attention weights to a GLU-gated
    projection of x (not a plain linear V, unlike standard attention) -
    ARCHITECTURE.md section 4."""

    def __init__(self, d_model: int, n_heads: int, hidden_dim: int = 192, dropout: float = 0.1):
        super().__init__()
        assert hidden_dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

        self.norm = BiasNorm(d_model)
        self.in_proj = nn.Linear(d_model, hidden_dim * 3)  # values, gate_a, gate_b
        self.out_proj = nn.Linear(hidden_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_weights: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model), attn_weights: (B, H, T, T)."""
        b, t, _ = x.shape
        x = self.norm(x)
        values, gate_a, gate_b = self.in_proj(x).chunk(3, dim=-1)
        gated = torch.tanh(gate_a) * torch.sigmoid(gate_b) * values  # (B, T, hidden_dim)

        gated = gated.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, T, head_dim)
        out = torch.matmul(attn_weights, gated)  # (B, H, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(b, t, self.n_heads * self.head_dim)
        return self.dropout(self.out_proj(out))


class SharedWeightMHSA(nn.Module):
    """Module 5c: standard attention output (softmax(QK^T) @ V) but reusing
    attention weights computed once by AttentionWeights, only doing its own
    V projection + output projection."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.norm = BiasNorm(d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_weights: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model), attn_weights: (B, H, T, T)."""
        b, t, _ = x.shape
        x = self.norm(x)
        v = self.linear_v(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(b, t, self.n_heads * self.head_dim)
        return self.dropout(self.linear_out(out))


class FeedForwardModule(nn.Module):
    """Module 6: feed-forward module - BiasNorm + SwooshR instead of
    Conformer's LayerNorm + Swish. Does not apply the residual itself."""

    def __init__(self, d_model: int, expansion_factor: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm = BiasNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_model * expansion_factor)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model * expansion_factor, d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.linear1(x)
        x = swoosh_r(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        return self.dropout2(x)


class ConvolutionModule(nn.Module):
    """Module 7: convolution module - BiasNorm + SwooshL, no BatchNorm
    (ARCHITECTURE.md section 6 - Zipformer drops BatchNorm entirely)."""

    def __init__(self, d_model: int, kernel_size: int = 9, dropout: float = 0.1):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd for symmetric 'same' padding"
        self.norm = BiasNorm(d_model)
        self.pointwise_conv1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(
            d_model, d_model, kernel_size, padding=kernel_size // 2, groups=d_model,
        )
        self.mid_norm = BiasNorm(d_model)
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor = None) -> torch.Tensor:
        """x: (B, T, d_model). padding_mask: (B, T) bool, True where padded."""
        x = self.norm(x)
        x = x.transpose(1, 2)  # (B, d_model, T)

        if padding_mask is not None:
            x = x.masked_fill(padding_mask.unsqueeze(1), 0.0)

        x = self.pointwise_conv1(x)
        x = self.glu(x)

        if padding_mask is not None:
            # pointwise_conv1 has a bias, so the zeroed padding columns above are
            # non-zero again by now. Re-zero them before the depthwise conv, whose
            # kernel_size // 2 receptive field would otherwise pull that padding
            # into the last few *valid* frames.
            x = x.masked_fill(padding_mask.unsqueeze(1), 0.0)

        x = self.depthwise_conv(x)
        x = self.mid_norm(x.transpose(1, 2)).transpose(1, 2)
        x = swoosh_l(x)
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        return x.transpose(1, 2)  # (B, T, d_model)


class BypassModule(nn.Module):
    """Module 8: learnable per-channel lerp between a block's input and
    output, clamped to [min_scale, 1.0] - replaces a plain residual add
    (ARCHITECTURE.md section 5a). Lets a channel mostly skip a block without
    a hard architectural gate."""

    def __init__(self, d_model: int, min_scale: float = 0.4):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d_model))
        self.min_scale = min_scale

    def forward(self, x: torch.Tensor, block_out: torch.Tensor) -> torch.Tensor:
        scale = self.scale.clamp(min=self.min_scale, max=1.0)
        return x + scale * (block_out - x)


class ZipformerBlock(nn.Module):
    """Module 9: one Zipformer block (ARCHITECTURE.md section 5).
    Simplified to compute attention weights once per block (shared by NLA and
    both MHSA calls) rather than the reference's twice-per-block split."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        attn_head_dim: int = 24,
        nla_hidden_dim: int = 192,
        conv_kernel_size: int = 9,
        ff_expansion_factor: int = 4,
        dropout: float = 0.1,
        bypass_min_scale: float = 0.4,
    ):
        super().__init__()
        self.attn_weights = AttentionWeights(d_model, n_heads, attn_head_dim, dropout)
        self.nla = NonLinearAttention(d_model, n_heads, nla_hidden_dim, dropout)
        self.ff1 = FeedForwardModule(d_model, ff_expansion_factor, dropout)
        self.self_attn1 = SharedWeightMHSA(d_model, n_heads, dropout)
        self.conv1 = ConvolutionModule(d_model, conv_kernel_size, dropout)
        self.ff2 = FeedForwardModule(d_model, ff_expansion_factor, dropout)
        self.self_attn2 = SharedWeightMHSA(d_model, n_heads, dropout)
        self.conv2 = ConvolutionModule(d_model, conv_kernel_size, dropout)
        self.final_norm = BiasNorm(d_model)
        self.bypass = BypassModule(d_model, bypass_min_scale)

    def forward(self, x: torch.Tensor, pos_emb: torch.Tensor, padding_mask: torch.Tensor = None):
        residual = x
        attn_weights = self.attn_weights(x, pos_emb, padding_mask)

        x = x + self.nla(x, attn_weights)
        x = x + self.ff1(x)
        x = x + self.self_attn1(x, attn_weights)
        x = x + self.conv1(x, padding_mask)
        x = x + self.ff2(x)
        x = x + self.self_attn2(x, attn_weights)
        x = x + self.conv2(x, padding_mask)
        x = self.final_norm(x)

        return self.bypass(residual, x)


class ZipformerEncoder(nn.Module):
    """Module 10: the full encoder - subsampling + N Zipformer blocks."""

    def __init__(
        self,
        n_mels: int = 80,
        d_model: int = 512,
        n_layers: int = 17,
        n_heads: int = 8,
        attn_head_dim: int = 24,
        nla_hidden_dim: int = 192,
        conv_kernel_size: int = 9,
        ff_expansion_factor: int = 4,
        dropout: float = 0.1,
        bypass_min_scale: float = 0.4,
    ):
        super().__init__()
        self.subsampling = ConvSubsampling(d_model, n_mels)
        self.pos_encoding = RelPositionalEncoding(d_model)
        self.blocks = nn.ModuleList(
            [
                ZipformerBlock(
                    d_model, n_heads, attn_head_dim, nla_hidden_dim,
                    conv_kernel_size, ff_expansion_factor, dropout, bypass_min_scale,
                )
                for _ in range(n_layers)
            ]
        )
        # ARCHITECTURE.md section 7: the stack ends in a BiasNorm before the
        # heads. Each block's own final_norm sits *before* its bypass, so the
        # encoder output would otherwise have an unconstrained scale.
        self.final_norm = BiasNorm(d_model)
        self.d_model = d_model

    def forward(self, features: torch.Tensor, feature_lengths: torch.Tensor):
        """features: (B, T, n_mels). Returns (encoder_out (B, T', d_model), out_lengths)."""
        x, lengths = self.subsampling(features, feature_lengths)
        x, pos_emb = self.pos_encoding(x)

        padding_mask = lengths_to_padding_mask(lengths, x.size(1))
        for block in self.blocks:
            x = block(x, pos_emb, padding_mask)
        return self.final_norm(x), lengths


class CTCHead(nn.Module):
    """Module 11: CTC decoder head - unchanged from the FastConformer reference."""

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.blank_id = vocab_size  # CTC blank token, appended after the real vocab
        self.linear = nn.Linear(d_model, vocab_size + 1)

    def forward(self, encoder_out: torch.Tensor) -> torch.Tensor:
        """encoder_out: (B, T, d_model). Returns log-probs (B, T, vocab_size + 1)."""
        return torch.log_softmax(self.linear(encoder_out), dim=-1)


class RNNTPredictionNetwork(nn.Module):
    """Module 12a: RNNT prediction network - unchanged from the FastConformer
    reference. Teacher-forced during training."""

    def __init__(self, vocab_size: int, pred_dim: int = 320, n_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.blank_id = vocab_size
        self.embedding = nn.Embedding(vocab_size + 1, pred_dim)
        self.lstm = nn.LSTM(
            pred_dim, pred_dim, num_layers=n_layers, batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

    def forward(self, targets: torch.Tensor, states=None):
        """targets: (B, U) target token ids, no blank. Returns (B, U + 1, pred_dim), states."""
        blank_col = torch.full(
            (targets.size(0), 1), self.blank_id, dtype=targets.dtype, device=targets.device,
        )
        prepended = torch.cat([blank_col, targets], dim=1)
        embedded = self.embedding(prepended)
        return self.lstm(embedded, states)


class RNNTJoint(nn.Module):
    """Module 12b: RNNT joint network - unchanged from the FastConformer reference."""

    def __init__(self, encoder_dim: int, pred_dim: int, joint_dim: int, vocab_size: int):
        super().__init__()
        self.enc_proj = nn.Linear(encoder_dim, joint_dim)
        self.pred_proj = nn.Linear(pred_dim, joint_dim)
        self.activation = nn.Tanh()
        self.out = nn.Linear(joint_dim, vocab_size + 1)

    def forward(self, encoder_out: torch.Tensor, pred_out: torch.Tensor) -> torch.Tensor:
        """
        encoder_out: (B, T, encoder_dim)
        pred_out: (B, U, pred_dim)
        Returns log-probs (B, T, U, vocab_size + 1).
        """
        enc = self.enc_proj(encoder_out).unsqueeze(2)  # (B, T, 1, joint_dim)
        pred = self.pred_proj(pred_out).unsqueeze(1)  # (B, 1, U, joint_dim)
        joint = self.activation(enc + pred)  # (B, T, U, joint_dim)
        return torch.log_softmax(self.out(joint), dim=-1)


class ZipformerFromScratch(nn.Module):
    """Module 13: wires everything above into one model. Exposes the same
    forward_ctc/forward_rnnt interface as the sibling FastConformer repo's
    model, so dataset.py/train.py/eval.py port over unchanged."""

    def __init__(
        self,
        vocab_size: int,
        sample_rate: int = 16000,
        n_mels: int = 80,
        d_model: int = 512,
        n_layers: int = 17,
        n_heads: int = 8,
        attn_head_dim: int = 24,
        nla_hidden_dim: int = 192,
        conv_kernel_size: int = 9,
        ff_expansion_factor: int = 4,
        dropout: float = 0.1,
        bypass_min_scale: float = 0.4,
        use_rnnt: bool = True,
        pred_dim: int = 320,
        joint_dim: int = 512,
    ):
        super().__init__()
        self.feature_extractor = LogMelFeatureExtractor(sample_rate=sample_rate, n_mels=n_mels)
        self.encoder = ZipformerEncoder(
            n_mels=n_mels,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            attn_head_dim=attn_head_dim,
            nla_hidden_dim=nla_hidden_dim,
            conv_kernel_size=conv_kernel_size,
            ff_expansion_factor=ff_expansion_factor,
            dropout=dropout,
            bypass_min_scale=bypass_min_scale,
        )
        self.ctc_head = CTCHead(d_model, vocab_size)

        self.use_rnnt = use_rnnt
        if use_rnnt:
            self.prediction_network = RNNTPredictionNetwork(vocab_size, pred_dim)
            self.joint_network = RNNTJoint(d_model, pred_dim, joint_dim, vocab_size)

    def forward(self, waveform: torch.Tensor, waveform_lengths: torch.Tensor, targets: torch.Tensor = None):
        """Dispatches to forward_rnnt when `targets` is given, else forward_ctc.

        DistributedDataParallel only installs its gradient-sync hooks around
        `forward()`, so distributed training must call the module itself
        (`model(...)`) rather than `model.forward_ctc(...)` - the latter bypasses
        the wrapper and silently trains each rank on its own gradients.
        """
        if targets is not None:
            return self.forward_rnnt(waveform, waveform_lengths, targets)
        return self.forward_ctc(waveform, waveform_lengths)

    def encode(self, waveform: torch.Tensor, waveform_lengths: torch.Tensor):
        """waveform: (B, T_samples). Returns (encoder_out (B, T', d_model), out_lengths)."""
        features, feature_lengths = self.feature_extractor(waveform, waveform_lengths)
        return self.encoder(features, feature_lengths)

    def forward_ctc(self, waveform: torch.Tensor, waveform_lengths: torch.Tensor):
        """Returns (log_probs (B, T', vocab_size + 1), encoded_lengths)."""
        encoder_out, encoded_lengths = self.encode(waveform, waveform_lengths)
        return self.ctc_head(encoder_out), encoded_lengths

    def forward_rnnt(self, waveform: torch.Tensor, waveform_lengths: torch.Tensor, targets: torch.Tensor):
        """targets: (B, U) target token ids (no blank), teacher-forced.
        Returns (joint_log_probs (B, T', U + 1, vocab_size + 1), encoded_lengths)."""
        if not self.use_rnnt:
            raise RuntimeError("Model was built with use_rnnt=False")
        encoder_out, encoded_lengths = self.encode(waveform, waveform_lengths)
        pred_out, _ = self.prediction_network(targets)
        joint_out = self.joint_network(encoder_out, pred_out)
        return joint_out, encoded_lengths
