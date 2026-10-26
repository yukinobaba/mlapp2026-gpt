"""小さなGPT（文字単位の言語モデル）

構成：埋め込み（文字＋位置） → Transformerブロック × n_layer → LayerNorm → 出力層
各ブロック：LayerNorm → Self-attention → 残差接続 → LayerNorm → MLP → 残差接続
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """マルチヘッドのSelf-attention（causal mask付き）"""

    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        # query・key・value をまとめて1つの線形変換で作る
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)
        # causal mask：位置 i は位置 0〜i だけを見られる（下三角が1の行列）
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size))
        self.register_buffer("mask", mask.view(1, 1, cfg.block_size, cfg.block_size))
        self.last_attn = None  # 「覗く」ために、直近の注意の重みを残せるようにする
        self.keep_attn = False

    def forward(self, x):
        B, T, C = x.shape  # バッチ, 系列長, 埋め込み次元
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) → (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # 注目の度合い（スコア）：query と key の内積を √head_dim で割る
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, n_head, T, T)
        # causal mask：後ろの位置のスコアを −∞ にして、softmax 後の重みを 0 にする
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)  # 各行の重みの合計が 1 になる
        if self.keep_attn:
            self.last_attn = att.detach()
        att = self.attn_drop(att)
        y = att @ v  # value の重み付き和 (B, n_head, T, head_dim)

        y = y.transpose(1, 2).contiguous().view(B, T, C)  # ヘッドをつなぎ直す
        return self.resid_drop(self.proj(y))


class MLP(nn.Module):
    """位置ごとの全結合2層（間に GELU）"""

    def __init__(self, cfg):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    """Transformerブロック"""

    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = SelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))  # 残差接続：ブロックの入力を足す
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)  # 文字の埋め込み
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)  # 位置の埋め込み（学習する）
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)  # 次の文字のスコア
        self.head.weight = self.tok_emb.weight  # 入力と出力で埋め込みを共有する
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None):
        """idx: 文字IDの列 (B, T)。targets を渡すと交差エントロピー損失も返す"""
        B, T = idx.shape
        assert T <= self.cfg.block_size
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.ln_f(x))  # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, generator=None):
        """確率に従って1文字選び、末尾に足して、また次を予測する"""
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.cfg.block_size:])
            probs = F.softmax(logits[:, -1, :].float() / temperature, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1, generator=generator)
            idx = torch.cat([idx, nxt], dim=1)
        return idx

    def num_params(self):
        """パラメータ数（位置埋め込みは除く）"""
        return sum(p.numel() for p in self.parameters()) - self.pos_emb.weight.numel()
