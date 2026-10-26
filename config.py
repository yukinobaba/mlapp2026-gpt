"""モデルと学習の設定。標準構成は S"""
from dataclasses import dataclass, asdict

SIZES = {
    "XS": dict(n_layer=4, n_head=4, n_embd=128),   # 約130万パラメータ
    "S": dict(n_layer=4, n_head=4, n_embd=256),    # 約420万パラメータ（標準）
    "M": dict(n_layer=6, n_head=6, n_embd=384),    # 約1,200万パラメータ
    "L": dict(n_layer=8, n_head=8, n_embd=512),    # 約2,700万パラメータ
}


@dataclass
class Config:
    # モデル
    size: str = "S"
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    block_size: int = 256      # 文脈長（文字数）
    vocab_size: int = 4000
    dropout: float = 0.1
    # 学習
    max_steps: int = 5000      # T4 で約5〜6分（標準構成 S）
    batch_size: int = 64       # 1 step あたり 64 × 256 = 16,384 文字
    lr: float = 1e-3
    warmup_steps: int = 100
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    seed: int = 0
    compile: bool = True       # torch.compile で速くする（初回に20秒ほどかかる）
    # 記録
    eval_every: int = 500      # val loss を見積もる間隔
    eval_batches: int = 20
    sample_at: tuple = (100, 300, 1000, 2000, 3000)  # 生成文を記録する step（最後にも記録する）
    sample_prompt: str = "吾輩は"
    sample_len: int = 100

    def to_dict(self):
        d = asdict(self)
        d["sample_at"] = list(self.sample_at)
        return d


def make_config(size="S", **overrides):
    cfg = Config(size=size, **SIZES[size])
    for k, v in overrides.items():
        if v is None:
            continue
        if not hasattr(cfg, k):
            raise ValueError(f"unknown config key: {k}")
        setattr(cfg, k, v)
    return cfg
