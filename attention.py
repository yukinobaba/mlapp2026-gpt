"""「覗く」：学習したモデルの注意の重み（どの文字にどれだけ注目しているか）を見る

例：python attention.py runs/base_seed0 --text "吾輩は猫である。名前はまだ無い。"

- 層ごと・ヘッドごとの重みをヒートマップにして runs/<実験名>/attention.png に保存する
  （縦：注目する位置、横：見る位置。講義の causal mask の表と同じ向き。色の濃さはパネルごとに調整）
- 各ヘッドが「平均で何文字前を見ているか」を表示する
- --pos を付けると、その位置の文字が、どの文字にどれだけ重みを置いているかを表示する
"""
import argparse
import json
import os
import subprocess
import sys

import torch

from config import make_config
from model import GPT


def japanese_font():
    """ヒートマップの軸に日本語を出すためのフォントを用意する"""
    from matplotlib import font_manager

    for f in font_manager.fontManager.ttflist:
        if any(s in f.name for s in ("Noto Sans CJK JP", "Noto Sans JP", "IPAexGothic", "IPAGothic", "Hiragino")):
            return f.name
    # 見つからなければ、IPAexゴシックを含むパッケージを入れる（Kaggle では Internet を on にしておく）
    try:
        import matplotlib_fontja  # noqa: F401
    except ImportError:
        r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "matplotlib-fontja"], capture_output=True)
        if r.returncode != 0:
            print("注意：日本語フォントを用意できませんでした。軸の文字が表示されないことがあります")
            return None
        import matplotlib_fontja  # noqa: F401
    return "IPAexGothic"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--text", default="吾輩は猫である。名前はまだ無い。")
    ap.add_argument("--pos", type=int, help="この位置（0始まり）の文字の重みを表示する")
    ap.add_argument("--out", help="画像の保存先（既定は <run_dir>/attention.png）")
    ap.add_argument("--data", default="data")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(f"{args.run_dir}/model.pt", map_location=device)
    c = ckpt["config"]
    cfg = make_config(c["size"], **{k: v for k, v in c.items() if k not in ("size", "sample_at")})
    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    itos = json.load(open(f"{args.data}/vocab.json", encoding="utf-8"))["itos"]
    stoi = {ch: i for i, ch in enumerate(itos)}
    chars = list(args.text)[: cfg.block_size]
    idx = torch.tensor([[stoi.get(ch, 0) for ch in chars]], device=device)

    # 各層の SelfAttention に重みを残させてから、1回だけ順伝播する
    for block in model.blocks:
        block.attn.keep_attn = True
    with torch.no_grad():
        model(idx)
    # attn[l]：(ヘッド数, T, T)。attn[l][h, i, j] は「位置 i が位置 j に置いた重み」
    attn = [block.attn.last_attn[0].float().cpu() for block in model.blocks]
    T = len(chars)
    n_layer, n_head = len(attn), attn[0].shape[0]

    # 平均で何文字前を見ているか：Σ_j 重み(i, j) × (i − j) を位置 i について平均する
    # （マイナスなら、自分より後ろの位置を見ている）
    dist = torch.arange(T).view(T, 1) - torch.arange(T).view(1, T)
    print(f"入力（{T}文字）：{''.join(chars)}")
    print("平均で何文字前を見ているか（行：層、列：ヘッド）")
    for l in range(n_layer):
        d = (attn[l] * dist).sum(-1).mean(-1)
        print(f"  層{l}: " + "  ".join(f"{x:6.2f}" for x in d.tolist()))

    if args.pos is not None:
        i = args.pos
        print(f"\n位置 {i}「{chars[i]}」が各文字に置いた重み（上位5つ）")
        for l in range(n_layer):
            for h in range(n_head):
                w = attn[l][h, i]
                top = torch.topk(w, min(5, T))
                s = "  ".join(f"{chars[j]}({j}):{v:.2f}" for v, j in zip(top.values.tolist(), top.indices.tolist()))
                print(f"  層{l} ヘッド{h}: {s}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    font = japanese_font()
    if font:
        plt.rcParams["font.family"] = font
    size = max(2.2, 0.16 * T)
    fig, axes = plt.subplots(n_layer, n_head, figsize=(size * n_head, size * n_layer), squeeze=False)
    for l in range(n_layer):
        for h in range(n_head):
            ax = axes[l][h]
            # 先頭の数文字は見る相手が少なく重みが 1 近くになるので、それ以外の最大値で色の濃さを決める
            w = attn[l][h]
            vmax = w[min(3, T - 1):].max().item() if T > 3 else 1.0
            ax.imshow(w.numpy(), cmap="Blues", vmin=0, vmax=vmax)
            ax.set_title(f"層{l} ヘッド{h}", fontsize=10)
            ax.set_xticks(range(T))
            ax.set_yticks(range(T))
            ax.set_xticklabels(chars, fontsize=7)
            ax.set_yticklabels(chars, fontsize=7)
            ax.tick_params(length=0)
    fig.supxlabel("見る位置")
    fig.supylabel("注目する位置")
    fig.tight_layout()
    out = args.out or os.path.join(args.run_dir, "attention.png")
    fig.savefig(out, dpi=110)
    print(f"\nヒートマップを保存しました：{out}")


if __name__ == "__main__":
    main()
