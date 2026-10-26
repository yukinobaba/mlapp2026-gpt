"""学習したモデルで文章を生成する

例：python generate.py runs/base_seed0 --prompt "吾輩は" --temperature 0.8 --n 3
"""
import argparse
import json

import torch

from config import make_config
from model import GPT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--prompt", default="吾輩は")
    ap.add_argument("--length", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--n", type=int, default=1, help="生成する本数")
    ap.add_argument("--seed", type=int, default=0)
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
    unknown = [ch for ch in args.prompt if ch not in stoi]
    if unknown:
        print(f"注意：語彙にない文字 {unknown} は <unk> として扱います")
    gen = torch.Generator(device=device).manual_seed(args.seed)
    for i in range(args.n):
        idx = torch.tensor([[stoi.get(ch, 0) for ch in args.prompt]], device=device)
        out = model.generate(idx, args.length, temperature=args.temperature, generator=gen)[0].tolist()
        print(f"--- {i + 1} ---")
        print("".join(itos[j] for j in out))


if __name__ == "__main__":
    main()
