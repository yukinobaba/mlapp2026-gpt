"""学習スクリプト

例：
  python train.py --name base_seed0                 # 標準構成 S、5,000 step
  python train.py --name base_seed1 --seed 1        # seed だけ変える
  python train.py --name xs_test --size XS --max_steps 300

結果は runs/<name>/ に保存する：
  config.json   この実験の設定（seed を含む）
  log.csv       step ごとの train loss / val loss（見積もり）
  samples.txt   途中の step での生成文
  metrics.json  最終結果（検証データ全体での val loss など）
  model.pt      学習したモデル
"""
import argparse
import csv
import json
import math
import os
import subprocess
import time

import numpy as np
import torch

from config import make_config
from model import GPT


def load_split(data_dir, split):
    return np.memmap(f"{data_dir}/{split}.bin", dtype=np.uint16, mode="r")


def get_batch(data, cfg, gen, device):
    """学習データからランダムに batch_size 個の区間を切り出す。y は x を1文字ずらしたもの"""
    ix = torch.randint(len(data) - cfg.block_size - 1, (cfg.batch_size,), generator=gen).tolist()
    x = torch.from_numpy(np.stack([data[i:i + cfg.block_size] for i in ix]).astype(np.int64))
    y = torch.from_numpy(np.stack([data[i + 1:i + 1 + cfg.block_size] for i in ix]).astype(np.int64))
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


@torch.no_grad()
def estimate_val_loss(model, val, cfg, device, n_batches):
    """検証データの決まった区間で val loss を見積もる（学習中の目安）"""
    model.eval()
    gen = torch.Generator().manual_seed(1234)  # 毎回同じ区間を使う
    losses = []
    for _ in range(n_batches):
        x, y = get_batch(val, cfg, gen, device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


@torch.no_grad()
def full_val_loss(model, val, cfg, device, batch_size=64):
    """検証データ全体を重ならない区間に分けて平均した val loss（実験の比較にはこれを使う）"""
    model.eval()
    T = cfg.block_size
    n = (len(val) - 1) // T
    total, count = 0.0, 0
    for s in range(0, n, batch_size):
        starts = np.arange(s, min(s + batch_size, n)) * T
        x = torch.from_numpy(np.stack([val[i:i + T] for i in starts]).astype(np.int64)).to(device)
        y = torch.from_numpy(np.stack([val[i + 1:i + 1 + T] for i in starts]).astype(np.int64)).to(device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            _, loss = model(x, y)
        total += loss.item() * len(starts)
        count += len(starts)
    model.train()
    return total / count


def sample_text(model, itos, stoi, cfg, device, seed=0, temperature=0.8):
    model.eval()
    idx = torch.tensor([[stoi.get(c, 0) for c in cfg.sample_prompt]], device=device)
    gen = torch.Generator(device=device).manual_seed(seed)
    out = model.generate(idx, cfg.sample_len, temperature=temperature, generator=gen)[0].tolist()
    model.train()
    return "".join(itos[i] for i in out)


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="実験の名前（runs/<name>/ に保存）")
    ap.add_argument("--size", default="S", choices=["XS", "S", "M", "L"])
    ap.add_argument("--max_steps", type=int)
    ap.add_argument("--batch_size", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--dropout", type=float)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--no_compile", action="store_true")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="runs")
    args = ap.parse_args()

    cfg = make_config(args.size, max_steps=args.max_steps, batch_size=args.batch_size, lr=args.lr,
                      dropout=args.dropout, seed=args.seed)
    if args.no_compile:
        cfg.compile = False
    out_dir = os.path.join(args.out, args.name)
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    train, val = load_split(args.data, "train"), load_split(args.data, "val")
    itos = json.load(open(f"{args.data}/vocab.json", encoding="utf-8"))["itos"]
    stoi = {c: i for i, c in enumerate(itos)}

    model = GPT(cfg).to(device)
    print(f"パラメータ数: {model.num_params():,}  device: {device}")
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(0.9, 0.95),
                            weight_decay=cfg.weight_decay)
    fwd = torch.compile(model) if (cfg.compile and device.type == "cuda") else model
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    gen = torch.Generator().manual_seed(cfg.seed)

    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(dict(cfg.to_dict(), name=args.name, git_commit=git_commit(),
                       gpu=torch.cuda.get_device_name(0) if device.type == "cuda" else None),
                  f, ensure_ascii=False, indent=1)
    log_f = open(os.path.join(out_dir, "log.csv"), "w", newline="")
    log = csv.writer(log_f)
    log.writerow(["step", "sec", "lr", "train_loss", "val_loss"])
    samples_f = open(os.path.join(out_dir, "samples.txt"), "w", encoding="utf-8")

    t0 = time.time()
    for step in range(1, cfg.max_steps + 1):
        # 学習率：最初の warmup_steps で線形に上げ、その後コサインで 0 まで下げる
        if step <= cfg.warmup_steps:
            lr = cfg.lr * step / cfg.warmup_steps
        else:
            p = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
            lr = cfg.lr * 0.5 * (1 + math.cos(math.pi * p))
        for g in opt.param_groups:
            g["lr"] = lr

        x, y = get_batch(train, cfg, gen, device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            _, loss = fwd(x, y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(opt)
        scaler.update()

        if step % 100 == 0 or step == cfg.max_steps:
            val_loss = ""
            if step % cfg.eval_every == 0 or step == cfg.max_steps:
                val_loss = round(estimate_val_loss(model, val, cfg, device, cfg.eval_batches), 4)
                print(f"step {step:5d}  {time.time() - t0:6.0f}s  train {loss.item():.3f}  val {val_loss}",
                      flush=True)
            log.writerow([step, round(time.time() - t0, 1), f"{lr:.2e}", round(loss.item(), 4), val_loss])
            log_f.flush()
        if step in cfg.sample_at or step == cfg.max_steps:
            samples_f.write(f"=== step {step} ===\n{sample_text(model, itos, stoi, cfg, device)}\n\n")
            samples_f.flush()

    train_sec = time.time() - t0
    vl = full_val_loss(model, val, cfg, device)
    metrics = dict(name=args.name, steps=cfg.max_steps, train_sec=round(train_sec, 1),
                   ms_per_step=round(1000 * train_sec / cfg.max_steps, 1),
                   full_val_loss=round(vl, 4), params=model.num_params(), seed=cfg.seed)
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=1)
    torch.save({"model": model.state_dict(), "config": cfg.to_dict()}, os.path.join(out_dir, "model.pt"))
    log_f.close()
    samples_f.close()
    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
