"""青空文庫のデータを準備する（初回だけ。Kaggle では毎セッション30秒ほど）

データ：globis-university/aozorabunko-clean（CC BY 4.0）から新字新仮名の作品を約2,000万字
作品単位で学習用（train）と検証用（val）に分け、出現頻度の高い4,000文字を語彙にする
出力：data/train.bin, data/val.bin（文字IDの列）, data/vocab.json, data/info.json
"""
import argparse
import json
import os
from collections import Counter

import numpy as np

DATASET_REVISION = "42a9c9c0f1d67e6a5554d9bea4201973dc9b049c"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--n_chars", type=int, default=20_000_000)
    ap.add_argument("--vocab_size", type=int, default=4000)
    ap.add_argument("--val_frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if os.path.exists(f"{args.out}/train.bin"):
        print(f"{args.out}/train.bin がすでにあるので何もしません")
        return

    from datasets import load_dataset
    # 配布元が更新しても同じデータになるよう、バージョン（2023-10-27 のコミット）を固定する
    ds = load_dataset("globis-university/aozorabunko-clean", split="train", revision=DATASET_REVISION)
    metas = ds["meta"]
    key = next(k for k in metas[0] if "文字遣い" in k)  # 「文字遣い種別」
    texts = [t for t, m in zip(ds["text"], metas) if m.get(key) == "新字新仮名"]

    # 作品の順番をシャッフルし、合計が約2,000万字になるまで取る
    rng = np.random.default_rng(args.seed)
    picked, total = [], 0
    for i in rng.permutation(len(texts)):
        picked.append(texts[i])
        total += len(texts[i])
        if total >= args.n_chars:
            break
    n_val = max(1, int(len(picked) * args.val_frac))
    val_txt = "\n".join(picked[:n_val])
    train_txt = "\n".join(picked[n_val:])

    # 語彙：学習用データで多い順に vocab_size-1 文字。ID 0 は語彙外の文字（<unk>）
    counts = Counter(train_txt)
    chars = [c for c, _ in counts.most_common(args.vocab_size - 1)]
    stoi = {c: i + 1 for i, c in enumerate(chars)}
    coverage = sum(counts[c] for c in chars) / sum(counts.values())

    os.makedirs(args.out, exist_ok=True)
    for name, txt in (("train", train_txt), ("val", val_txt)):
        ids = np.array([stoi.get(c, 0) for c in txt], dtype=np.uint16)
        ids.tofile(f"{args.out}/{name}.bin")
    with open(f"{args.out}/vocab.json", "w", encoding="utf-8") as f:
        json.dump({"itos": ["<unk>"] + chars}, f, ensure_ascii=False)
    info = dict(dataset_revision=DATASET_REVISION, works=len(picked), val_works=n_val, train_chars=len(train_txt),
                val_chars=len(val_txt), vocab_size=args.vocab_size, coverage=round(coverage, 5))
    with open(f"{args.out}/info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=1)
    print(json.dumps(info, ensure_ascii=False))


if __name__ == "__main__":
    main()
