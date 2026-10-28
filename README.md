# 演習2：小さなGPTを育てて、壊して、覗く

2026年度「機械学習応用」［情報工学III／広域システム特論V（７）］演習2の配布リポジトリです。
文章を1文字ずつ生成する小さなGPTを、グループでゼロから学習させ、部品を外したり変えたりして仕組みを確かめます。

## 最終報告までにやること

| 時期 | グループでやること |
|---|---|
| 10/30（第5回）授業内 | グループ名を決め、このリポジトリをテンプレートにしてグループの Private リポジトリを作る（下の手順）。全員が GitHub のトークンを作り、Kaggle に登録する。全員が Kaggle の電話番号認証を済ませる／`model.py` を読み、講義の手順がどの行に当たるかを確かめる／担当の問いで、何を変えて何と比べるかを話し合う |
| 11/6（第6回）までに | 課題①：標準構成 S を seed 0・1 で学習し、結果を `results/` に残す／課題②の予想を、プルリクエストで提出する／担当の問いの、試す条件と予想を、プルリクエストで提出する（[questions.md](questions.md)） |
| 11/6（第6回）授業内 | 課題②：causal mask を外す実験／担当の問いの実験を始める |
| 11/13（第7回）までに | 担当の問いの必須の比較（グループの人数と同じ数）を seed を変えて複数回行い、結果を `results/` に残す／（余力があれば）覗く：attention weight を見る／最終報告のスライドを、テンプレートに書き込む |

第7回（11/13）は発表です。

## ファイル構成

| ファイル | 中身 |
|---|---|
| `model.py` | GPT 本体（Self-attention、MLP、Transformerブロック）。部品を変える実験では主にここを編集する |
| `config.py` | モデルの大きさ（XS／S／M／L）と学習の設定。標準構成は S（4層、4ヘッド、256次元、約420万パラメータ） |
| `prepare.py` | 青空文庫のデータを準備する（約2,000万字、文字単位、語彙4,000文字） |
| `train.py` | 学習する。結果は `runs/<実験名>/` に保存される |
| `generate.py` | 学習したモデルで文章を生成する |
| `attention.py` | 学習したモデルの attention weight をヒートマップにする（「覗く」） |
| `kaggle_notebook.ipynb` | Kaggle でこのリポジトリを動かすためのノートブック |
| `questions.md` | 各グループの問いと、必須の比較・発展・注意 |
| `expectations/` | 実験前の予想を書くところ（`TEMPLATE.md` をコピーして使う） |
| `results/` | 実験結果（設定・ログ・生成文）を置くところ |

## グループのリポジトリの作り方

1. グループの代表者が、このページ右上の **Use this template** → **Create a new repository** を押す
2. Owner は代表者のアカウント、Repository name はグループ名などにし、**Private** を選ぶ
3. 作ったリポジトリの Settings → Collaborators から、メンバー全員と教員（`yukinobaba`）を招待する。招待された人はメールか GitHub の通知から承認する

以降の変更（予想、モデルの変更、実験結果）は、すべてこのグループのリポジトリにプルリクエストで入れます。
リポジトリは Private なので、他のグループからは予想や結果は見えません。

## Kaggle から Private リポジトリを読むためのトークン

Kaggle のノートブックはグループのリポジトリを clone して使います。Private リポジトリを読むには、各自の GitHub のトークンが必要です。

1. https://github.com/settings/tokens/new?scopes=repo&description=kaggle-mlapp2026 を開く（GitHub の Settings → 左メニューいちばん下の Developer settings → Personal access tokens → **Tokens (classic)** → Generate new token (classic) と同じ画面）
   - Note：`kaggle-mlapp2026` など
   - Expiration：30 days（演習2が終わるまで使えればよい）
   - Scopes：**repo** にだけチェック
   - リポジトリの持ち主（代表者）は、代わりに Fine-grained token（対象をグループのリポジトリだけ、Contents を Read-only）でもよい。他のメンバーは、他人のリポジトリを Fine-grained token で選べないので classic を使う
2. いちばん下の Generate token を押し、表示されたトークンをコピーする（この画面を離れると二度と表示されない）
3. Kaggle のノートブックの編集画面で、上のメニューの「Add-ons」→「Secrets」→「Add Secret」を開き、Label を `GITHUB_TOKEN`、Value をトークンにして保存する。一覧の `GITHUB_TOKEN` にチェックが入っている（このノートブックで使える）ことを確かめる

トークンはパスワードと同じです。ノートブックのコードやリポジトリ、チャットに貼らないでください。演習2が終わったら GitHub の同じ画面から削除してください。

## Kaggle での実行

1. Kaggle アカウントの電話番号認証を済ませる（GPU を使うのに必要）
2. `kaggle_notebook.ipynb` をダウンロードし、Kaggle にアップロードする（Kaggle の「Create」→「New Notebook」→「File」→「Import Notebook」）
3. 右側の「Session options」で Accelerator を **GPU T4 x2** にし、Internet を on にする
4. ノートブック先頭の `REPO_URL` をグループのリポジトリにする（`USE_TOKEN = True` のまま）
5. 学習は **Save Version（Save & Run All）** で実行する。対話セッションは途中で止まることがある。結果は保存したバージョンの「Output」から取り出せる

GPU は1人あたり週30時間まで使えます。標準構成 S の学習1回（5,000 step）は約5〜6分です（最初の数十秒はコンパイル）。
T4 は2枚使えるので、seed 違いなどの2本を同時に学習させられます（ノートブックに例があります）。

手元のPCでも `python prepare.py`、`python train.py --name test --size XS --max_steps 300` のように動きます（GPU がないと遅い）。

## 課題①：育てる（11/6までに）

1. 標準構成 S をゼロから学習させる：`python train.py --name base_seed0`
2. `runs/base_seed0/samples.txt` で、途中の step（100, 300, 1000, 2000, 3000, 5000）の生成文を比べる。何 step 目で単語らしく、文らしくなるか
3. `runs/base_seed0/log.csv` で train loss と val loss の推移を見る
4. seed だけを変えてもう一度学習し（`--seed 1`）、`metrics.json` の `full_val_loss` のばらつきを確かめる。このばらつきが、今後「差がある」と言えるかの基準になる

## 課題②：causal mask を外す（予想は11/6までに提出、実験は第6回）

1. **実験の前に**、グループで予想とその理由を書いてプルリクエストで提出する（`expectations/causal_mask.md`）
   - 学習時の loss はどうなるか
   - 生成される文章はどうなるか
2. `model.py` の `SelfAttention` を変更して、causal mask を外す（変更もプルリクエストで）
3. 学習させて、loss と生成文を予想と比べる。違ったら、なぜ違ったのかを説明する
4. 演習1で扱った「リーク」との関係も考えてみる

## 各グループの問い（第6回）

各グループは担当の問いについて、グループの人数と同じ数の必須の比較を行います（1人1つを担当、余力があれば発展も）。内容と注意は [questions.md](questions.md) を見てください。

| グループ | 問い |
|---|---|
| F | 位置の情報はどこまで必要か |
| G | 何が学習を安定させているか |
| H | 大きさとデータをどう配分するか |

## （余力があれば）覗く：attention weight を見る

学習したモデルに文を入れ、各層・各ヘッドの attention weight（どの文字にどれだけ注目しているか）を見ます。

```
python attention.py runs/base_seed0 --text "吾輩は猫である。名前はまだ無い。" --pos 7
```

- `runs/base_seed0/attention.png` に、層ごと・ヘッドごとのヒートマップを保存する（縦：注目する位置、横：見る位置）
- 各ヘッドが「平均で何文字前を見ているか」を表示する
- `--pos` で指定した位置（0始まり）の文字が、どの文字にどれだけ重みを置いているかを表示する

ヘッドや層によって見方がどう違うか、causal mask を外したモデルや担当の問いで変えたモデルでは基準とどう違うかを見てみてください。
見て分かったことは、発表に入れてかまいません。

## 実験の進め方のルール

- **予想が先**：実験の前に予想を書き、プルリクエストで提出する（提出時刻が残る）
- **1人1本以上**：メンバー全員が、プルリクエストを1本以上作り、自分以外のプルリクエストを1本以上レビューする
- **再現できるように**：実験名・seed・設定はすべて `runs/<実験名>/config.json` に残る。結果は `config.json`、`log.csv`、`samples.txt`、`metrics.json` を `results/<実験名>/` にコピーしてプルリクエストで残す（`model.pt` は大きいので入れない）
- **差を主張するときは複数の seed で**：seed を変えた結果のばらつきより大きい差であることを示す。比較には `metrics.json` の `full_val_loss`（検証データ全体での loss）を使う
- **コードは AI に書かせてよい**。ただし、モデルとして何が変わるかを説明できない変更はマージしない
- **AI の使い方を記録する**：AI に任せたこと、AI が誤っていたことを記録し、発表で説明する

## データの出典

青空文庫のテキストを整形したデータセット [globis-university/aozorabunko-clean](https://huggingface.co/datasets/globis-university/aozorabunko-clean)（CC BY 4.0）を使っています。このうち新字新仮名の作品から約2,000万字を使います。
