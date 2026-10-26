# 実験結果

実験ごとに `results/<実験名>/` を作り、`runs/<実験名>/` から次の4つをコピーしてプルリクエストで残してください（`model.pt` は入れない）。

- `config.json`：設定（seed、モデルの大きさ、git のコミット）
- `log.csv`：train loss と val loss の推移
- `samples.txt`：途中の step での生成文
- `metrics.json`：最終結果（`full_val_loss` を比較に使う）

「覗く」で作ったヒートマップ（`attention.png`）も、あれば一緒に置いてください。
