# 実験方針メモ

作成日: 2026-05-29
対象リポジトリ: `DInov2-IIC`

## 1. リポジトリ分析

このリポジトリは、腎臓糸球体画像を教師なしでクラスタリングし、糖尿病性病変の候補や形態サブタイプを探索するための Python パッケージです。現状の中心は、学習済み DINOv2 による特徴抽出、k-means ベースライン、DINOv2 を凍結した IIC クラスタヘッド学習、クラスタごとの画像書き出し、簡易可視化です。

主要ファイルは以下です。

- `README.md`: 研究目的、データ構造、推奨ロードマップ、コマンド例がまとまっている。
- `CODEX.md`: 実装済み範囲と次作業が整理されている。
- `configs/default.yaml`: 標準設定。`dataset/mouse_dataset`、Excelメタデータ、DINOv2設定、クラスタ数候補、IIC設定を定義している。
- `src/dinov2_iic/metadata.py`: `dataset/<name>/diabetes` と `not-diabetes` を走査し、画像パス、mouse_id、section_id、糖尿病ラベル、週齢、腎臓番号を結合する。
- `src/dinov2_iic/features.py`: `torch.hub.load("facebookresearch/dinov2", model_name)` でDINOv2を読み込み、CLS tokenなどの特徴を `.npz` に保存する。
- `src/dinov2_iic/clustering.py`: k-means、クラスタ指標、`assignments.csv`、代表画像、HTMLサムネイル、分析レポートを出力する。
- `src/dinov2_iic/iic.py`: DINOv2を凍結し、MLPクラスタヘッドのみを IIC loss で学習する。
- `src/dinov2_iic/visualization.py`: レビュー用のラベル付き画像と、IICヘッド向けの occlusion heatmap を作る。

現在、`dataset/` と `outputs/` はまだ存在しません。`.gitignore` では `outputs/`、`*.npz`、`*.pt`、`*.pth` は除外されていますが、`dataset/` はまだ除外されていません。

## 2. 現状評価

実験開始前の骨格はかなり揃っています。最初から IIC に進むより、README の方針どおり、まずは frozen DINOv2 特徴量によるベースラインを作るのが妥当です。

良い点:

- 糖尿病ラベルと週齢を学習に使わず、クラスタ後の評価に回す設計になっている。
- Excel の個体情報と画像フォルダラベルの不一致を検出する実装がある。
- `mouse_id` を単位にした集計の入口がある。
- k-means と IIC の両方が CLI から実行できる。
- 出力に `assignments.csv` とクラスタ代表画像が含まれ、病理レビューに進みやすい。

注意点:

- 現在の k-means は画像単位で評価しており、個体単位の分割や安定性評価はまだ薄い。
- `dataset/` が `.gitignore` に入っていないため、大量画像を置いたあと誤って Git 管理対象にしてしまうリスクがある。
- DINOv2 のロードに `torch.hub` を使うため、初回実行時にネットワークまたはキャッシュが必要になる可能性がある。
- IIC の augmentation は暫定的で、病理画像として強すぎないかを代表画像で確認する必要がある。
- occlusion heatmap は候補可視化であり、病理学的根拠として断定しない方がよい。

## 3. 最初の実験方針

### Phase 0: データ配置とメタデータ検証

目的は、画像と Excel メタデータが正しく対応するかを確認することです。

推奨配置:

```text
dataset/
  mouse_dataset/
    diabetes/
      <section_id>/
        *.jpg
    not-diabetes/
      <section_id>/
        *.jpg
```

最初に実行するコマンド:

```powershell
dinov2-iic --config configs/default.yaml prepare-metadata
```

確認項目:

- `outputs/metadata.csv` の行数が画像数に近いか。
- `mouse_id` が 217 匹に対応しているか。
- `A=diabetes`, `WT=not-diabetes` の対応に不一致がないか。
- `age_week` と `kidney_number` に欠損がないか。
- 同一 `mouse_id` の画像が複数 section にまたがる場合でも自然に集計できるか。

### Phase 1: 小さなサブセットで動作確認

75,000枚をいきなり処理する前に、各群・各週齢から少量を選んで、パイプライン全体を確認します。

推奨サブセット:

- 各ラベルから 5〜10 匹程度。
- 各週齢ができるだけ入るようにする。
- 合計 1,000〜3,000 画像程度から開始する。

確認したいこと:

- DINOv2 の特徴抽出がGPUで動くか。
- 画像読み込みエラーが出ないか。
- `features_dinov2.npz` の行数が `metadata.csv` と一致するか。
- k-means の `assignments.csv` とクラスタ画像が出るか。

### Phase 2: frozen DINOv2 + k-means ベースライン

IICの前に、学習済みDINOv2特徴が形態差を拾えているか確認します。

推奨条件:

- `K=4, 8, 16, 32`
- `feature_type=cls` を初期値にする。
- 余裕があれば `patch_mean` と `cls_patch_mean` も比較する。
- seedを複数変えてクラスタ安定性を見る。

見るべき出力:

- `analysis_report.md`
- `cluster_summary.csv`
- `assignments.csv`
- `clusters/cluster_xx/thumbnails.html`

評価観点:

- クラスタごとの代表画像に形態的な一貫性があるか。
- 糖尿病群に偏るクラスタがあるか。
- その偏りが週齢差や特定個体への偏りではないか。
- 1匹または少数個体だけで成立するクラスタがないか。
- 背景、余白、明るさ、切り出しサイズなどの技術差を拾っていないか。

### Phase 3: IIC 学習

ベースラインで形態的に意味のあるまとまりが見えてから IIC に進みます。

初期条件:

- DINOv2 は凍結。
- `n_clusters=8` と `16` を優先。
- `epochs=20` はまず妥当だが、損失だけで判断しない。
- 学習後は `assign-iic` で全画像にクラスタを割り当てる。

比較観点:

- k-means より代表画像のまとまりが良くなるか。
- 糖尿病ラベルや週齢への偏りが強くなりすぎていないか。
- クラスタ崩壊が起きていないか。
- 複数seedで似たクラスタが再現するか。

### Phase 4: 病理レビュー用パッケージ化

有望なクラスタが出たら、病理レビューしやすい形式に整えます。

最低限まとめるもの:

- クラスタごとの代表画像タイル。
- 糖尿病比率、週齢分布、個体数、腎臓番号分布。
- クラスタ中心例と境界例。
- IICの場合は確信度が高い例とエントロピーが高い例。
- heatmap は補助情報として添付し、判断根拠として断定しない。

## 4. データセット配置と Codex トークン消費について

結論として、リポジトリにデータセットを配置するだけで Codex のトークンを激しく消費することは通常ありません。Codex がトークンを消費するのは、ファイルの中身や一覧を会話コンテキストへ読み込んだときです。画像ファイルそのものを置くだけなら、Codex は自動で全画像を読みません。

ただし、次の操作は読み込みが重くなったり、出力が巨大になったりする可能性があります。

- `Get-ChildItem -Recurse` や `rg --files` で `dataset/` 全体を大量列挙する。
- 画像ファイル名を何万行もそのまま表示する。
- 生成された `outputs/` の大量ファイルを全列挙する。
- notebookやCSVを丸ごと表示する。
- 画像を個別に多数開いて視覚解析する。

安全に運用するための推奨:

- `dataset/` は置いてよいが、Codex に全列挙させない。
- 件数確認は `Measure-Object` のような集計だけにする。
- ファイル一覧を見る場合は `Select-Object -First 20` などで先頭だけ見る。
- `dataset/` と `outputs/` は Git 管理から除外する。
- 大量画像の確認は、代表画像だけを抽出した HTML やサムネイルで見る。

追加で推奨する `.gitignore` 追記候補:

```gitignore
dataset/
```

この追記は、データセットをリポジトリ直下に置く運用ならかなり重要です。データを Git に入れたい特殊な理由がない限り、次に最初に入れてよい変更です。

## 5. 直近の実行順

1. `.gitignore` に `dataset/` を追加する。完了。
2. `dataset/rat_PAS_crop_longside_1.3/diabetes` と `dataset/rat_PAS_crop_longside_1.3/not_diabetes` に画像を配置する。完了。
3. 画像数を集計だけで確認する。完了。`diabetes=42,719`, `not_diabetes=33,628`, 合計 `76,347` 画像。
4. `configs/default.yaml` を実データの配置に合わせる。完了。
5. `prepare-metadata` を実行する。
6. `outputs/metadata.csv` の件数、個体数、ラベル分布、週齢分布を確認する。
7. 小さなサブセットで `extract-features` と `cluster` を動かす。
8. 問題なければ全量で `K=4,8,16,32` の k-means を実行する。
9. 代表画像と集計を見て、IIC に進むか判断する。

## 7. 2026-05-29 実データ確認メモ

実データの本体は `dataset/rat_PAS_crop_longside_1.3` に配置されている。ラベルフォルダ名は `diabetes` と `not_diabetes` であり、初期READMEに書かれていた `not-diabetes` とは異なるため、`configs/default.yaml` の `dataset.negative_dir` を `not_diabetes` に変更した。

確認済み件数:

- `diabetes`: 42,719 images, 498 section folders
- `not_diabetes`: 33,628 images, 379 section folders
- total: 76,347 images

`dataset/説明.txt` によると、`rat_PAS_crop_longside_1.3` はセグメンテーションした元画像で、セグメンテーション時に 512 x 512 pixels へリサイズされている。`json` と `masks` は補助データであり、まずは `rat_PAS_crop_longside_1.3` の画像だけを使ってメタデータ作成とDINOv2特徴抽出を行う。

## 8. 2026-05-29 初期実行ログ

GPU確認:

- `.venv` に CUDA 対応 PyTorch `2.5.1+cu121` を導入した。
- `torch.cuda.is_available()` は `True`。
- GPUは `NVIDIA RTX 5000 Ada Generation` が2枚、各約32 GiB。
- CUDAテンソルの smoke test は成功。

メタデータ作成:

- `dinov2-iic --config configs/default.yaml prepare-metadata` は成功。
- `outputs/metadata.csv` は 76,347 行。
- `mouse_id` は216匹。Excel上は217匹だが、67番は `dataset/67(data setに含めない)` に分けられているため、本体データには含めない。
- ラベル内訳は `diabetes=42,719`, `not-diabetes=33,628`。
- 週齢内訳は `8w=18,184`, `12w=18,702`, `16w=19,460`, `20w=20,001`。
- 欠損はなし。

小サブセット smoke test:

- `outputs/subsets/metadata_smoke.csv` を作成。64画像、32匹、ラベルと週齢を均等化。
- DINOv2 `dinov2_vits14` のCLS特徴抽出は成功。出力は `outputs/subsets/features_smoke_dinov2.npz`、shape は `(64, 384)`。
- `K=4` の k-means は成功。出力は `outputs/smoke_k4`。

全量実験に入る前の修正:

- 全量76,347画像に対する silhouette score の全組み合わせ計算は重すぎるため、`src/dinov2_iic/clustering.py` を変更し、silhouette は最大5,000画像のサンプルで評価する。

全量 DINOv2 特徴抽出:

- `outputs/features_dinov2_cls_full.npz` を作成。
- 対象は `outputs/metadata.csv` の全 76,347 画像。
- モデルは `dinov2_vits14`、特徴は `cls`。
- 出力shapeは `(76347, 384)`、dtypeは `float32`。
- 実行時間はおよそ18分。`cuda:0`, batch size 128。

全量 k-means:

| K | silhouette(sample=5000) | Davies-Bouldin | min cluster images | max cluster images | max diabetes ratio | min diabetes ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 0.0345 | 3.5263 | 14,098 | 22,357 | 0.6601 | 0.4664 |
| 8 | 0.0195 | 3.5685 | 4,903 | 11,694 | 0.7210 | 0.3304 |
| 16 | 0.0099 | 3.4632 | 2,113 | 6,360 | 0.7661 | 0.2747 |
| 32 | 0.0067 | 3.4626 | 870 | 3,545 | 0.8082 | 0.2212 |

出力:

- `outputs/full_k4`
- `outputs/full_k8`
- `outputs/full_k16`
- `outputs/full_k32`
- `outputs/full_kmeans_comparison.csv`

初期所見:

- silhouette は全体的に低く、DINOv2 CLS特徴の空間で明確に分離した大クラスタがあるというより、連続的な形態差を k-means で切っている可能性が高い。
- Kを大きくすると糖尿病比率の偏りが強いクラスタが出る。特に `K=32` では糖尿病寄り `cluster_16`、`cluster_24`、`cluster_19` と、非糖尿病寄り `cluster_01`、`cluster_11`、`cluster_20` が候補。
- 一方で各クラスタには多くの個体が含まれており、少数個体だけの極端なクラスタではない。これは探索候補としては悪くない。
- 次は代表画像HTMLを人間が確認し、糖尿病比率の偏りが形態差なのか、染色・明るさ・余白・セグメンテーション品質の差なのかを見分ける。

次に見る候補:

- `outputs/full_k32/clusters/cluster_16/thumbnails.html`
- `outputs/full_k32/clusters/cluster_24/thumbnails.html`
- `outputs/full_k32/clusters/cluster_19/thumbnails.html`
- `outputs/full_k32/clusters/cluster_01/thumbnails.html`
- `outputs/full_k32/clusters/cluster_11/thumbnails.html`
- `outputs/full_k32/clusters/cluster_20/thumbnails.html`

## 9. 2026-05-29 特徴表現比較

IICに進む前に、`cls` 以外のDINOv2特徴でも形態差と糖尿病・非糖尿病の偏りが再現するかを確認した。

追加で作成した特徴:

- `outputs/features_dinov2_patch_mean_full.npz`: patch token平均、shape `(76347, 384)`。
- `outputs/features_dinov2_cls_patch_mean_full.npz`: CLSとpatch token平均の連結、shape `(76347, 768)`。

追加クラスタリング:

- `outputs/full_patch_mean_k32`
- `outputs/full_cls_patch_mean_k32`

比較出力:

- `outputs/feature_type_k32_comparison.csv`
- `outputs/feature_type_k32_candidate_clusters.csv`
- `outputs/feature_type_k32_assignment_agreement.csv`

K=32 比較:

| feature | silhouette(sample=5000) | Davies-Bouldin | max diabetes ratio | min diabetes ratio |
|---|---:|---:|---:|---:|
| `cls` | 0.0067 | 3.4626 | 0.8082 | 0.2212 |
| `patch_mean` | 0.0103 | 3.4472 | 0.8190 | 0.3053 |
| `cls_patch_mean` | 0.0090 | 3.4634 | 0.8090 | 0.2807 |

割当一致:

| pair | ARI | NMI |
|---|---:|---:|
| `cls` vs `patch_mean` | 0.2200 | 0.4495 |
| `cls` vs `cls_patch_mean` | 0.2956 | 0.5227 |
| `patch_mean` vs `cls_patch_mean` | 0.2532 | 0.5003 |

所見:

- `patch_mean` は silhouette と Davies-Bouldin がわずかに良く、糖尿病寄りクラスタの最大比率も `0.819` と最も高い。
- `cls_patch_mean` は `cls` との割当一致が最も高く、CLS由来の構造を保ちながらpatch情報も入っている。
- 3種類すべてで糖尿病寄りクラスタと非糖尿病寄りクラスタが出ており、最初の `cls` の形態差は単一特徴表現だけの偶然ではなさそう。
- ただし ARI は中程度以下なので、クラスタ番号そのものが完全に再現しているわけではない。病理レビューでは「同一クラスタの再現」より「同じ形態テーマが複数特徴で出るか」を確認する。

次に見る候補:

- `outputs/full_patch_mean_k32/clusters/cluster_17/thumbnails.html`: diabetes ratio `0.819`
- `outputs/full_patch_mean_k32/clusters/cluster_09/thumbnails.html`: diabetes ratio `0.785`
- `outputs/full_patch_mean_k32/clusters/cluster_30/thumbnails.html`: diabetes ratio `0.305`
- `outputs/full_patch_mean_k32/clusters/cluster_01/thumbnails.html`: diabetes ratio `0.310`
- `outputs/full_cls_patch_mean_k32/clusters/cluster_23/thumbnails.html`: diabetes ratio `0.809`
- `outputs/full_cls_patch_mean_k32/clusters/cluster_09/thumbnails.html`: diabetes ratio `0.772`
- `outputs/full_cls_patch_mean_k32/clusters/cluster_01/thumbnails.html`: diabetes ratio `0.281`
- `outputs/full_cls_patch_mean_k32/clusters/cluster_07/thumbnails.html`: diabetes ratio `0.281`

現時点の推奨:

1. 病理レビューでは `cls`, `patch_mean`, `cls_patch_mean` の上位候補を横断して見る。
2. 形態差が最も明瞭な特徴表現を主解析に採用する。指標だけなら `patch_mean` が第一候補。
3. 代表画像確認で同じテーマが複数特徴にまたがって出るなら、次に IIC へ進む。
4. IICの前に、クラスタごとの高確信度例、境界例、個体別比率を出すレビュー用集計を追加する。

## 10. 2026-05-29 レビュー用集計

`scripts/build_review_package.py` を追加し、`K=32` の3特徴表現比較に対してレビュー用集計を作成した。

出力先:

- `outputs/review_k32_feature_comparison`

主な出力:

- `outputs/review_k32_feature_comparison/README.md`
- `outputs/review_k32_feature_comparison/candidate_clusters.csv`
- `outputs/review_k32_feature_comparison/<feature>/cluster_<id>_<direction>/high_confidence_examples.csv`
- `outputs/review_k32_feature_comparison/<feature>/cluster_<id>_<direction>/boundary_examples.csv`
- `outputs/review_k32_feature_comparison/<feature>/cluster_<id>_<direction>/mouse_summary.csv`
- `outputs/review_k32_feature_comparison/<feature>/cluster_<id>_<direction>/top_mice.csv`
- `outputs/review_k32_feature_comparison/<feature>/cluster_<id>_<direction>/age_label_table.csv`
- `outputs/review_k32_feature_comparison/<feature>/cluster_<id>_<direction>/kidney_label_table.csv`

対象クラスタ:

- 各特徴表現について、糖尿病寄り上位5クラスタ、非糖尿病寄り上位5クラスタ。
- 合計30クラスタ。

レビュー時の見方:

- `high_confidence_examples.csv` は k-means 中心に近い代表例。
- `boundary_examples.csv` は k-means 中心から遠い境界例。
- ここでの `cluster_probability` は真の確率ではなく、k-means距離から作った便宜的な指標。病理レビューでは値そのものより、中心例と境界例の並びとして扱う。
- `mouse_summary.csv` では、特定個体がクラスタを支配していないか、また個体内の何割がそのクラスタに入っているかを見る。
- `age_label_table.csv` と `kidney_label_table.csv` では、糖尿病偏りが週齢や腎臓番号の偏りで説明されすぎていないかを確認する。

確認済み例:

- `outputs/review_k32_feature_comparison/patch_mean/cluster_17_diabetes_enriched`
- このクラスタは `diabetes_ratio=0.819`。
- `high_confidence_examples.csv`, `boundary_examples.csv`, `mouse_summary.csv`, `top_mice.csv`, `age_label_table.csv`, `kidney_label_table.csv` が正常に生成された。

次の推奨:

1. `outputs/review_k32_feature_comparison/README.md` を入口に、各候補クラスタの代表画像HTMLとレビューCSVを確認する。
2. 病理的に意味がありそうな形態テーマを、特徴表現をまたいでメモする。
3. 形態テーマが再現するクラスタを優先して、IICの初期K候補を決める。
4. 必要なら、レビュー用に画像コピー付きの小さな提出パッケージを別途作る。

## 11. 2026-05-29 メタデータ可視化コマンド

`inspect-metadata` コマンドを追加した。画像数、個体数、ラベル分布、週齢分布、週齢 x ラベル、腎臓番号分布、個体別画像数を、CSV、Markdown、HTMLで確認できる。

追加実装:

- `src/dinov2_iic/inspection.py`
- `src/dinov2_iic/cli.py` の `inspect-metadata` サブコマンド

実行コマンド:

```powershell
dinov2-iic --config configs/default.yaml inspect-metadata --metadata outputs/metadata.csv --output-dir outputs/metadata_inspection --top-mice 30
```

出力:

- `outputs/metadata_inspection/inspection.html`
- `outputs/metadata_inspection/inspection_report.md`
- `outputs/metadata_inspection/overview.csv`
- `outputs/metadata_inspection/label_counts.csv`
- `outputs/metadata_inspection/age_counts.csv`
- `outputs/metadata_inspection/label_age_counts.csv`
- `outputs/metadata_inspection/label_age_section_counts.csv`
- `outputs/metadata_inspection/kidney_counts.csv`
- `outputs/metadata_inspection/mouse_image_counts.csv`
- `outputs/metadata_inspection/top_mouse_image_counts.csv`

確認済みの概要:

- images: 76,347
- mice: 216
- sections: 877
- labels: 2
- ages: 4
- kidney_numbers: 43
- missing_values: 0
- min_images_per_mouse: 116
- median_images_per_mouse: 352
- max_images_per_mouse: 508
- diabetes: 42,719 images
- not-diabetes: 33,628 images
- age 8: 18,184 images
- age 12: 18,702 images
- age 16: 19,460 images
- age 20: 20,001 images

使い方:

- `inspection.html` を最初に開いて、棒グラフと表で全体像を確認する。
- 正確な数値確認や論文・レポート用の転記にはCSVを使う。
- `mouse_image_counts.csv` で個体ごとの画像数偏りを確認し、クラスタ解析で特定個体に引っ張られていないかを見る。

## 12. 2026-05-29 IIC本番前整備とW&B対応

IIC本番実験に入る前に、`train-iic` を長時間実験向けに整備した。

追加・変更:

- `src/dinov2_iic/iic.py`
- `src/dinov2_iic/cli.py`
- `requirements.txt`
- `pyproject.toml`
- `.gitignore`

追加した主な機能:

- epochごとの `loss`, `used_clusters`, `max_cluster_fraction`, `mean_entropy` を `train_log.csv` に保存。
- epochごとのクラスタ使用率を `cluster_usage_by_epoch.csv` に保存。
- `run_config.json` に実験設定を保存。
- `training_report.md` に学習中のクラスタ使用率ベースの崩壊チェックを保存。
- 学習後に自動で `assignments.csv`, `cluster_summary.csv`, `analysis_report.md` を作成。
- `cluster_summary.csv` には空クラスタも含める。
- `analysis_report.md` に最終割当ベースの崩壊警告を出す。
- W&B loggingに対応。`--wandb-mode disabled` がデフォルトで、`online` / `offline` を明示すると有効になる。
- W&B offline出力用に `.gitignore` に `wandb/` を追加。

追加CLI引数:

```powershell
--seed
--representative-count
--no-copy-images
--no-assign-after-train
--wandb-project
--wandb-entity
--wandb-run-name
--wandb-mode disabled|online|offline
--wandb-tags
```

W&B offline smoke test:

```powershell
dinov2-iic --config configs/default.yaml train-iic `
  --metadata outputs/subsets/metadata_smoke.csv `
  --output-dir outputs/iic_smoke_wandb_offline_k8 `
  --n-clusters 8 `
  --epochs 1 `
  --batch-size 16 `
  --device cuda:0 `
  --seed 42 `
  --no-copy-images `
  --wandb-project dinov2-iic-smoke `
  --wandb-mode offline `
  --wandb-run-name smoke_k8_offline `
  --wandb-tags smoke,iic
```

smoke test結果:

- W&B offline runは作成成功。
- `train_log.csv`, `cluster_usage_by_epoch.csv`, `iic_head.pt`, `run_config.json`, `assignments.csv`, `cluster_summary.csv`, `analysis_report.md` が生成された。
- 1 epochの小データでは最終割当が1クラスタに潰れたため、`analysis_report.md` の崩壊警告が `True` になった。これは本番前の検出経路確認として期待通り。

本番IICの初期推奨:

- まず `K=16` と `K=32`。
- `epochs=20`。
- `batch_size=64` または `128` をGPUメモリを見ながら選ぶ。
- 最初の本番runでは `--wandb-mode offline` でもよい。W&Bログイン済みなら `--wandb-mode online --wandb-project dinov2-iic` にする。
- 初回は `--no-copy-images` を付けて学習と割当の安定性を確認し、有望なrunだけ代表画像を書き出す。

## 13. 2026-05-29 W&B online付き IIC pilot

W&B onlineの初期化確認:

- `wandb.init(..., mode="online")` は成功。
- account: `kajinagi0601`
- entity/project: `kajinagi0601-kanazawa-university/dinov2-iic`
- online init check run: `https://wandb.ai/kajinagi0601-kanazawa-university/dinov2-iic/runs/8wxaqt4j`

IIC pilot subset:

- `outputs/subsets/metadata_iic_pilot.csv`
- 1,152 images
- 96 mice
- diabetes: 576
- not-diabetes: 576
- age 8/12/16/20: 各288 images
- label x age: 各セル144 images

実行条件:

- model: `dinov2_vits14`
- image_size: 224
- batch_size: 64
- device: `cuda:0`
- seed: 42
- epochs: 10
- `--no-copy-images`
- W&B: `--wandb-mode online --wandb-project dinov2-iic`

実行したrun:

- `outputs/iic_pilot_k16_e10_wandb`
- `outputs/iic_pilot_k32_e10_wandb`

W&B run:

- K=16: `https://wandb.ai/kajinagi0601-kanazawa-university/dinov2-iic/runs/1zp3mcn4`
- K=32: `https://wandb.ai/kajinagi0601-kanazawa-university/dinov2-iic/runs/df39jvo7`

比較:

| run | final_loss | train_used_clusters | assign_used_clusters | max_cluster_fraction | max_diabetes_ratio | min_diabetes_ratio | mean_entropy_weighted |
|---|---:|---:|---:|---:|---:|---:|---:|
| `iic_pilot_k16_e10` | -1.1510 | 15/16 | 14/16 | 0.3411 | 0.7500 | 0.0000 | 0.2493 |
| `iic_pilot_k32_e10` | -1.3115 | 24/32 | 17/32 | 0.2118 | 0.6433 | 0.0000 | 0.4399 |

所見:

- 両方ともW&B online同期に成功。
- 両方とも最終割当の崩壊警告は `False`。
- K=16はラベル偏りがやや強いが、最大クラスタが34%あり、粗めの分割。
- K=32は学習中の使用クラスタが24/32まで増え、最終割当でも17/32クラスタを使用。分散は良いが、pilot subsetではラベル偏りはK=16より弱い。
- `min_diabetes_ratio=0` はごく小さいクラスタ由来なので過解釈しない。
- pilot subsetでは、IICが正常に学習・記録・割当まで動くこと、W&B onlineが使えること、崩壊検出が機能することを確認できた。

次の推奨:

1. 全量に進む前に、中規模subsetを作って `K=16`, `K=32`, `epochs=20` を比較する。
2. 全量IICはかなり時間がかかる見込み。まずは `--no-copy-images` で `K=16` を1本走らせる。
3. 全量runの候補コマンド:

```powershell
dinov2-iic --config configs/default.yaml train-iic `
  --metadata outputs/metadata.csv `
  --output-dir outputs/iic_full_k16_e20_wandb `
  --n-clusters 16 `
  --epochs 20 `
  --batch-size 64 `
  --device cuda:0 `
  --seed 42 `
  --no-copy-images `
  --wandb-project dinov2-iic `
  --wandb-mode online `
  --wandb-run-name iic_full_k16_e20
```

## 6. 次にコード面で足すとよいもの

優先度の高い改善:

- `.gitignore` に `dataset/` を追加する。
- 画像数、個体数、ラベル分布、週齢分布を表示する `inspect-metadata` コマンドを追加する。
- k-means の複数K一括実行コマンドを追加する。
- 個体単位の偏りを `cluster_summary.csv` により詳しく出す。
- seed違いのクラスタ安定性を比較するスクリプトを追加する。

急がなくてよい改善:

- UMAP/PCAプロット出力。
- overclustering用IICヘッド。
- ViT向けattention rolloutやtoken attribution。
- 画像正規化や染色正規化の比較。
