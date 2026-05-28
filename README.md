# Dinov2-IIC

腎臓糸球体画像を教師なしにクラスタリングし、人間の目視や既存ラベルだけでは見落としやすい糖尿病性病変の候補を探索するための研究リポジトリです。

想定する入力データは、プロジェクト直下の `dataset/` に複数のデータセットを置く構造です。実行時にはconfigで使用するデータセットを選択します。

```text
dataset/
  mouse_dataset/
    diabetes/
      1-left-1/
        1-left-1_rois_crop_0002_score0.90_longside.jpg
      1-left-2/
        1-left-2_rois_crop_0003_score0.90_longside.jpg
    not-diabetes/
      ...
  human_dataset/
    diabetes/
      ...
    not-diabetes/
      ...
```

現在の初期対象はマウスPAS染色標本です。

- 画像数: 約75,000枚
- サンプル数: 217匹
- 疾患ラベル: 糖尿病群と非糖尿病群
- 週齢: 8週、12週、16週、20週
- 画像状態: 糸球体単位で切り出し済み
- 染色法: PAS染色で統一
- 撮影・標本条件: 現時点では統一済みとして扱う
- 学習方針: 完全な教師なしクラスタリング
- ラベル利用: 糖尿病ラベルと週齢は学習には使わず、分析時のみ使用する

補足: 追加された `PAS染色標本_まとめ.xlsx` では、`Sheet1` に `sample_number`, `diabetes(A/WT)`, `age`, `kidney_number` が含まれています。Excel上の集計は A=122匹、WT=95匹、合計217匹です。この集計を正とし、`A=diabetes`, `WT=not-diabetes` として扱います。

設定ファイルは `configs/default.yaml` を標準とし、通常は `dataset.name` と `dataset.root` を変えるだけで別データセットに切り替えられる形にします。

```yaml
dataset:
  name: mouse_dataset
  root: dataset/mouse_dataset
  metadata_excel: PAS染色標本_まとめ.xlsx
  positive_dir: diabetes
  negative_dir: not-diabetes
  id_parse:
    source: parent_dir_then_filename
    pattern: leading_integer
    on_failure: error
```

## 目的

最終目的は、単に糖尿病群と非糖尿病群を分類することではありません。DINOv2で得られる形態特徴を使って糸球体画像をクラスタリングし、各クラスタに集まる形態的特徴、糖尿病群での偏り、週齢との関係、病理学的解釈を組み合わせて、新しい病変候補や既知病変のサブタイプを見つけることです。

そのため、このプロジェクトでは以下を重視します。

- 個体単位でのデータ分割と評価を行い、同一マウスの画像が学習・評価に混在することを防ぐ
- クラスタの見た目だけでなく、糖尿病ラベル、個体ID、週齢、腎臓番号との関連を確認する
- IICだけに依存せず、複数のクラスタリング手法・可視化手法を比較して再現性を見る
- Grad-CAM風の可視化は「根拠の候補」として扱い、病理学的妥当性は別途レビューする

## 推奨方針

当初案の「DINOv2 + MLP + IIC」は有力な候補ですが、医学画像探索では評価設計が特に重要です。以下の段階的な方針を推奨します。

### 1. データ整備

まず、画像パス、糖尿病ラベル、個体ID、週齢、腎臓番号を含むメタデータ表を作成します。

最低限必要な列:

```text
image_path, mouse_id, diabetes_label, age_week, kidney_number
```

Excel由来の列との対応:

```text
sample_number -> mouse_id
diabetes(A/WT) -> diabetes_label
age -> age_week
kidney_number -> kidney_number
```

画像パスからは、疾患ラベルとマウスIDを復元します。例えば、以下のようなパスを想定します。

```text
.../diabetes/1-left-1/1-left-1_rois_crop_0002_score0.90_longside.jpg
.../diabetes/1-left-2/1-left-2_rois_crop_0003_score0.90_longside.jpg
```

この場合、`diabetes` が疾患フォルダ、`1-left-1` や `1-left-2` は画像群をまとめるサブフォルダ名です。サブフォルダ名や画像ファイル名の詳細な形式には強く依存せず、先頭に現れる数値 `1` を `sample_number` として抽出し、Excelの個体情報と結合します。

同じマウスでも異なるWSIまたは切片に分かれることがありますが、クラスタリングの基本単位として重要なのは「何番のマウスか」です。そのため、`section_id` は分析補助用として保存してもよい一方、学習・評価・集計の主キーは `mouse_id` とします。

推奨するメタデータ列:

```text
image_path, mouse_id, section_id, diabetes_label, age_week, kidney_number
```

注意点:

- 学習・検証・解析は個体単位で分ける
- 糖尿病ラベルと週齢は学習には使わず、クラスタ後の分析にのみ使う
- 糖尿病群と非糖尿病群、週齢、腎臓番号の偏りがクラスタを支配していないか確認する
- 画像サイズ、染色、明るさ、背景領域の違いを前処理でそろえる
- 画像が糸球体中心に切り出されているか、余白・スケール差が大きすぎないか確認する

### 2. ベースライン特徴抽出

最初からDINOv2を学習せず、まずは学習済みDINOv2を凍結した特徴抽出器として使います。

推奨ベースライン:

- DINOv2のCLS tokenまたはpatch tokenを抽出
- PCAまたはUMAPで可視化
- k-means、Gaussian Mixture、HDBSCANなどでクラスタリング
- クラスタごとに代表画像、糖尿病比率、週齢分布、個体数、腎臓番号の偏りを確認

この段階で有望なクラスタ構造が見えない場合、IICで学習してもデータの偏りや前処理の問題を強めるだけになる可能性があります。

### 3. DINOv2 + IICモデル

ベースラインで形態的なまとまりが確認できたら、DINOv2の上にMLPクラスタヘッドを接続してIICで学習します。

基本構成:

```text
image
  -> DINOv2 encoder
  -> feature vector
  -> MLP projection head
  -> cluster logits
  -> softmax over K clusters
```

IICでは、同じ画像に異なるaugmentationをかけた2ビューを作り、それらのクラスタ割り当てが高い相互情報量を持つように学習します。

重要な設計:

- DINOv2は最初は凍結し、MLPだけ学習する
- 次に必要ならDINOv2の後段ブロックだけ低学習率でfine-tuningする
- augmentationは病理学的意味を壊さない範囲に制限する
- rotation/flip、軽いcolor jitter、軽いblur、crop/resizeを候補にする
- 強すぎる色変換や大きすぎるcropは、病変そのものを消す可能性があるため避ける
- cluster数 `K` は固定せず、複数候補で比較する
- overclustering用の補助ヘッドを追加し、細かい形態差を拾えるか確認する

推奨する実験条件:

```text
K = 4, 8, 16, 32
encoder = frozen DINOv2, partial fine-tuning DINOv2
head = 2-layer MLP, 3-layer MLP
features = CLS token, pooled patch tokens, CLS + pooled patch tokens
```

### 4. クラスタ評価

クラスタリングの成否は、損失値だけでは判断しません。以下を組み合わせて評価します。

定量評価:

- silhouette score、Davies-Bouldin indexなどのクラスタ分離指標
- 糖尿病ラベルとの関連: enrichment、chi-square test、mutual information
- 週齢との関連: 8週、12週、16週、20週の分布差
- 個体単位での偏り: 1匹のマウスだけで構成されるクラスタがないか
- 複数seedでの安定性: Adjusted Rand Index、Normalized Mutual Information
- 腎臓番号や撮影バッチとの関連

定性評価:

- クラスタごとの代表画像をタイル表示する
- クラスタ中心に近い画像と境界付近の画像を分けて確認する
- 病理医にblind reviewしてもらい、クラスタごとの共通所見を記録する
- 糖尿病に偏るクラスタが、単なる週齢差・個体差・腎臓番号差・画質差ではないことを確認する

### 5. クラスタごとの画像保存

学習後、各画像についてクラスタ確率を計算し、最大確率のクラスタに保存します。

推奨出力:

```text
outputs/
  experiment_001/
    config.yaml
    metrics.csv
    assignments.csv
    clusters/
      cluster_00/
        images/
        thumbnails.html
      cluster_01/
        images/
        thumbnails.html
    visualizations/
      cluster_00/
      cluster_01/
```

`assignments.csv` には以下を保存します。

```text
image_path, mouse_id, section_id, diabetes_label, age_week, kidney_number, cluster_id, cluster_probability, entropy
```

`cluster_probability` が低い画像や `entropy` が高い画像は、どのクラスタにも明確に属さない境界例として別途確認します。

### 6. 可視化

Grad-CAMはCNN向けに設計された手法なので、DINOv2のようなVision Transformerではそのまま使うより、ViTに適した可視化も比較します。

候補:

- Grad-CAM風のViT対応実装
- attention rollout
- token attribution
- Eigen-CAM
- occlusion sensitivity
- patch token similarity map

可視化の目的は「モデルがどこを見たか」を断定することではなく、「クラスタ割り当てに寄与した可能性のある領域を病理医が確認しやすくすること」です。

推奨出力:

```text
outputs/
  experiment_001/
    visualizations/
      cluster_00/
        mouse_1_1_original.jpg
        mouse_1_1_heatmap.jpg
        mouse_1_1_overlay.jpg
```

可視化結果はクラスタごとに代表例、典型例、境界例を分けて保存します。

## 実装ロードマップ

1. データセット読み込みとメタデータ作成
2. Excelの個体情報と画像パスの対応付け
3. 学習済みDINOv2による特徴抽出
4. k-meansなどの非学習クラスタリングでベースライン作成
5. クラスタごとの画像保存とHTMLサムネイル作成
6. DINOv2 + MLP + IICの学習実装
7. 複数クラスタ数・複数seedでの実験管理
8. クラスタ評価指標と個体単位集計の実装
9. ViT向け可視化手法の実装
10. 分析レポート作成

## 使い方

依存関係をインストールします。

```powershell
pip install -r requirements.txt
pip install -e .
```

メタデータを作成します。

```powershell
dinov2-iic --config configs/default.yaml prepare-metadata
```

DINOv2特徴量を抽出します。

```powershell
dinov2-iic --config configs/default.yaml extract-features
```

k-meansのベースラインクラスタリングを実行します。

```powershell
dinov2-iic --config configs/default.yaml cluster --n-clusters 8
```

メタデータ作成、特徴抽出、クラスタリングをまとめて実行する場合は以下を使います。

```powershell
dinov2-iic --config configs/default.yaml run-baseline --n-clusters 8
```

IICのクラスタヘッドを学習する場合は以下を使います。

```powershell
dinov2-iic --config configs/default.yaml train-iic --n-clusters 8
```

学習済みIICヘッドで全画像をクラスタへ割り当てる場合は以下を使います。

```powershell
dinov2-iic --config configs/default.yaml assign-iic --checkpoint outputs/<experiment>/iic_head.pt
```

クラスタリング結果からレビュー用の可視化ファイルを作成します。

```powershell
dinov2-iic --config configs/default.yaml visualize --assignments outputs/<experiment>/assignments.csv
```

学習済みIICヘッドに対して、画像領域を隠したときのクラスタ確率低下を可視化するocclusion感度マップを作成します。

```powershell
dinov2-iic --config configs/default.yaml visualize-iic --assignments outputs/<experiment>/assignments.csv --checkpoint outputs/<experiment>/iic_head.pt
```

主な出力は `outputs/` 以下に保存されます。

## 最初に作るべき成果物

最初のマイルストーンでは、IICの学習まで急がず、以下を完成させます。

- `metadata.csv`
- DINOv2特徴量ファイル
- PCA/UMAPプロット
- k-meansクラスタ結果
- クラスタごとの代表画像一覧
- 糖尿病ラベル・週齢・個体ID・腎臓番号との関連集計

この結果を見てから、IICで改善すべき点を決めます。

## 最終成果物

このプロジェクトでは、最終的に以下を出力します。

- クラスタリング済み画像フォルダ
- 学習済みモデル
- 分析レポート
- 画像ごとの判断根拠可視化結果

## 実装時の固定方針

現時点で大きな未確定事項はありません。実装時は以下を固定方針とします。

- configは `configs/default.yaml` を標準にする
- 疾患フォルダ名は `diabetes` / `not-diabetes` を標準にする
- サブフォルダ名・画像名から先頭のマウス番号を抽出できない画像があった場合はエラーにする

## 推奨する研究上の注意

糖尿病に偏ったクラスタが見つかっても、それが新規病変とは限りません。週齢差、個体差、腎臓番号差、標本品質、既知病変の強さなどを拾っている可能性があります。そのため、クラスタの発見は仮説生成として扱い、個体単位の再現性、別データでの再現性、病理学的レビューを通して検証します。
