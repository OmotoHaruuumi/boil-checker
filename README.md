# Boil Checker

Raspberry Pi とカメラを使い、**お湯の沸騰状態をリアルタイムで自動検知**する組み込み AI システムです。
軽量な 3 層 CNN を設計・学習し、TFLite への FP16 量子化とエッジ推論によって PC 不要の省メモリ動作を実現しています。

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)
![TFLite](https://img.shields.io/badge/TFLite-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Raspberry Pi](https://img.shields.io/badge/Raspberry%20Pi-C51A4A?style=for-the-badge&logo=raspberrypi&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

---

## システム構成

```
[Raspberry Pi]
  ├── PiCamera2  →  カメラ映像取得（640×480）
  ├── TFLite Runtime  →  量子化モデルでエッジ推論
  └── スピーカー（はんだ付け）  →  沸騰検知時にアラーム音

      ↕ SSH + stdout ストリーミング

[Windows PC]
  └── PowerShell  →  ログ監視 ＋ 沸騰アラーム通知
```

---

## Demo


https://github.com/user-attachments/assets/1663a089-b956-4790-a7ff-2462e3372716




---

## 実行例

Raspberry Pi 上で推論スクリプトを起動し、**Windows PC からリアルタイムで検知結果を監視**します。
沸騰を検知すると Raspberry Pi 側のスピーカーが鳴り、PC 側にもアラーム表示・警告音が出ます。

```powershell
ssh omoto@raspberrypi.local "export PYTHONIOENCODING=utf-8; source /home/omoto/venv/bin/activate && python3 -u /home/omoto/edge-ai/boiling_detect.py" | ForEach-Object {
    $line = $_
    Write-Host $line

    if ($line -match "警告") {
        Write-Host "--- 沸騰アラーム（PC） ---" -ForegroundColor Red
        [console]::Beep(1500, 500)
        [console]::Beep(1500, 500)
    }
}
```

| 処理 | 担当 |
|------|------|
| カメラ映像の取得・推論 | Raspberry Pi |
| スピーカーアラーム | Raspberry Pi（はんだ付けで自作） |
| ログ監視・PC 側通知 | Windows PC（PowerShell） |

---

## 工夫した点

### 1. 連続フレーム判定によるノイズ除去（デバウンス）

単一フレームの判定ではなく、**5 フレーム連続で沸騰と判定した場合のみ**アラームを発報する仕組みを実装しました。
一時的な映り込みや気泡の揺らぎによる誤検知を防ぎ、実環境での安定動作を実現しています。

```python
if boil_prob >= BOIL_THRESHOLD:
    boil_count += 1
else:
    boil_count = max(0, boil_count - 1)  # 即リセットせず漸減
is_boiling = boil_count >= BOIL_STABLE_FRAMES  # 5フレーム連続で確定
```

### 2. カウント漸減方式による判定の安定化

非沸騰フレームが来た際に `boil_count` を即ゼロリセットするのではなく、`-1` ずつ漸減させています。
短い揺り戻しで判定が頻繁に切り替わる「チャタリング」を防ぎ、滑らかな状態遷移を実現しています。

### 3. 量子化対応の前後処理

`preprocess()` と `postprocess()` で、TFLite モデルの量子化パラメータ（`scale` / `zero_point`）を取得し、
float ↔ uint8 の変換を動的に行います。モデルが量子化されているかどうかにかかわらず同じコードで動作します。

### 4. GlobalAveragePooling2D によるパラメータ削減

全結合層の前に `Flatten` ではなく `GlobalAveragePooling2D` を採用しました。
各チャンネルを 1 スカラーへ圧縮するため、特徴マップサイズに依存しない大幅なパラメータ削減が可能です。
軽量モデルでありながら高い汎化性能を保てる理由の一つです。

### 5. FP16 量子化による軽量化（TFLite）

学習済みモデルを TFLite 形式へ変換し、`tf.float16` 量子化を適用しました。
モデルサイズを約 50% 削減しつつ、精度劣化をほとんど抑えられています。
Raspberry Pi 上では `tflite_runtime`（フル TensorFlow 不使用）のみをインポートすることで、さらなる RAM 節約も実現しています。

### 6. AUC ベースのモデル選択と早期終了

`ModelCheckpoint` と `EarlyStopping` の監視指標に Accuracy ではなく **AUC** を採用しました。
AUC は閾値に依存しない評価指標であり、クラスバランスが多少偏っていても信頼できるモデル選択が可能です。
その結果、最終的に **val_AUC = 0.9999** を達成しています。

### 7. 多角的な評価指標の同時監視

Loss・Accuracy に加えて **AUC・Precision・Recall** を同時に計測しました。
沸騰を見逃す（偽陰性）リスクと誤検知（偽陽性）のトレードオフを可視化し、
最終エポックで Precision = 1.00、Recall = 0.9944 を達成しています。

### 8. tf.data パイプラインの最適化

`AUTOTUNE` による並列データロード・`cache()` によるメモリキャッシュ・`prefetch()` による先読みを組み合わせ、
GPU/CPU の待機時間を最小化する効率的なデータパイプラインを構築しました。

### 9. ドメイン知識に基づくカスタム前処理

入力画像の左上 5px を意図的にクロップする独自の前処理を実装しました（学習・推論の両方で統一）。
カメラ映像に埋め込まれるタイムスタンプ等のノイズ領域を除去することで、
モデルが本質的な特徴のみを学習できるよう配慮しています。

### 10. コスト削減のためのハードウェア自作（はんだ付け）

スピーカーモジュールを既製品で購入せず、**スピーカーユニットを Raspberry Pi の GPIO に直接はんだ付け**することでコストを削減しました。
ハードウェア工作は初挑戦でしたが、回路図の調査から実装まで自力で行い、`speaker-test` コマンドによるアラーム発報まで動作を確認しました。

---

## 性能

| Metric    | Train  | Validation |
|-----------|--------|------------|
| AUC       | 0.9956 | 0.9999     |
| Accuracy  | 0.9850 | 0.9962     |
| Precision | 0.9884 | 1.0000     |
| Recall    | 0.9872 | 0.9944     |

---

## ディレクトリ構成

```
boil-checker/
├── boil_checker.ipynb    # 学習・評価ノートブック
├── boiling_detect.py     # Raspberry Pi 上で動作する推論スクリプト
├── dataset/
│   ├── boil/             # 沸騰画像
│   └── unboil/           # 非沸騰画像
├── output/
│   ├── boil_best.keras   # val_AUC 最良モデル
│   ├── boil_final.keras  # 最終エポックモデル
│   └── history.json      # 学習履歴
└── model_quant.tflite    # FP16 量子化モデル（Raspberry Pi へ転送）
```
