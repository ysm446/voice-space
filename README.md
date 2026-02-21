# voice-echo

[Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) を使ったテキスト読み上げ（TTS）Web UIアプリです。
Gradioベースのブラウザインターフェースから3つのモードで音声を生成できます。

## 機能

| モード | 説明 |
|---|---|
| **Custom Voice** | プリセット話者（9種）を選び、感情・スタイルを指示して読み上げ |
| **Voice Design** | 自然言語で声の特徴を記述してオリジナルの声を生成 |
| **Voice Clone** | 参照音声をアップロードしてその声でテキストを読み上げ |

### 対応言語

Chinese / English / Japanese / Korean / German / French / Russian / Portuguese / Spanish / Italian / Auto

### プリセット話者（Custom Voice）

| 話者 | 説明 |
|---|---|
| Vivian | 明るく少しエッジの効いた若い女性 |
| Serena | 温かく優しい若い女性 |
| Uncle_Fu | 落ち着いた低い声の男性 |
| Dylan | 若々しい北京男性 |
| Eric | 活発な成都男性 |
| Ryan | リズム感のある動的な男性（英語） |
| Aiden | 明るいアメリカ男性（英語） |
| Ono_Anna | 明るい日本語女性 |
| Sohee | 温かな韓国語女性 |

## セットアップ

### 必要環境

- Python 3.12
- NVIDIA GPU（CUDA対応）推奨
- conda

### インストール

```bash
# 1. conda環境のセットアップ（main環境にインストール）
conda activate main

# 2. PyTorch CUDA版をインストール（CUDAバージョンに合わせて変更）
pip install torch --index-url https://download.pytorch.org/whl/cu124

# 3. 依存パッケージをインストール
pip install -r requirements.txt

# 4. (オプション) FlashAttention 2（VRAM使用量削減）
pip install flash-attn --no-build-isolation
```

## 起動

### Windows（バッチファイル）

```
start.bat をダブルクリック
```

### コマンドライン

```bash
conda activate main
python app.py
```

起動後、ブラウザで http://localhost:7860 を開いてください。

> 各タブで初回「生成」クリック時にモデルが `models/` へダウンロードされます（1モデル約3〜4GB）。

## ファイル構成

```
voice-echo/
├── app.py            # メインアプリ（Gradio UI + モデル管理）
├── requirements.txt  # Python依存パッケージ
├── start.bat         # Windows起動スクリプト
├── models/           # モデルキャッシュ（自動生成・gitignore済み）
└── .gitignore
```

## 使用モデル

| モード | モデルID |
|---|---|
| Custom Voice | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` |
| Voice Design | `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` |
| Voice Clone | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` |
