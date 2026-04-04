# Transcripts Pipeline

Omi (wearable device) から送られる音声を自動で文字起こし・話者識別し、Obsidian Vault に会話単位の Markdown として保存するパイプライン。

## Features

- **VAD (Voice Activity Detection)** - Silero VAD v5 で音声区間を検出
- **STT (Speech-to-Text)** - Qwen3-ASR-1.7B / Whisper で日本語文字起こし
- **Speaker Identification** - ECAPA-TDNN で話者識別 (自分 vs 他者)
- **Auto Speaker Registry** - 未知の話者を自動登録、後から名前をつけるだけで認識
- **Obsidian Integration** - frontmatter 付き Markdown を日付別ディレクトリに出力
- **Webhook Server** - Hono (Node.js) で Omi App からの音声を受信
- **Session Management** - 60 秒の沈黙で会話を自動区切り

## Architecture

```
Omi App (60s audio chunks)
    │ HTTPS POST /audio
    ▼
Hono Webhook Server (Node.js, port 3000)
    │ Bearer auth, session buffer management
    │ 60s silence → finalize
    ▼
Python ML Pipeline (FastAPI, port 8000)
    ├── Silero VAD → speech detection
    ├── Qwen3-ASR → Japanese STT
    ├── ECAPA-TDNN → speaker identification
    └── Markdown Writer → Obsidian Vault
```

## Requirements

- Python 3.12+
- Node.js 20+
- NVIDIA GPU (CUDA, for local/dev) or ARM CPU (for VPS deployment)
- ~6GB disk for ML models

## Setup

### 1. Clone & install Python dependencies

```bash
git clone https://github.com/yourname/transcripts-pipeline.git
cd transcripts-pipeline

python3 -m venv .venv
source .venv/bin/activate

# GPU (local development)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install qwen-asr speechbrain numpy soundfile pytest

# CPU only (VPS deployment)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install qwen-asr speechbrain numpy soundfile
```

### 2. Install Node.js dependencies

```bash
cd server
npm install
cd ..
```

### 3. Configure

`config.toml` を編集:

```toml
[stt]
model_name = "Qwen/Qwen3-ASR-1.7B"  # or "openai/whisper-large-v3-turbo"
device = "cuda"                       # "cuda" for GPU, "cpu" for VPS

[speaker_id]
known_threshold = 0.55    # cosine similarity for known speaker match
unknown_threshold = 0.25  # threshold for re-identifying unknown speakers

[paths]
speakers_dir = "output/transcripts/_speakers"
output_dir = "output/transcripts"
```

## Speaker Registration

話者識別には事前の声の登録が必要です。

### Recording tips

| ポイント | 詳細 |
|---|---|
| 環境 | 静かな場所。エアコン・換気扇の音も避ける |
| 長さ | 1サンプルあたり **30〜60秒** |
| サンプル数 | **3つ以上** が理想 |
| トーン | 異なる話し方で録る (普通の会話、説明調、笑いながら等) |
| マイク距離 | 20〜30cm を一定に保つ |
| NG | BGM入り、複数人の声、5秒未満、ほぼ無音 |

### Talk scripts for recording

何を話せばいいか迷ったら、以下のスクリプトを読み上げてください。
**各サンプルで異なるスクリプトを使う**と、声の特徴を幅広く学習できます。

<details>
<summary>サンプル1: 普通の会話調 (~40秒)</summary>

> 昨日スーパーに行ったんだけど、新しくできたパン屋がすごく混んでてさ。
> 30分くらい並んで、やっとクロワッサンを買えたんだよね。
> 焼きたてでバターの香りがすごくて、帰りの車の中で我慢できなくて一つ食べちゃった。
> サクサクでめちゃくちゃ美味しかった。今度一緒に行かない？
> 土曜の朝イチが狙い目らしいよ。

</details>

<details>
<summary>サンプル2: 説明・解説調 (~40秒)</summary>

> 話者識別の仕組みを簡単に説明すると、まず音声からスペクトログラムっていう
> 周波数の特徴を抽出します。それをニューラルネットワークに通すと、
> 話者ごとに固有のベクトル、つまりembeddingが得られるんですね。
> このembedding同士のコサイン類似度を計算して、閾値以上なら同一人物と判定します。
> 声の高さだけじゃなくて、話し方の癖とか、母音の出し方の特徴も反映されるので、
> 意外と精度高く判別できるんですよ。

</details>

<details>
<summary>サンプル3: 感情を込めて・リラックス調 (~40秒)</summary>

> いやー、今日はマジで疲れた。朝から会議が3つもあってさ、
> しかも全部オンラインだから、ずっとパソコンの前。目が痛いわ肩凝るわで。
> でもまあ、プロジェクトの承認がやっと降りたのは嬉しかったな。
> 半年くらいかかったもんね。チームのみんなも喜んでたし。
> 今日は帰ったらビールでも飲んで、録画してたドラマ見て寝よう。
> あ、そういえば明日は休みだった。最高じゃん。

</details>

<details>
<summary>English sample (~40 sec, for bilingual speakers)</summary>

> So I've been thinking about this new project idea. Basically, it's a tool that
> automatically transcribes conversations and identifies who's speaking.
> The cool thing is it runs entirely on your own server, so no data leaves your network.
> I'm using a combination of voice activity detection, speech-to-text, and speaker embeddings.
> It's still a work in progress, but the initial results look pretty promising.
> I'll probably open-source it once the core features are stable.

</details>

### Register yourself

録音した音声ファイルを `samples/` に配置してから登録します:

```bash
# 録音ファイルを samples/ に配置
cp ~/recordings/me_*.wav samples/

# 登録
source .venv/bin/activate
python scripts/register_speaker.py --name me samples/me_01.wav samples/me_02.wav samples/me_03.wav
```

スクリプトが以下を自動で行います:
1. 各音声ファイルの品質チェック (長さ、音声比率、音量)
2. Embedding の一貫性チェック (サンプル間の類似度)
3. 平均 embedding を `me.pt` として保存
4. `registry.md` を更新

### Register other speakers

```bash
# 友人の音声を samples/ に配置して登録
python scripts/register_speaker.py --name "田中" samples/tanaka_01.wav samples/tanaka_02.wav

# 別の友人
python scripts/register_speaker.py --name "鈴木" samples/suzuki_01.wav samples/suzuki_02.wav samples/suzuki_03.wav
```

### Register via microphone (requires sounddevice)

```bash
pip install sounddevice
python scripts/register_speaker.py --name me --record 3
```

### Speaker registry

登録された話者は `output/transcripts/_speakers/registry.md` で管理されます:

```markdown
| ID | 名前 | 初出 |
|---|---|---|
| me | 自分 | - |
| tanaka | 田中 | 2026-04-04 09:13 |
| spk_a3f8 | （未登録） | 2026-04-04 14:30 |
```

パイプライン実行中に未知の話者が出現すると `spk_XXXX` として自動登録されます。
Obsidian で `registry.md` を開いて名前を書き換えるだけで、次回から自動認識されます。

## Usage

### Process a single audio file

```bash
source .venv/bin/activate
python scripts/run_pipeline.py recording.wav
python scripts/run_pipeline.py recording.wav --timestamp "2026-04-04 09:13"
```

### Run the servers (webhook mode)

**Terminal 1: Python ML API**
```bash
source .venv/bin/activate
uvicorn src.server:app --host 127.0.0.1 --port 8000
```

**Terminal 2: Node.js Webhook**
```bash
cd server
API_KEY=your-secret-key node index.js
```

### Test the webhook

```bash
source .venv/bin/activate
python scripts/test_webhook.py sample.wav --api-key your-secret-key
```

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `PORT` | `3000` | Webhook server port |
| `API_KEY` | `dev-secret-key` | Bearer token for authentication |
| `SILENCE_TIMEOUT_MS` | `60000` | Silence duration to finalize session |
| `PYTHON_HTTP` | `http://localhost:8000` | Python API URL |
| `CHECK_INTERVAL_MS` | `10000` | Silence check interval |

## Output format

```markdown
---
date: 2026-04-04
start: "09:13"
end: "09:16"
duration_sec: 180
type: conversation
speakers: [自分, 田中]
tags: [omi, auto-transcript]
---

# 会話 09:13〜09:16

🙋 自分 (0.0s): さっきのプルリクの件なんだけど
👤 田中 (2.1s): ああ、あれね。レビュー見たよ
🙋 自分 (4.5s): マージしていい？
👤 田中 (5.8s): いいよ、LGTM
```

## Project structure

```
transcripts-pipeline/
├── config.toml                  # All settings (thresholds, model names, paths)
├── pyproject.toml               # Python dependencies
├── samples/                     # Place speaker audio samples here
├── src/
│   ├── config.py                # Configuration loader
│   ├── vad.py                   # Silero VAD wrapper
│   ├── stt.py                   # Qwen3-ASR / Whisper wrapper
│   ├── speaker_id.py            # ECAPA-TDNN speaker identification
│   ├── speaker_registry.py      # Speaker registry management
│   ├── hallucination_filter.py  # Repeated text removal
│   ├── markdown_writer.py       # Obsidian Markdown generation
│   ├── pipeline.py              # Pipeline orchestrator
│   └── server.py                # FastAPI backend
├── server/
│   ├── package.json
│   └── index.js                 # Hono webhook server + session management
└── scripts/
    ├── register_speaker.py      # Speaker registration with quality checks
    ├── run_pipeline.py          # CLI entry point
    └── test_webhook.py          # Webhook integration test
```

## Deployment (Oracle Cloud VPS)

See `spec.md` for full infrastructure specifications.

### Quick deploy via GitHub Actions

1. Set up SSH deploy key on VPS
2. Add secrets to GitHub: `VPS_HOST`, `SSH_PRIVATE_KEY`
3. Push to main triggers: `git pull` + `systemctl restart`

```yaml
# .github/workflows/deploy.yml
on:
  push:
    branches: [main]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: appleboy/ssh-action@master
        with:
          host: ${{ secrets.VPS_HOST }}
          username: deploy
          key: ${{ secrets.SSH_PRIVATE_KEY }}
          script: |
            cd /opt/transcripts-pipeline
            git pull origin main
            source .venv/bin/activate
            pip install -e .
            sudo systemctl restart transcripts-pipeline
```

## License

MIT
