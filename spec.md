# Omi ライフログ自動文字起こし＆話者識別パイプライン — 仕様書

**作成日**: 2026-04-04
**ステータス**: 設計完了・実装準備段階

---

## 1. プロジェクト概要

### 1-1. 目的

Limitless Pendant（ウェアラブルデバイス）で録音した日常の会話を、Omiアプリ経由でクラウドに送信し、自動的に文字起こし・話者識別（自分 vs 他人）を行い、Obsidian Vaultに会話単位のMarkdownとして保存する。さらに日次で要約を自動生成し、朝のブリーフィングとして提供する。

### 1-2. 設計方針

- **コスト最小化**: 月額0〜106円を目標（Omi Unlimited $19/月 = 2,850円の1/30以下）
- **プライバシー重視**: 音声データは自前サーバーで処理。STT・話者識別はローカル完結
- **会話単位の記録**: 機械的な時間分割ではなく、VADベースで自然な会話の開始〜終了を検出
- **自動化**: 手動操作なしで録音→文字起こし→保存→同期→要約まで完結

### 1-3. 月額コスト

| 項目 | 月額 |
|---|---|
| Oracle Cloud VPS（Always Free） | 0円 |
| STT: qwen3-asr-1.7b（VPS上CPU推論） | 0円 |
| 話者識別: ECAPA-TDNN（VPS上CPU推論） | 0円 |
| VAD: Silero VAD（VPS上CPU） | 0円 |
| 日次要約: Gemini 3.1 Flash-Lite API | 0〜106円 |
| **合計** | **0〜106円/月** |

---

## 2. システムアーキテクチャ

### 2-1. 全体構成図

```
┌──────────────────────────────────────────────────────────┐
│ ウェアラブル層                                              │
│  [Limitless Pendant] --BLE--> [Omi App (スマホ)]          │
│                                    │                      │
│                          Audio Bytes Webhook (60秒毎)     │
│                          PCM16, 16kHz, mono               │
└────────────────────────────────────┼───────────────────────┘
                                     │ HTTPS POST
                                     ▼
┌──────────────────────────────────────────────────────────┐
│ Oracle Cloud VPS (ARM A1, 4 OCPU, 24GB RAM)             │
│                                                          │
│  [Hono HTTPサーバー] ← Webhook受信                       │
│       │                                                  │
│       ▼                                                  │
│  [Silero VAD] ← 音声有無判定（CPU, <1% 負荷）           │
│       │                                                  │
│       ├─ 音声あり → セッションバッファに追記              │
│       │             last_voice_at 更新                    │
│       │                                                  │
│       └─ 60秒沈黙 → 会話確定                             │
│              │                                           │
│              ▼                                           │
│  [qwen3-asr-1.7b] ← STT（CPU推論, ONNX/CTranslate2）   │
│       │                                                  │
│       ▼                                                  │
│  [ECAPA-TDNN] ← 話者識別（自分 vs 他人, CPU推論）        │
│       │                                                  │
│       ▼                                                  │
│  [Markdown生成] → Obsidian Vault に書き込み              │
│                                                          │
│  [Syncthing デーモン] ←→ 全端末同期                      │
│  [cron: Git push] → GitHub Private Repo バックアップ     │
│                                                          │
│  [cron: 毎朝8:00] → Gemini API で日次要約生成           │
└──────────────────────────────────────────────────────────┘
                     ↕ Syncthing
┌─────────────────────────────────────┐
│ クライアント端末                      │
│  デスクトップPC / ノートPC / スマホ   │
│  Obsidian で Vault 参照・編集        │
└─────────────────────────────────────┘
```

### 2-2. データフロー詳細

```
1. Limitless Pendant → Omi App（BLE接続）
2. Omi App → VPS（HTTPS POST, 60秒毎, PCM16バイナリ）
3. VPS: Silero VAD で音声区間検出
4. VPS: 60秒間音声なし → 会話確定
5. VPS: 確定した会話の音声を qwen3-asr-1.7b でSTT
6. VPS: 各セグメントを ECAPA-TDNN で話者識別
7. VPS: frontmatter付きMarkdown生成 → Vault書き込み
8. Syncthing: Vault変更を全端末に自動同期
9. cron: 1時間毎に Vault を git commit + push
10. cron: 毎朝8:00に Gemini API で日次要約生成
```

---

## 3. インフラ仕様

### 3-1. Oracle Cloud VPS

| 項目 | 値 |
|---|---|
| プラン | Always Free |
| リージョン | ap-osaka-1（大阪）**※確保済み** |
| シェイプ | VM.Standard.A1.Flex（ARM Ampere）|
| OCPU | 4 |
| メモリ | 24 GB |
| ストレージ | 100 GB（Boot Volume, Standard）|
| エグレス | 10 TB/月（無料枠）|
| OS | Ubuntu 24.04 LTS (ARM64) |
| グローバルIP | Reserved Public IP（1つ無料）|

### 3-2. 注意事項

- Home Region は変更不可（大阪で確保済み）
- アイドル回収ポリシー: 7日間のCPU使用率95パーセンタイルが20%未満だと回収される可能性あり。常時処理が走るこの用途なら問題ないが、フェイルセーフとしてcronジョブを定期実行しておく
- Security List（OCI側）と OS内のiptables の両方でポート開放が必要

### 3-3. フォールバック（Oracle利用不能時）

| 優先度 | 方式 | 初期費用 | 月額 | 備考 |
|---|---|---|---|---|
| 1 | Hetzner Cloud CAX31（ARM 4コア8GB） | 0円 | ~1,100円 | 即時代替。ドイツだがバッチ処理なのでレイテンシ問題なし。RAM 8GBのため軽量モデル（whisper-turbo）推奨 |
| 2 | Mini PC（N100, 16GB RAM） | ~3万円 | ~400円（電気代） | 長期運用確定時に投資。省電力15W。3.5年でHetzner費用を回収 |
| 3 | 既存デスクトップPC（i7-12700F + RTX 3080） | 0円 | ~970円（8-18時） | 夜間は翌朝バッチ処理。GPU使えるのでSTT最速 |

---

## 4. コンポーネント仕様

### 4-1. Webhook受信サーバー

| 項目 | 値 |
|---|---|
| フレームワーク | Hono (Node.js) |
| ポート | 3000（内部）→ Nginx リバースプロキシ → 443 |
| SSL | Let's Encrypt（certbot + Nginx）|
| エンドポイント | `POST /audio` |
| 認証 | API Key ヘッダー（Bearer token） |
| 受信データ | `application/octet-stream`（PCM16バイナリ）|
| クエリパラメータ | `uid`, `sample_rate` |

### 4-2. VAD（Voice Activity Detection）

| 項目 | 値 |
|---|---|
| モデル | Silero VAD v5 |
| ランタイム | ONNX Runtime（ARM対応）|
| 入力 | PCM16, 16kHz, mono |
| 出力 | 音声区間のタイムスタンプ（開始/終了）|
| 判定ロジック | 直近60秒間に音声未検出 → 会話終了 |
| CPU負荷 | <1%（1時間の音声を15秒で処理）|

#### 会話境界判定ロジック

```
状態管理（ユーザーごと）:
  - session_buffer: bytes    # 現在の会話の音声データ
  - last_voice_at: float     # 最後に音声が検出されたUNIXタイムスタンプ
  - start_at: float          # 会話開始タイムスタンプ
  - is_active: bool          # セッションが存在するか

遷移ルール:
  チャンク到着 + 音声あり + セッションなし → 新セッション開始
  チャンク到着 + 音声あり + セッションあり → バッファ追記、last_voice_at更新
  チャンク到着 + 音声なし + セッションあり → バッファ追記のみ（沈黙カウント中）
  チャンク到着 + 音声なし + セッションなし → 無視（雑音のみ）
  タイマーチェック + (now - last_voice_at >= 60秒) → 会話確定、処理開始
```

### 4-3. STT（Speech-to-Text）

| 項目 | 値 |
|---|---|
| モデル | qwen3-asr-1.7b（第1候補）|
| フォールバック | whisper-large-v3-turbo（ARMで速度不足の場合）|
| ランタイム | Transformers pipeline or CTranslate2 |
| 入力 | WAV（PCM16から変換）|
| チャンク処理 | chunk_length_s=30, stride_length_s=(4, 4) |
| 言語 | 日本語（`language="ja"`）|
| VRAMメモリ | ~4GB（モデル常駐）|
| 想定速度（ARM CPU） | 実時間の1〜2倍（60秒音声→60〜120秒処理）|

#### ハルシネーション・ループ対策

1. **チャンク分割処理**: 長い会話音声も30秒チャンク + 4秒オーバーラップで処理（2時間一括投入はしない）
2. **VAD前処理**: Silero VADで無音・非音声区間を除去してからSTTに投げる
3. **ループ検出後処理**: 同一テキストが3回以上連続した場合に除去するフィルタ

### 4-4. 話者識別

| 項目 | 値 |
|---|---|
| モデル | speechbrain/spkrec-ecapa-voxceleb (ECAPA-TDNN) |
| ランタイム | ONNX Runtime（ARM対応）推奨。PyTorchでも可 |
| 方式 | 登録済みembeddingリストとのcosine similarity比較（複数話者対応） |
| 閾値 | 2段階（既知話者: 0.7 / 既存unknown再識別: 0.4）|
| ラベル | `me`（自分）/ 登録名 / `spk_XXXX`（未登録話者）|
| 処理単位 | STTセグメントごと（0.5秒未満のセグメントはスキップ）|

#### 初期セットアップ（自分の声登録）

1. 静かな環境で30秒〜1分間の音声を3つ録音
2. 各サンプルからembedding抽出
3. 3つの平均embeddingを `speakers/me.pt` として保存
4. テスト音声で閾値チューニング（自分の声と他人の声を判定させて調整）

#### 話者レジストリシステム

未知の話者が出現した際にembeddingを自動保存し、あとからObsidian上で名前を付けるだけで次回以降自動認識される仕組み。

##### ファイル構成

```
ObsidianVault/
└── transcripts/
    └── _speakers/
        ├── registry.md           # 話者名マッピング（Obsidianで手動編集）
        ├── me.pt                 # 自分のembedding
        ├── spk_a3f8.pt           # 未登録話者（自動生成）
        └── spk_b2c1.pt           # 別の未登録話者（自動生成）
```

##### registry.md（Obsidian上で編集）

```markdown
# 話者レジストリ

| ID | 名前 | 初出 |
|---|---|---|
| me | 自分 | - |
| spk_a3f8 | （未登録） | 2026-04-04 09:13 |
| spk_b2c1 | （未登録） | 2026-04-04 14:30 |
```

名前を入れるだけで次回から自動適用される：

```markdown
| ID | 名前 | 初出 |
|---|---|---|
| me | 自分 | - |
| spk_a3f8 | 田中 | 2026-04-04 09:13 |
| spk_b2c1 | （未登録） | 2026-04-04 14:30 |
```

##### 識別ロジック（2段階閾値）

```
セグメント音声 → embedding抽出
    ↓
全登録embeddingとcosine similarity比較
    ↓
similarity > 0.7 (KNOWN_THRESHOLD)
    → 既知の話者として確定（"me", "田中" 等）
    ↓
similarity 0.4〜0.7 (UNKNOWN_THRESHOLD)
    → 既存のunknown話者と同一人物（同じspk_IDを再利用）
    ↓
similarity < 0.4
    → 完全に新しい話者
    → 新しいembeddingを spk_{random_hex}.pt として自動保存
    → registry.md に行を自動追加
```

##### 運用フロー

1. **新しい話者が出現**（自動）: パイプラインがembeddingを自動保存し、registry.mdに「（未登録）」として行を追加
2. **名前の登録**（手動）: Obsidianでregistry.mdを開き、該当行の名前を書き換えるだけ。初出の日時とtranscriptを見れば誰か特定できる
3. **自動反映**: 次回の処理時にパイプラインがregistryを読み込み、以降その話者は登録名で表示される
4. **過去の修正**（任意）: 過去のtranscript内の `spk_XXXX` を一括置換するスクリプトを用意（手動実行）

##### 疑似コード

```python
def identify_speaker(segment_audio, registered_embeddings, registry):
    seg_embedding = extract_embedding(segment_audio)

    best_match = None
    best_similarity = 0

    for speaker_id, speaker_embedding in registered_embeddings.items():
        similarity = cosine_similarity(seg_embedding, speaker_embedding)
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = speaker_id

    if best_similarity > KNOWN_THRESHOLD:       # 0.7
        label = registry.get(best_match, best_match)
        return best_match, label

    elif best_similarity > UNKNOWN_THRESHOLD:    # 0.4
        label = registry.get(best_match, best_match)
        return best_match, label

    else:
        # 完全に新しい話者
        new_id = f"spk_{random_hex(4)}"
        save_embedding(new_id, seg_embedding)
        add_to_registry(new_id, first_seen=now())
        return new_id, new_id
```

### 4-5. Markdown生成

#### 出力形式

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

未登録話者がいる場合：

```markdown
---
speakers: [自分, spk_a3f8]
tags: [omi, auto-transcript, unknown-speaker]
---

# 会話 14:30〜14:35

🙋 自分 (0.0s): こんにちは
❓ spk_a3f8 (1.2s): はじめまして
```

`unknown-speaker` タグで未登録話者を含むtranscriptをObsidian上で一覧検索できる。

#### ディレクトリ構造

```
ObsidianVault/
├── transcripts/
│   ├── _speakers/
│   │   ├── registry.md          # 話者レジストリ（手動編集）
│   │   ├── me.pt                # 自分のembedding
│   │   ├── spk_a3f8.pt          # 未登録話者embedding
│   │   └── ...
│   ├── 2026-04-04/
│   │   ├── conversation_091300.md
│   │   ├── conversation_101530.md
│   │   └── conversation_143200.md
│   └── 2026-04-05/
│       └── ...
├── daily-briefing/
│   ├── briefing_2026-04-04.md
│   └── briefing_2026-04-05.md
└── ...（既存のVaultコンテンツ）
```

### 4-6. 日次要約

| 項目 | 値 |
|---|---|
| モデル | Gemini 3.1 Flash-Lite |
| API | Google AI Studio（有料API。学習に使われない）|
| 実行タイミング | 毎朝8:00（cron）|
| 入力 | 前日の全transcriptファイルを結合（推定6〜8万トークン）|
| 出力 | briefing_YYYY-MM-DD.md |
| 月額コスト | ~106円（無料枠で収まる可能性あり）|

#### 要約テンプレート

```markdown
---
date: 2026-04-04
type: daily-briefing
tags: [omi, briefing, auto-generated]
---

## サマリー
（全体の要約 3-5文）

## 重要なポイント
- （主要な話題・決定事項）

## アクションアイテム
- （やるべきこと）

## メモ
- （気になるキーワードや後で調べたいこと）
```

---

## 5. Syncthing + Git バックアップ

### 5-1. Syncthing

| 項目 | 値 |
|---|---|
| 役割 | VPS上のVaultを全端末（デスクトップ、ノート、スマホ）にリアルタイム同期 |
| 同期ディレクトリ | `~/ObsidianVault` |
| デーモン化 | systemd service |
| 既存構成との統合 | 既存のSyncthing母艦（GCE e2-micro）からVPSに移行 or 併存 |

### 5-2. Git バックアップ

| 項目 | 値 |
|---|---|
| リポジトリ | GitHub Private Repository |
| 実行間隔 | 1時間毎（cron）|
| コミットメッセージ | `auto-sync: YYYY-MM-DD HH:MM` |
| 変更なし時 | スキップ（`git diff --cached --quiet`） |

#### スクリプト

```bash
#!/bin/bash
cd ~/ObsidianVault
git add -A
git diff --cached --quiet && exit 0
git commit -m "auto-sync: $(date '+%Y-%m-%d %H:%M')"
git push origin main
```

---

## 6. Omi アプリ設定

| 設定項目 | 値 |
|---|---|
| Developer Mode | 有効化 |
| Realtime audio bytes URL | `https://<VPSドメイン>/audio` |
| 送信間隔 | 60秒 |
| カスタムSTT | 不使用（OmiのSTTは使わない。音声バイト直接処理）|

---

## 7. セキュリティ

| 対策 | 実装方法 |
|---|---|
| Webhook認証 | Bearer Token（API Key）ヘッダー |
| 通信暗号化 | HTTPS（Let's Encrypt）|
| SSH | 公開鍵認証のみ（パスワード無効化）|
| ファイアウォール | OCI Security List + iptables で443/22のみ開放 |
| Git認証 | SSH Key（deploy key）|
| 音声データ保存 | 処理完了後にバッファから削除 |
| Vault同期 | Syncthing（E2E暗号化対応）|

---

## 8. 実装フェーズ

### Phase 0: ローカルテスト（デスクトップPC）

**目的**: VPSデプロイ前に各コンポーネントの品質・精度を検証

- [ ] Python環境構築（venv + PyTorch CUDA）
- [ ] qwen3-asr-1.7b ダウンロード・日本語STTテスト
- [ ] Silero VAD テスト（自分の声、環境音、雑音で検証）
- [ ] ECAPA-TDNN テスト（自分の声サンプル3つ録音、embedding生成）
- [ ] 話者識別の2段階閾値チューニング（既知: 0.7、unknown再識別: 0.4 を基準に調整）
- [ ] 複数話者テスト（自分+他人2名の音声で、spk_ID自動生成・再識別を検証）
- [ ] registry.md の読み書きロジックテスト
- [ ] パイプライン結合テスト（音声→VAD→STT→話者識別→Markdown）
- [ ] ハルシネーション・ループ対策の検証

### Phase 1: Oracle Cloud VPS 構築

- [x] Oracle Cloud アカウント作成（大阪リージョン ap-osaka-1）**✅ 確保済み**
- [ ] ARM A1 インスタンス作成（4 OCPU / 24GB RAM）
- [ ] SSH接続確認、初期セットアップ（apt update等）
- [ ] ドメイン設定 + Let's Encrypt SSL + Nginx
- [ ] iptables + Security List 設定

### Phase 2: パイプラインデプロイ（VPS）

- [ ] Node.js + Hono Webhookサーバーデプロイ
- [ ] Python環境構築（ARM64版PyTorch or ONNX Runtime）
- [ ] qwen3-asr-1.7b ARM動作確認・速度計測
  - 60秒音声の処理時間を計測
  - 遅すぎる場合は whisper-large-v3-turbo にフォールバック
- [ ] Silero VAD デプロイ
- [ ] ECAPA-TDNN デプロイ + 自分のembedding配置
- [ ] セッション管理ロジック実装（会話境界判定）
- [ ] Markdown生成 + Vault書き込みロジック
- [ ] systemd サービス登録
- [ ] ディスク容量管理（処理済み音声の自動削除）

### Phase 3: Syncthing + Git

- [ ] Syncthing インストール + Vault同期設定
- [ ] 既存端末との接続確認
- [ ] Git リポジトリ初期化 + GitHub Private Repo 作成
- [ ] SSH deploy key 設定
- [ ] git-sync cron 設定（1時間毎）

### Phase 4: 日次要約

- [ ] Google AI Studio APIキー取得
- [ ] daily_briefing.py 実装
- [ ] cron 設定（毎朝8:00）
- [ ] 要約品質の確認・プロンプトチューニング

### Phase 5: Omi 連携 + 結合テスト

- [ ] Omi App Developer Mode 有効化
- [ ] Audio Bytes Webhook URL 設定（60秒間隔）
- [ ] E2Eテスト: 実際に話す → VPSで処理 → Vault反映確認
- [ ] 会話境界判定の確認（沈黙60秒で区切られるか）
- [ ] 話者識別の確認（自分/相手が正しく判定されるか）
- [ ] Syncthing同期の確認（他端末に反映されるか）
- [ ] 日次要約の確認（翌朝ブリーフィングが生成されるか）
- [ ] 1日通しでの運用テスト

---

## 9. リスクと対策

| リスク | 影響 | 対策 |
|---|---|---|
| qwen3-asrがARM CPUで遅すぎる | チャンクが溜まる | whisper-large-v3-turboにフォールバック |
| Oracle アカウントBAN・サービス停止 | 全処理が止まる | §3-3のフォールバック先に移行（Hetzner→Mini PC→既存PC）。Vault自体はSyncthing済み端末に残っているのでデータロストなし |
| Oracle のアイドル回収 | インスタンス停止 | 常時処理が走るため低リスク。念のためcronジョブ |
| VPSダウン時の音声ロスト | 会話が記録されない | Omiアプリ側にも最低限の記録が残る（無料枠STT） |
| STTハルシネーション・ループ | 同じ文が繰り返し出力 | チャンク分割 + VAD前処理 + ループ検出後処理 |
| 話者誤判定 | 自分と他人を取り違える | 複数サンプルの平均embedding + 閾値チューニング |
| ディスク容量不足 | 処理停止 | 処理済み音声の自動削除（cron） |
| Gemini API無料枠超過 | 課金発生 | 月額最大106円。許容範囲 |

---

## 10. 将来の拡張

| 拡張案 | 概要 |
|---|---|
| 感情分析 | 音声特徴量から会話のトーン（ポジティブ/ネガティブ）を推定 |
| 要約のLLM切り替え | ローカルLLM or Claude API への変更 |
| 会議モード | pyannote等によるクラスタリング型diarization（事前登録なしで話者分離）|
| 過去transcript一括更新 | registry.mdの名前変更時に過去のspk_IDを一括置換するスクリプト |
| 話者統合 | 同一人物に複数のspk_IDが割り当てられた場合にembeddingをマージするツール |