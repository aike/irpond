# irpond — Impulse Response メーカー

スイープ測定から Impulse Response(IR)wav ファイルを作成する GUI ツールです。

## 機能

- **測定モード**: スイープ原音 wav + ターゲット録音 wav から正則化スペクトル除算で IR を算出
- **推定モード**: スイープ原音がなくても、ターゲット録音だけから IR を作成
  - STFT リッジ追跡で指数(Log)/リニアスイープを自動判別しパラメータを推定
  - デコンボリューション結果の鋭さを最大化するレート精密化+位相補正で高精度化
- **試聴**: 任意の wav に IR を畳み込んで再生。再生中に Wet(0–100%)をリアルタイム調整可能
- **保存**: サンプルレート(44.1k〜192k)/量子化(16/24-bit PCM, 32-bit float)/チャンネル(モノラル・ステレオ)を選んで書き出し
- 入力はモノラル・ステレオ、任意のサンプルレート・ビット深度に対応(レート不一致は自動リサンプル)

## 動作環境

- **Python**: 3.10 以上(開発・動作確認は Python 3.11.4)
- **OS**:
  - **Windows 10 / 11** — 動作確認済み(Windows 11 Home 上でテスト)
  - **macOS / Linux** — 使用ライブラリ(tkinter / numpy / scipy / soundfile /
    sounddevice)はいずれもクロスプラットフォームのため動作する想定(未確認)
- **必要なもの**:
  - tkinter(Windows / macOS の公式 Python インストーラには同梱。
    Linux では `sudo apt install python3-tk` など別途インストール)
  - PortAudio(sounddevice の再生に使用。Windows / macOS は pip の wheel に
    同梱。Linux では `sudo apt install libportaudio2` など別途インストール)
  - 試聴機能を使う場合は音声出力デバイス

## 使い方

```
pip install -r requirements.txt
python irpond.py
```

1. 「ターゲット録音 wav」を選択(スイープをターゲット機器・空間に通して録音したもの)
2. スイープ原音があれば「スイープ原音 wav」にも指定(なければ空欄のまま=推定モード)
3. IR 長さ(自動 or 秒数)と正規化を選び「IR を作成」
4. 「試聴用音源 wav」を選んで ▶ 再生。Wet スライダーで原音とのバランスを調整
5. 保存フォーマットを選んで「IR を保存...」

## デモとテスト

```
python make_demo.py      # demo/ にスイープ・録音・試聴用 wav を生成
python selftest.py       # DSP コアの検証(既知 IR の復元精度など)
python gui_smoketest.py  # GUI をスクリプト駆動で通しテスト
```

## ファイル構成

- `irpond.py` — GUI 本体(tkinter)
- `ir_core.py` — DSP コア(読み込み/リサンプル/スイープ推定/デコンボリューション/保存)
- `selftest.py` / `gui_smoketest.py` — 自動テスト
- `make_demo.py` — デモ用 wav 生成

## 仕組みのメモ

- IR 算出は `H = R·conj(S) / (|S|² + ε·max|S|²)`(ε=1e-6)の正則化除算。
  循環バッファからピーク前 2ms を含めて切り出し、末尾はフェードアウト
- 推定モードのスイープレートは、リッジ回帰(初期値)→ crest factor 高密度スキャン
  (0.025% 刻み、必要なら範囲自動拡張)→ エンベロープベースのズーム→
  解析信号ピーク位相によるスイープ初期位相補正、の順で追い込みます
- IR 長さ「自動」はエンベロープがピーク −60 dB またはノイズフロア +10 dB を
  下回る点までを採用

## ライセンス

MIT

