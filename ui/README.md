# EmotionEngine Steering UI

シンプルな text-to-text steering UI。バックエンドは FastAPI (Python) で
既存の `src.steering.generate.steered_generate` を再利用、
フロントエンドは Vite + TypeScript の素の構成。

## 構成

```
ui/
├── backend/        # FastAPI app — モデルと CAA ベクトルをロード、生成 API
│   ├── server.py
│   └── requirements.txt
└── frontend/       # Vite + TypeScript SPA
    ├── index.html
    ├── package.json
    └── src/{main.ts,style.css}
```

UI は感情カテゴリ（anger, joy, …）ごとに ±3 の std-norm スライダを並べ、
各 α を `α_unit · scale · ||v_cat||` で raw alpha に変換、
それを per-category ベクトルに乗じて足し合わせ、`steered_generate` に
1 回だけ渡す（=単一の合成 steering vector）。

## 実行

### 1) バックエンド

```bash
source .venv/bin/activate
pip install -r ui/backend/requirements.txt   # 初回のみ
uvicorn ui.backend.server:app --host 127.0.0.1 --port 8000
```

起動時に `configs/model.yaml` の active プロファイル
（既定: `llama` = Llama-3.1-8B-Instruct）と
`data/emotion_code/caa.pt` をロードする。
GPU が無い環境では `configs/model.yaml` の `active: gpt2` に切り替えること。

### 2) フロントエンド

別ターミナルで:

```bash
cd ui/frontend
npm install
npm run dev
```

ブラウザで http://localhost:5173 を開く。
`/api/*` は Vite の dev proxy 経由で `127.0.0.1:8000` に転送される。

## API

- `GET /api/meta` — `{profile, model_name, categories, layer, scale, norms}`
- `POST /api/generate` —
  ```json
  {
    "prompt": "...",
    "weights": {"joy": 1.5, "anger": -1.0},
    "max_new_tokens": 80,
    "temperature": 0.0,
    "top_p": 1.0,
    "repetition_penalty": 1.1,
    "no_repeat_ngram_size": 3
  }
  ```
  → `{generation, effective_alphas, layer}`

## 注意

- 単一の合成ベクトル前提で、inject layer は `configs/steering.yaml` の
  `caa.inject_layers[0]`（既定: 16）のみ。
- 8B モデルは GPU 24 GB 想定。生成は再フォワード方式（`generate.py` の
  コメント参照）なので長文は遅い。`max_new_tokens` を控えめに。
