# 01 — LM Studio setup

Source of truth: `scripts/check_lmstudio.py`.

LM Studio 0.4.19 is already installed at
`C:\Users\Admin\AppData\Local\Programs\LM Studio\LM Studio.exe`.

There is no chat model on disk yet — only an 84 MB embedding model, which
cannot answer prompts. Download one first.

GUI route:

1. Open LM Studio
2. **My Models** -> search `Qwen3 4B Instruct` -> Download (Q4_K_M, ~2.5 GB, fits 4 GB VRAM)
3. Load the model
4. **Developer** tab -> **Start Server** (default port 1234)

CLI route:

```powershell
lms pull qwen3-4b-instruct-q4_k_m
lms load qwen3-4b-instruct --context-length 8192
lms server start
lms ps            # confirm what is loaded
```

Verify:

```powershell
python scripts\check_lmstudio.py      # exit 0 = ready, 1 = not ready
```

The doctor distinguishes chat vs embedding models by id hints
(`embed`, `nomic`, `bge`, `gte`, `minilm`, `e5`). Only embedding models
loaded -> FAIL: an embedding model cannot answer chat prompts.
