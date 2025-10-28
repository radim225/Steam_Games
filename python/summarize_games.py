#!/usr/bin/env python3
# summarize_games.py  —  Fast, lean Steam reviews summarizer for macOS (M-series ready)
# - Auto-detects project root (keeps outputs in data/reviews_summary)
# - Uses MPS on Apple Silicon if available, otherwise CPU/CUDA
# - Lightweight model: sshleifer/distilbart-cnn-6-6 (good speed/quality balance)
# - Sequential, memory-safe; periodic checkpoints

import os, re, glob, gc, time, argparse, string, warnings
from pathlib import Path
import pandas as pd

warnings.filterwarnings("ignore")

# -------------------------
# Path & root detection
# -------------------------
def detect_project_root() -> Path:
    env_root = os.environ.get("STEAM_GAMES_ROOT")
    if env_root:
        p = Path(env_root).expanduser().resolve()
        return p
    here = Path.cwd().resolve()
    for cand in (here, *here.parents):
        if cand.name == "python":
            continue
        if (cand / "data").exists():
            return cand
    return here.parent if here.name == "python" else here

PROJECT_ROOT = detect_project_root()
DATA_DIR      = (PROJECT_ROOT / "data").resolve()
RAW_DIR       = (DATA_DIR / "raw").resolve()
PROCESSED_DIR = (DATA_DIR / "processed").resolve()
OUT_DIR       = (DATA_DIR / "reviews_summary").resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

CACHE_FILE = PROCESSED_DIR / "combined_reviews_cache.pkl"

# -------------------------
# Model / device setup
# -------------------------
MODEL_ID = "sshleifer/distilbart-cnn-6-6"  # smaller & faster than 12-6

def detect_device():
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.float16, "CUDA"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps"), torch.float32, "MPS"
    return torch.device("cpu"), torch.float32, "CPU"

# -------------------------
# Text cleaning & combining
# -------------------------
def clean_text(text: str) -> str | None:
    if not isinstance(text, str) or not text.strip():
        return None
    # strip HTML + URLs + emojis + control chars
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"http[s]?://\S+|www\.\S+", "", text)
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"[\U00010000-\U0010ffff]", "", text)  # emojis & misc
    allowed = set(string.ascii_letters + string.digits + " .,!?'-:/()[]")
    text = "".join(ch for ch in text if ch in allowed)
    text = re.sub(r"\s+", " ", text).strip()
    return text if text else None

def combine_reviews(row: pd.Series) -> str:
    reviews = []
    for i in range(1, 101):
        rv = row.get(f"review_{i}")
        if rv and isinstance(rv, str):
            cleaned = clean_text(rv)
            if cleaned:
                reviews.append(cleaned)
    # dedupe in order
    unique = list(dict.fromkeys(reviews))
    return " [SEP] ".join(unique) if unique else ""

# -------------------------
# Light content filter (optional, with safe fallback)
# -------------------------
KEYWORDS_RE = re.compile(
    r"gameplay|mechanic|combat|gun|weapon|campaign|story|mode|co-?op|multiplayer|"
    r"graphics|performance|bug|optim|difficulty|progression|content|map|level|class|rank|matchmaking",
    re.I,
)

def keep_relevant_sentences(text: str) -> str:
    sents = re.split(r"(?<=[.!?])\s+", text)
    kept = [s for s in sents if KEYWORDS_RE.search(s)]
    filtered = " ".join(kept).strip()
    # fallback if too short
    return filtered if len(filtered) >= 300 else text

# -------------------------
# Chunking
# -------------------------
def split_into_chunks(text: str, max_chunk: int = 1500, overlap: int = 100) -> list[str]:
    if not text:
        return []
    if len(text) <= max_chunk:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        end = start + max_chunk
        if end < len(text):
            # try to cut at sentence boundary or [SEP]
            window = text[start:end]
            cut = max(window.rfind(". "), window.rfind("! "), window.rfind("? "), window.rfind(" [SEP] "))
            if cut != -1:
                end = start + cut + 2  # keep punctuation
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap if end < len(text) else end
    return chunks

# -------------------------
# Summarization helpers
# -------------------------
GUIDANCE_PROMPT = (
    "Summarize the following user reviews into a concise, neutral third-person description of the game. "
    "Focus on: genre, core gameplay loop and mechanics, key features/modes (campaign/multiplayer/co-op), "
    "difficulty/learning curve, performance/technical notes if widely mentioned. Avoid slang, memes, and repetition.\n\nReviews:\n"
)

def token_budget(n_chars: int, cap: int = 96) -> int:
    # rough heuristic: ~4 chars/token; enforce bounds
    return max(32, min(cap, n_chars // 4))

def build_summarizer():
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
    import torch
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # quiet transformers logs
    try:
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error()
    except Exception:
        pass

    device, dtype, dev_name = detect_device()
    print(f"→ Device: {dev_name}")

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID, torch_dtype=dtype)
    mdl.to(device)

    # pipeline uses model's device; keep device=-1 to avoid re-moving
    summarizer = pipeline("summarization", model=mdl, tokenizer=tok, device=-1)
    return summarizer, device

def generate_summary(summarizer, text: str, cap: int = 96) -> str:
    import torch
    if not text or not text.strip():
        return ""
    max_new = token_budget(len(text), cap=cap)
    with torch.inference_mode():
        out = summarizer(
            text,
            max_new_tokens=max_new,
            num_beams=1,
            do_sample=False,
            truncation=True,
            batch_size=1,
        )[0]["summary_text"]
    return out

def summarize_reviews(summarizer, text: str) -> str:
    if not text or not text.strip():
        return ""
    base = keep_relevant_sentences(text)

    # short path
    if len(base) <= 800:
        guided = f"{GUIDANCE_PROMPT}{base}"
        return generate_summary(summarizer, guided, cap=64)

    # chunked
    parts = []
    for ch in split_into_chunks(base, max_chunk=1500, overlap=100):
        guided = f"{GUIDANCE_PROMPT}{ch}"
        parts.append(generate_summary(summarizer, guided, cap=64))

    combined = " ".join(p for p in parts if p).strip()
    if len(combined) < 200:
        return combined

    final_guided = f"{GUIDANCE_PROMPT}{combined}"
    return generate_summary(summarizer, final_guided, cap=96)

# -------------------------
# Data loading (cache-aware)
# -------------------------
def load_or_build_dataframe() -> pd.DataFrame:
    if CACHE_FILE.exists():
        try:
            obj = pd.read_pickle(CACHE_FILE)
            if isinstance(obj, pd.DataFrame) and "combined_reviews" in obj.columns:
                print(f"✓ Loaded cache: {len(obj)} games")
                return obj
            else:
                print("Cache invalid -> rebuilding…")
        except Exception as e:
            print(f"Cache error: {e} -> rebuilding…")

    csv_files = sorted(glob.glob(str(RAW_DIR / "app_reviews_*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No input files found in {RAW_DIR} (expected app_reviews_*.csv)")
    dfs = [pd.read_csv(f) for f in csv_files]
    df = pd.concat(dfs, ignore_index=True)

    df["combined_reviews"] = df.apply(combine_reviews, axis=1)
    df.to_pickle(CACHE_FILE)
    print(f"✓ Built & cached: {len(df)} games")
    return df

# -------------------------
# Main processing
# -------------------------
def run(start_id: int | None, end_id: int | None, checkpoint_every: int):
    df = load_or_build_dataframe()

    # Filter & sort
    df_sorted = df.sort_values("app_id").reset_index(drop=True)
    if start_id is not None:
        df_sorted = df_sorted[df_sorted["app_id"] >= start_id]
    if end_id is not None:
        df_sorted = df_sorted[df_sorted["app_id"] <= end_id]
    df_sorted = df_sorted.reset_index(drop=True)

    n_rows = len(df_sorted)
    if n_rows == 0:
        print("Nothing to process for the given range.")
        return

    print(f"Prepared rows: {n_rows}")
    summarizer, _ = build_summarizer()
    print("✓ Summarizer ready")
    ok = fail = skip = 0

    # Ensure target col exists
    if "reviews_summary" not in df_sorted.columns:
        df_sorted["reviews_summary"] = ""

    batch_size = max(1, checkpoint_every)
    # Compute overall id range for final filename
    overall_start_id = int(df_sorted.iloc[0]["app_id"])
    overall_end_id   = int(df_sorted.iloc[-1]["app_id"])

    # Progress (simple)
    total_batches = (n_rows + batch_size - 1) // batch_size
    for b in range(total_batches):
        s = b * batch_size
        e = min(s + batch_size, n_rows)
        batch = df_sorted.iloc[s:e]
        batch_start_id = int(batch.iloc[0]["app_id"])
        batch_end_id   = int(batch.iloc[-1]["app_id"])
        print(f"\nBatch {b+1}/{total_batches}  (app_id {batch_start_id}–{batch_end_id})  size={len(batch)}")

        for i, row in batch.iterrows():
            text = row.get("combined_reviews", "")
            app_id = int(row["app_id"])
            if not text or not text.strip():
                skip += 1
                continue
            try:
                summary = summarize_reviews(summarizer, text)
                df_sorted.loc[i, "reviews_summary"] = summary
                ok += 1
            except Exception as ex:
                fail += 1
                print(f"  ⚠️  app_id {app_id}: {ex}")

        # checkpoint
        ck_path = OUT_DIR / f"checkpoint_{batch_start_id:06d}_to_{batch_end_id:06d}.csv"
        df_sorted.loc[s:e-1, ["app_id", "reviews_summary"]].to_csv(ck_path, index=False)
        print(f"  💾 Saved {ck_path.name} | ok={ok}, fail={fail}, skip={skip}")
        # memory hygiene
        gc.collect()
        time.sleep(0.2)

    final_name = f"review_summaries_COMPLETE_{overall_start_id:06d}_to_{overall_end_id:06d}.csv"
    final_path = OUT_DIR / final_name
    df_sorted[["app_id", "combined_reviews", "reviews_summary"]].to_csv(final_path, index=False)
    print(f"\n✓ Complete: ok={ok}, fail={fail}, skip={skip}")
    print(f"✓ Saved: {final_path}")

# -------------------------
# CLI
# -------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Summarize Steam user reviews into neutral game descriptions.")
    p.add_argument("--start-id", type=int, default=None, help="First app_id to include (inclusive).")
    p.add_argument("--end-id",   type=int, default=None, help="Last app_id to include (inclusive).")
    p.add_argument("--checkpoint-every", type=int, default=10, help="Checkpoint size (games).")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print("Project root:", PROJECT_ROOT)
    print("Data dir:    ", DATA_DIR)
    print("Output dir:  ", OUT_DIR)
    run(args.start_id, args.end_id, args.checkpoint_every)
