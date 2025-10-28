# Steam Reviews Summarizer - Usage Guide

## Quick Start (macOS M-series optimized)

### 1. One-time setup

```bash
# Navigate to project root
cd "/Users/radimsoukal/Library/Mobile Documents/com~apple~CloudDocs/VŠE/05. SEMESTR/Text Analytics/R/Steam_Games 2"

# Create virtual environment
python3 -m venv .venv

# Activate it
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install "transformers>=4.44" torch pandas tqdm
```

### 2. Run the script

```bash
# Process all games in your data
./summarize_games.py

# Or with specific range (recommended for overnight runs)
./summarize_games.py --start-id 10 --end-id 12000 --checkpoint-every 10

# Process just a small batch for testing
./summarize_games.py --start-id 7000 --end-id 7100 --checkpoint-every 5
```

### 3. Find your results

All outputs are saved to: `data/reviews_summary/`

- **Checkpoints**: `checkpoint_XXXXXX_to_YYYYYY.csv` (saved every N games)
- **Final output**: `review_summaries_COMPLETE_<start>_to_<end>.csv`

## Key Features

✅ **Mac M-series optimized** - Automatically uses MPS (Metal Performance Shaders) for faster processing  
✅ **Memory-safe** - Sequential processing with garbage collection after each batch  
✅ **Smaller, faster model** - Uses `distilbart-cnn-6-6` instead of `12-6` (better speed/memory ratio)  
✅ **Automatic checkpoints** - Saves progress regularly so you can resume if interrupted  
✅ **Content filtering** - Focuses on gameplay-relevant sentences before summarization  

## Options

```
--start-id INT        First app_id to process (default: all)
--end-id INT          Last app_id to process (default: all)
--checkpoint-every INT Number of games per checkpoint (default: 10)
```

## Troubleshooting

**If you get "No input files found":**
- Make sure your CSV files are in `data/raw/` with pattern `app_reviews_*.csv`
- Or set: `export STEAM_GAMES_ROOT="/path/to/your/project"`

**If it's still running out of memory:**
- Reduce `--checkpoint-every` to 5 or even 3
- Close other applications
- Process smaller ranges (e.g., 500-1000 games at a time)

**To resume after interruption:**
- Just run with `--start-id` set to the last completed checkpoint
- The script will pick up where it left off

## Performance

- **Speed**: ~30-60 seconds per game on M1/M2 Mac
- **Memory**: ~2-4 GB RAM (way better than the parallel version)
- **Overnight run**: Can safely process 500-1000 games

## Comparison: Notebook vs Script

| Feature | Jupyter Notebook | This Script |
|---------|------------------|-------------|
| Workers | 3-9 parallel | 1 sequential |
| Memory usage | High (accumulates) | Low (cleaned regularly) |
| Stability | Can crash overnight | Rock solid |
| Device | CPU only | MPS/CUDA/CPU auto-detect |
| Model | distilbart-12-6 | distilbart-6-6 (faster) |
| Best for | Interactive testing | Production runs |
