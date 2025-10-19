# Steam_Games
Short repository for text analytics on Steam games.

## Repository structure
- `data/` – Project data folder (data tracked in Git)
  - `data/raw/` – Original immutable data dumps
  - `data/external/` – Third-party data
  - `data/interim/` – Intermediate/transformed data
  - `data/processed/` – Final datasets ready for analysis

See `data/README.md` for details. Data files are tracked in Git so collaborators receive them on pull.

## Adding data
- Place large/source files into `data/raw/` or `data/external/` as appropriate.
- Data files are committed to Git so collaborators receive them on pull.
- Suggested naming: `steam_games_YYYYMMDD.csv`.

## Python environment setup

Create a virtual environment (one-time):

```bash
python3 -m venv .venv
```

Activate the environment:

- macOS/Linux:

```bash
source .venv/bin/activate
```

- Windows (PowerShell):

```powershell
.venv\\Scripts\\Activate.ps1
```

Install requirements (when ready):

```bash
pip install -r requirements.txt
```

Optional: Jupyter kernel for this project:

```bash
pip install jupyter ipykernel
python -m ipykernel install --user --name steam-games --display-name "Python (steam-games)"
```

Deactivate when done:

```bash
deactivate
```