# Data Directory

This folder is where you and your colleagues should place datasets. The repository is configured to TRACK data files in Git, so that collaborators receive the data when pulling the repo.

Structure:
- raw/: Original, immutable data dumps.
- external/: Data from third-party sources.
- interim/: Intermediate data that has been transformed.
- processed/: Final datasets ready for modeling/analysis.

Guidelines:
- Data files ARE committed to Git. Ensure you have the right to share them and avoid sensitive content.
- Keep file names descriptive and consistent, e.g., `steam_games_YYYYMMDD.csv`.
- Prefer CSV/Parquet or RDS for tabular data.
- Document any schema or assumptions in this file.

Notes:
- Very large files (>100 MB) cannot be pushed to GitHub without Git LFS. Consider splitting or using LFS if needed.
