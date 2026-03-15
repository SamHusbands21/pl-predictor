# PL Betting Model - Full Retrain Script
# Run this periodically (e.g. after each matchweek) to refresh xG data,
# rebuild features, and retrain XGBoost + Random Forest models.
# Commits updated models and understat data to GitHub.

$ErrorActionPreference = "Stop"

$PYTHON   = "C:\Users\Sam\AppData\Local\Programs\Python\Python39\python.exe"
$PL_DIR   = "C:\Users\Sam\Documents\pl-predictor"
$LOG_DIR  = Join-Path $PL_DIR "logs"

New-Item -ItemType Directory -Force -Path $LOG_DIR | Out-Null
$LOG_FILE = Join-Path $LOG_DIR ("retrain_" + (Get-Date -Format "yyyy-MM-dd") + ".log")

function Log($msg) {
    $line = (Get-Date -Format "HH:mm:ss") + "  " + $msg
    Write-Host $line
    Add-Content $LOG_FILE $line
}

Set-Location $PL_DIR
Log "=== Full retrain started ==="

# --- Step 1: Download latest football-data results ---
Log "Step 1/4  Downloading latest match data..."
$ErrorActionPreference = "Continue"
& $PYTHON -m src.collect.football_data
$ec = $LASTEXITCODE
$ErrorActionPreference = "Stop"
if ($ec -ne 0) { Log "ERROR: football_data failed"; exit 1 }
Log "Football data done."

# --- Step 2: Refresh understat xG (all seasons) ---
Log "Step 2/4  Refreshing understat xG data..."
$ErrorActionPreference = "Continue"
& $PYTHON -m src.collect.understat
$ec = $LASTEXITCODE
$ErrorActionPreference = "Stop"
if ($ec -ne 0) { Log "WARNING: understat fetch had issues (continuing with cached data)" }
Log "Understat done."

# --- Step 3: Rebuild features ---
Log "Step 3/4  Building feature matrix..."
$ErrorActionPreference = "Continue"
& $PYTHON -m src.features.engineer
$ec = $LASTEXITCODE
$ErrorActionPreference = "Stop"
if ($ec -ne 0) { Log "ERROR: engineer failed"; exit 1 }
Log "Features done."

# --- Step 4: Retrain models ---
Log "Step 4/4  Training XGBoost + Random Forest..."
$ErrorActionPreference = "Continue"
& $PYTHON -m src.models.train
$ec = $LASTEXITCODE
$ErrorActionPreference = "Stop"
if ($ec -ne 0) { Log "ERROR: train failed"; exit 1 }
Log "Training done."

# --- Commit and push updated models + understat data ---
Log "Committing updated models and data to GitHub..."
$date = Get-Date -Format "yyyy-MM-dd"
git add "models\" "data\raw\understat\"
$diff = git diff --cached --name-only
if ($diff) {
    git commit -m "chore: retrain models and refresh xG data [$date]"
    git push
    Log "Pushed to GitHub."
} else {
    Log "No changes to commit."
}

Log "=== Retrain complete ==="
