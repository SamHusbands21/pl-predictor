# PL Betting Model - Full Retrain Script
# Run this periodically (e.g. after each matchweek) to refresh xG data,
# rebuild features, retrain XGBoost + Random Forest models, and evaluate.
# Commits updated models and understat data to the predictor repo, then
# pushes evaluation artefacts (calibration.png, shap_summary.png,
# evaluation.json) to the website repo.

$ErrorActionPreference = "Stop"

$PYTHON   = "C:\Users\Sam\AppData\Local\Programs\Python\Python39\python.exe"
$PL_DIR   = "C:\Users\Sam\Documents\pl-predictor"
$SITE_DIR = "C:\Users\Sam\Documents\sam_website"
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
Log "Step 1/6  Downloading latest match data..."
$ErrorActionPreference = "Continue"
& $PYTHON -m src.collect.football_data
$ec = $LASTEXITCODE
$ErrorActionPreference = "Stop"
if ($ec -ne 0) { Log "ERROR: football_data failed"; exit 1 }
Log "Football data done."

# --- Step 2: Refresh understat xG (all seasons) ---
Log "Step 2/6  Refreshing understat xG data..."
$ErrorActionPreference = "Continue"
& $PYTHON -m src.collect.understat
$ec = $LASTEXITCODE
$ErrorActionPreference = "Stop"
if ($ec -ne 0) { Log "WARNING: understat fetch had issues (continuing with cached data)" }
Log "Understat done."

# --- Step 3: Rebuild features ---
Log "Step 3/6  Building feature matrix..."
$ErrorActionPreference = "Continue"
& $PYTHON -m src.features.engineer
$ec = $LASTEXITCODE
$ErrorActionPreference = "Stop"
if ($ec -ne 0) { Log "ERROR: engineer failed"; exit 1 }
Log "Features done."

# --- Step 4: Retrain models ---
Log "Step 4/6  Training XGBoost + Random Forest..."
$ErrorActionPreference = "Continue"
& $PYTHON -m src.models.train
$ec = $LASTEXITCODE
$ErrorActionPreference = "Stop"
if ($ec -ne 0) { Log "ERROR: train failed"; exit 1 }
Log "Training done."

# --- Step 5: Evaluate models ---
Log "Step 5/6  Evaluating models (ROC-AUC, Brier, SHAP, gambling metrics)..."
$ErrorActionPreference = "Continue"
& $PYTHON -m src.models.evaluate
$ec = $LASTEXITCODE
$ErrorActionPreference = "Stop"
if ($ec -ne 0) { Log "ERROR: evaluate failed"; exit 1 }
Log "Evaluation done."

# --- Step 6: Push evaluation artefacts to website repo ---
Log "Step 6/6  Syncing evaluation artefacts to website..."
$date = Get-Date -Format "yyyy-MM-dd"

$filesToCopy = @(
    @{ Src = "output\calibration.png";         Dst = "data\calibration.png" },
    @{ Src = "output\shap_summary.png";        Dst = "data\shap_summary.png" },
    @{ Src = "output\pnl_curve.png";           Dst = "data\pnl_curve.png" },
    @{ Src = "output\threshold_sweep.png";     Dst = "data\threshold_sweep.png" },
    @{ Src = "output\website_evaluation.json"; Dst = "data\evaluation.json" }
)
foreach ($f in $filesToCopy) {
    $src = Join-Path $PL_DIR  $f.Src
    $dst = Join-Path $SITE_DIR $f.Dst
    if (Test-Path $src) {
        New-Item -ItemType Directory -Force -Path (Split-Path $dst) | Out-Null
        if (Test-Path $dst) {
            # Grant current user full control (handles files created by other processes/accounts)
            icacls $dst /grant "${env:USERNAME}:(F)" /Q 2>&1 | Out-Null
            Remove-Item $dst -Force -ErrorAction SilentlyContinue
        }
        try {
            Copy-Item $src $dst -ErrorAction Stop
            Log "  Copied $($f.Src) -> $($f.Dst)"
        } catch {
            Log "  WARNING: Could not copy $($f.Src) - $($_.Exception.Message)"
        }
    } else {
        Log "  WARNING: $src not found - skipping"
    }
}

Set-Location $SITE_DIR
git add "data\calibration.png" "data\shap_summary.png" "data\pnl_curve.png" "data\threshold_sweep.png" "data\evaluation.json"
$siteDiff = git diff --cached --name-only
if ($siteDiff) {
    git commit -m "chore: update evaluation artefacts [$date]"
    git push
    Log "Website evaluation artefacts pushed."
} else {
    Log "No evaluation artefact changes to push."
}

# --- Commit and push updated models + understat data to predictor repo ---
Set-Location $PL_DIR
Log "Committing updated models and data to GitHub..."
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
