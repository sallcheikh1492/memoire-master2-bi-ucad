# Regenere le rapport .docx complet (figures + chiffres + assemblage Word).
# Usage : clic droit > Executer avec PowerShell, ou depuis ce dossier :  .\make_report.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# Localise python (non present dans le PATH sur ce poste)
$py = "C:\Users\sall1\AppData\Local\Programs\Python\Python38\python.exe"
if (-not (Test-Path $py)) { $py = (Get-Command python -ErrorAction SilentlyContinue).Source }
if (-not $py) { Write-Error "python introuvable"; exit 1 }

Write-Host "[1/2] Generation des figures et des chiffres..." -ForegroundColor Cyan
& $py "$PSScriptRoot\make_figures.py"

Write-Host "[2/2] Assemblage du document Word..." -ForegroundColor Cyan
node "$PSScriptRoot\build_report.js"

Write-Host "Termine : $PSScriptRoot\Rapport_Chapitres_III_IV.docx" -ForegroundColor Green
