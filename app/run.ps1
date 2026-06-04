# Lanceur de l'application (PowerShell)
# Usage : clic droit > Executer avec PowerShell, ou  .\run.ps1
Set-Location $PSScriptRoot
Write-Host "Lancement de l'application BI & ML du centre d'appel..." -ForegroundColor Cyan
python -m streamlit run app.py
