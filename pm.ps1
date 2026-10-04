# PowerShell shortcut for Project Manager Orchestrator
$PYTHON = "D:\Users\tssud\miniconda3\python.exe"
if (-not (Test-Path $PYTHON)) {
    $PYTHON = (Get-Command python -ErrorAction SilentlyContinue).Source
}

& $PYTHON "$PSScriptRoot\pm_orchestrator.py" @args
