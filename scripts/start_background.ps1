param(
    [int]$Updates = 300,
    [string]$Seeds = '42 43 44',
    [int]$Episodes = 32
)
$ErrorActionPreference = 'Stop'
$repoPath = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $repoPath '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'Create .venv and install requirements-assignment.txt first.' }
$logPath = Join-Path $repoPath 'logs'
New-Item -ItemType Directory -Force $logPath | Out-Null
$pidFile = Join-Path $logPath 'pipeline.pid'
if (Test-Path -LiteralPath $pidFile) {
    $workerId = [int](Get-Content -LiteralPath $pidFile)
    $existingWorker = Get-Process -Id $workerId -ErrorAction SilentlyContinue
    if ($existingWorker) { throw "A worker with PID $workerId exists. Verify it before launching another run." }
}
if ($Seeds -notmatch '^\d+( \d+)*$') { throw 'Seeds must be space-separated integers.' }
$worker = Start-Process -FilePath $pythonPath -WorkingDirectory $repoPath `
    -ArgumentList "-u scripts/run_pipeline.py --updates $Updates --seeds $Seeds --episodes $Episodes" `
    -RedirectStandardOutput (Join-Path $logPath 'pipeline.stdout.log') `
    -RedirectStandardError (Join-Path $logPath 'pipeline.stderr.log') `
    -WindowStyle Hidden -PassThru
$worker.Id | Set-Content -LiteralPath $pidFile
Write-Output "Background pipeline started. PID: $($worker.Id)"
Write-Output 'Status: logs/pipeline.status.json; per-stage logs: logs/*.log'
