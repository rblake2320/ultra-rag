# UltraRAG Windows Service Installer
# Run as Administrator:
# powershell -ExecutionPolicy Bypass -File D:\ultra-rag\setup-ultrarag-service.ps1

$ErrorActionPreference = "Stop"

$NSSM = "D:\tools\nssm\nssm.exe"
$PYTHON = "C:\Python312\python.exe"
$RAG_DIR = "D:\ultra-rag"
$LOG_DIR = "$RAG_DIR\logs"

$principal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw "Administrator access is required. Re-run PowerShell as Administrator."
}

New-Item -ItemType Directory -Force -Path $LOG_DIR | Out-Null

if (-not (Test-Path $NSSM)) {
    throw "NSSM not found at $NSSM"
}

if (-not (Test-Path "$RAG_DIR\.env")) {
    throw "Missing $RAG_DIR\.env. Create it with RAG_DB_PASSWORD before installing the service."
}

$existing = Get-Service UltraRAG-API -ErrorAction SilentlyContinue
if ($existing) {
    & $NSSM stop UltraRAG-API 2>$null
    & $NSSM remove UltraRAG-API confirm 2>$null
}

& $NSSM install UltraRAG-API $PYTHON
& $NSSM set UltraRAG-API AppParameters "ultra_server.py --host 0.0.0.0 --port 8300"
& $NSSM set UltraRAG-API AppDirectory $RAG_DIR
& $NSSM set UltraRAG-API AppStdout "$LOG_DIR\ultrarag_service.log"
& $NSSM set UltraRAG-API AppStderr "$LOG_DIR\ultrarag_service_err.log"
& $NSSM set UltraRAG-API AppStdoutCreationDisposition 4
& $NSSM set UltraRAG-API AppStderrCreationDisposition 4
& $NSSM set UltraRAG-API AppRotateFiles 1
& $NSSM set UltraRAG-API AppRotateBytes 10485760
& $NSSM set UltraRAG-API AppRestartDelay 5000
& $NSSM set UltraRAG-API AppThrottle 10000
& $NSSM set UltraRAG-API Description "UltraRAG API - FastAPI/Uvicorn on port 8300"
& $NSSM set UltraRAG-API Start SERVICE_AUTO_START
& $NSSM set UltraRAG-API ObjectName LocalSystem

& $NSSM start UltraRAG-API
Start-Sleep -Seconds 8

Get-Service UltraRAG-API | Format-Table Name, Status, StartType -AutoSize
