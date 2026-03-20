#Requires -Version 5.1
<#
.SYNOPSIS
    Health-check and auto-restart for Ultra RAG server (port 8300).
    Registered as a Windows Scheduled Task that runs every 5 minutes.
#>

$logFile = "D:\rag-ingest\logs\auto_restart.log"
$ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

# Ensure log directory exists
$logDir = Split-Path $logFile
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir -Force | Out-Null }

# ── Circuit breaker ────────────────────────────────────────────────────────────
# Stop restarting if we've restarted 3+ times in the last 15 minutes.
if (Test-Path $logFile) {
    $cutoff = (Get-Date).AddMinutes(-15)
    $recentRestarts = 0
    Get-Content $logFile | ForEach-Object {
        if ($_ -match "^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) - Ultra RAG restarted") {
            try {
                $entryTime = [datetime]::ParseExact($Matches[1], "yyyy-MM-dd HH:mm:ss", $null)
                if ($entryTime -gt $cutoff) { $recentRestarts++ }
            } catch {}
        }
    }
    if ($recentRestarts -ge 3) {
        Add-Content -Path $logFile -Value "$ts - CIRCUIT BREAKER: $recentRestarts restarts in last 15min — skipping. Check D:\rag-ingest\logs\ultra_server.log"
        exit 0
    }
}

# ── Health check via HTTP ─────────────────────────────────────────────────────
$healthy = $false
try {
    $resp = Invoke-WebRequest -Uri "http://localhost:8300/api/health" -TimeoutSec 5 -UseBasicParsing -ErrorAction Stop
    if ($resp.StatusCode -eq 200) { $healthy = $true }
} catch {
    # Connection refused, timeout, or non-200 → not healthy
}

if ($healthy) {
    # Server is up — nothing to do
    exit 0
}

# ── Kill any zombie holding port 8300 ─────────────────────────────────────────
$existing = Get-NetTCPConnection -LocalPort 8300 -State Listen -ErrorAction SilentlyContinue
if ($existing) {
    foreach ($conn in $existing) {
        Stop-Process -Id $conn.OwningProcess -Force -ErrorAction SilentlyContinue
    }
    Start-Sleep -Seconds 2
}

# ── Start Ultra RAG server ────────────────────────────────────────────────────
$pythonExe = "C:\Python312\python.exe"
if (-not (Test-Path $pythonExe)) {
    Add-Content -Path $logFile -Value "$ts - ERROR: Python not found at $pythonExe"
    exit 1
}

Start-Process -FilePath $pythonExe `
    -ArgumentList "ultra_server.py" `
    -WorkingDirectory "D:\rag-ingest" `
    -WindowStyle Hidden `
    -RedirectStandardOutput "D:\rag-ingest\logs\ultra_server_stdout.log" `
    -RedirectStandardError  "D:\rag-ingest\logs\ultra_server_stderr.log"

Add-Content -Path $logFile -Value "$ts - Ultra RAG restarted (was not healthy)"
