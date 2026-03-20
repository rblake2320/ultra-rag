$action = New-ScheduledTaskAction `
    -Execute "C:\Python312\python.exe" `
    -Argument "bridge_worker.py --once" `
    -WorkingDirectory "D:\rag-ingest"

$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date) `
    -RepetitionInterval (New-TimeSpan -Minutes 2) `
    -RepetitionDuration (New-TimeSpan -Days 1826)

$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 5)

Unregister-ScheduledTask -TaskName "UltraRAG-BridgeWorker" -Confirm:$false -ErrorAction SilentlyContinue

Register-ScheduledTask `
    -TaskName "UltraRAG-BridgeWorker" `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Force

Write-Host "BridgeWorker task registered"
