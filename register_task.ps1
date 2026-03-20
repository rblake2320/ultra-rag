#Requires -Version 5.1
# Register the UltraRAG-AutoRestart scheduled task

$action = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument "-ExecutionPolicy Bypass -WindowStyle Hidden -File D:\rag-ingest\ensure_running.ps1"

$triggerBoot   = New-ScheduledTaskTrigger -AtStartup
$triggerRepeat = New-ScheduledTaskTrigger -Once -At (Get-Date) `
    -RepetitionInterval (New-TimeSpan -Minutes 5) `
    -RepetitionDuration (New-TimeSpan -Days 1826)

$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable

Unregister-ScheduledTask -TaskName "UltraRAG-AutoRestart" -Confirm:$false -ErrorAction SilentlyContinue

$task = Register-ScheduledTask `
    -TaskName "UltraRAG-AutoRestart" `
    -Action $action `
    -Trigger $triggerBoot, $triggerRepeat `
    -Settings $settings `
    -Force

Write-Host "Task '$($task.TaskName)' registered with state: $($task.State)"
