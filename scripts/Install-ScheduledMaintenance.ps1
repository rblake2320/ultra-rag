param(
    [Parameter(Mandatory=$true)][string]$ConfigPath,
    [Parameter(Mandatory=$true)][string]$PythonPath
)
$ErrorActionPreference = 'Stop'
$ConfigPath = (Resolve-Path -LiteralPath $ConfigPath).Path
$PythonPath = (Resolve-Path -LiteralPath $PythonPath).Path
$runner = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot 'scheduled_maintenance.py')).Path
$backup = Join-Path (Split-Path $ConfigPath -Parent) ('definitions-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
New-Item -ItemType Directory -Path $backup | Out-Null
$jobs = [ordered]@{
    'EchoStateBackup'='echo'
    'PersonalRAG-Sync'='personal'
    'UltraRAG-BridgeWorker'='bridge'
    'UltraRAG-AutoRestart'='watchdog'
}
foreach ($name in $jobs.Keys) {
    Export-ScheduledTask -TaskName $name -TaskPath '\' | Set-Content -LiteralPath (Join-Path $backup ($name+'.xml')) -Encoding Unicode
    $minutes = if ($jobs[$name] -eq 'personal') {240} else {4}
    $settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes $minutes) -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 5)
    $arguments = '"'+$runner+'" '+$jobs[$name]+' --config "'+$ConfigPath+'"'
    $action = New-ScheduledTaskAction -Execute $PythonPath -Argument $arguments -WorkingDirectory (Split-Path $PSScriptRoot -Parent)
    Set-ScheduledTask -TaskName $name -TaskPath '\' -Action $action -Settings $settings | Out-Null
    Enable-ScheduledTask -TaskName $name -TaskPath '\' | Out-Null
}
$name = 'PKA Terminal Recovery Autosave'
Export-ScheduledTask -TaskName $name -TaskPath '\' | Set-Content -LiteralPath (Join-Path $backup ($name+'.xml')) -Encoding Unicode
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 4)
Set-ScheduledTask -TaskName $name -TaskPath '\' -Settings $settings | Out-Null
Enable-ScheduledTask -TaskName $name -TaskPath '\' | Out-Null
Write-Output 'Task settings installed. This does not assert that task outcomes passed.'
