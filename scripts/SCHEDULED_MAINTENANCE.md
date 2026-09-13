# Scheduled maintenance

`scheduled_maintenance.py` replaces silent batch/PowerShell maintenance actions.
Supply a private JSON configuration with `receipts` and per-job configuration;
host paths, SSH targets, bucket names and secret-file references stay outside Git.
The installer requires an existing definition for each of the five tasks. It
preserves schedules and run-as principals, backs up XML, enables missed-start
catch-up, removes battery blocking, uses explicit executable/working paths and
prevents overlapping task instances. It does not wake a sleeping computer.

Run the installer from the deployed checkout:

```powershell
./scripts/Install-ScheduledMaintenance.ps1 -ConfigPath <private-json> -PythonPath <python-exe>
```

Jobs and evidence:

* `echo`: verify exact Git root and approved private remote before staging;
  reject unreadable state; commit only when changed; push and compare remote SHA.
* `bridge`: use the configured database and user-scoped MemoryWeb key; retain
  deterministic session files and asynchronous task IDs; only mark rows done
  after verified ingestion. Ambiguous sends stop for reconciliation, never replay
  automatically. An empty queue is reported as zero processed, not new ingestion.
* `personal`: incremental S3 downloads and native OpenSSH/SFTP replace malformed
  multiline batch Python and rsync. Required-source failures are recorded and
  fail the run. Parsing and embedding must complete without errors before success.
  A failed source delays ingestion until the next successful complete sync.
* `watchdog`: check identity of the existing service and restart authorization,
  check HTTP and database health, restart through SCM only, verify health after
  restart, limit attempts to three per 15 minutes. Never kill an arbitrary port
  owner or launch another installation.

Every run produces a private `*.latest.json`, timestamped receipt and bounded log.
A kernel lock releases automatically when the process exits. Exit 1 means work
failed; exit 75 means the job lock or receipt storage was unavailable. Neither is
success. PKA retains its original snapshot implementation and output locations.
Periodic tasks also require actual scheduled-run timestamps and their receipts
to establish trigger operation. A future NextRunTime alone is insufficient.

Dependencies: Python 3.12, boto3, httpx, psycopg2, the configured UltraRAG modules,
Git, Windows OpenSSH, existing AWS/SSH credentials. The task user must have read
access to state/configuration and query/start/stop rights on the watched service.
Do not grant an entire task administrator rights merely to restart one service.

Regression tests: `python -m pytest tests/test_scheduled_maintenance.py -q`.
The local Git-boundary and child-exit tests execute real binaries; ingestion
response tests use HTTP/DB doubles and do not constitute live ingestion proof.
Remote-source and service-recovery proof must be retained separately.
