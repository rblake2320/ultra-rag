@echo off
echo [%TIME%] Starting personal collection - stage 1: parse + embed
cd /d D:\rag-ingest
python ultra_ingest.py personal --stages parse,embed >> logs\personal_full.log 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [%TIME%] Stage 1 FAILED - check logs\personal_full.log
    exit /b 1
)
echo [%TIME%] Stage 1 complete. Starting stage 2: contextual + parents + kg + communities + raptor
python ultra_ingest.py personal --stages contextual,parents,kg,communities,raptor >> logs\personal_full.log 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [%TIME%] Stage 2 FAILED - check logs\personal_full.log
    exit /b 1
)
echo [%TIME%] Personal collection fully ingested.
