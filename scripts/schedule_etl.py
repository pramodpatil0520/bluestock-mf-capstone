import os
import sys
import time
import argparse
import subprocess
from datetime import datetime
from pathlib import Path

# Resolve paths relative to this script
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
VENV_PYTHON = BASE_DIR / "venv" / "Scripts" / "python.exe"

# Fallback to current python interpreter if venv python doesn't exist
if not VENV_PYTHON.exists():
    VENV_PYTHON = Path(sys.executable)

def run_pipeline_step(script_name, args=[]):
    script_path = SCRIPT_DIR / script_name
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting step: {script_name}...")
    
    cmd = [str(VENV_PYTHON), str(script_path)] + args
    result = subprocess.run(cmd, cwd=str(BASE_DIR), capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ {script_name} completed successfully.")
        return True
    else:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ {script_name} failed with error:")
        print(result.stderr)
        return False

def run_full_etl():
    print("=" * 60)
    print(f"Bluestock Mutual Fund - Full Scheduled Pipeline Run")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Step 1: Live NAV Fetch (fetches NAV from mfapi.in)
    if not run_pipeline_step("live_nav_fetch.py"):
        print("Pipeline aborted at step 1.")
        return False
        
    # Step 2: ETL Pipeline (cleans and loads to DB)
    if not run_pipeline_step("etl_pipeline.py"):
        print("Pipeline aborted at step 2.")
        return False
        
    # Step 3: Compute Metrics (calculates CAGR, Sharpe, Sortino, etc.)
    if not run_pipeline_step("compute_metrics.py"):
        print("Pipeline aborted at step 3.")
        return False
        
    print("=" * 60)
    print(f"Pipeline completed successfully at {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 60)
    return True

def run_daemon_loop():
    print(f"Starting scheduled ETL pipeline daemon loop...")
    print(f"Will run every weekday (Mon-Fri) at 8:00 PM (20:00). Press Ctrl+C to exit.")
    
    last_run_date = None
    
    while True:
        now = datetime.now()
        # Monday = 0, Friday = 4, Saturday = 5, Sunday = 6
        is_weekday = now.weekday() < 5
        is_8pm = now.hour == 20 and now.minute == 0
        current_date = now.date()
        
        if is_weekday and is_8pm and current_date != last_run_date:
            print(f"Scheduled time reached. Triggering pipeline...")
            run_full_etl()
            last_run_date = current_date
            
        # Sleep for 30 seconds to avoid high CPU usage
        time.sleep(30)

def print_windows_scheduler_command():
    script_path = SCRIPT_DIR / "schedule_etl.py"
    
    print("\n" + "=" * 65)
    print(" NATIVE WINDOWS TASK SCHEDULER SETUP INSTRUCTIONS (B1)")
    print("=" * 65)
    print("To natively schedule the ETL pipeline to run every weekday at 8 PM")
    print("on Windows (without needing a console window open all the time),")
    print("open PowerShell as Administrator and run the following command:\n")
    
    ps_cmd = (
        f"$Action = New-ScheduledTaskAction -Execute '{VENV_PYTHON}' -Argument '{script_path} --run-now'\n"
        f"$Trigger = New-ScheduledTaskTrigger -Daily -At 8:00PM\n"
        f"$Settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries\n"
        f"Register-ScheduledTask -TaskName \"Bluestock_ETL_Weekday_8PM\" -Action $Action -Trigger $Trigger -Settings $Settings -Description \"Runs Bluestock Mutual Fund ETL Pipeline every weekday at 8 PM\""
    )
    print(ps_cmd)
    print("=" * 65 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bluestock Scheduled ETL Pipeline Manager")
    parser.add_argument("--run-now", action="store_true", help="Run the full ETL pipeline immediately")
    parser.add_argument("--daemon", action="store_true", help="Start the daemon loop to check time in background")
    args = parser.parse_args()
    
    if args.run_now:
        run_full_etl()
    elif args.daemon:
        run_daemon_loop()
    else:
        # Default: print instructions and run once for verification
        print_windows_scheduler_command()
        print("To run the ETL pipeline immediately for verification, run: python scripts/schedule_etl.py --run-now")
        print("To run as a daemon loop in background, run: python scripts/schedule_etl.py --daemon")
