import subprocess
import time
import requests
import sys
import os

services = [
    {
        "name": "gateway_service",
        "port": 8000,
        "cwd": r"c:\Users\USER\Desktop\fifth year\وظائف ومشاريع الجامعة\IR\IR-Project\Moli-Search-Engine\services\gateway_service",
        "pythonpath": r"c:\Users\USER\Desktop\fifth year\وظائف ومشاريع الجامعة\IR\IR-Project\Moli-Search-Engine\services\gateway_service"
    },
    {
        "name": "preprocessing_service",
        "port": 8001,
        "cwd": r"c:\Users\USER\Desktop\fifth year\وظائف ومشاريع الجامعة\IR\IR-Project\Moli-Search-Engine\services\preprocessing_service",
        "pythonpath": r"c:\Users\USER\Desktop\fifth year\وظائف ومشاريع الجامعة\IR\IR-Project\Moli-Search-Engine\services\preprocessing_service"
    },
    {
        "name": "indexing_service",
        "port": 8002,
        "cwd": r"c:\Users\USER\Desktop\fifth year\وظائف ومشاريع الجامعة\IR\IR-Project\Moli-Search-Engine\services\indexing_service",
        "pythonpath": r"c:\Users\USER\Desktop\fifth year\وظائف ومشاريع الجامعة\IR\IR-Project\Moli-Search-Engine\services\indexing_service"
    },
    {
        "name": "retrieval_service",
        "port": 8003,
        "cwd": r"c:\Users\USER\Desktop\fifth year\وظائف ومشاريع الجامعة\IR\IR-Project\Moli-Search-Engine\services\retrieval_service",
        "pythonpath": r"c:\Users\USER\Desktop\fifth year\وظائف ومشاريع الجامعة\IR\IR-Project\Moli-Search-Engine\services\retrieval_service"
    },
    {
        "name": "clustering_service",
        "port": 8004,
        "cwd": r"c:\Users\USER\Desktop\fifth year\وظائف ومشاريع الجامعة\IR\IR-Project\Moli-Search-Engine\services\clustering_service",
        "pythonpath": r"c:\Users\USER\Desktop\fifth year\وظائف ومشاريع الجامعة\IR\IR-Project\Moli-Search-Engine\services\clustering_service"
    },
    {
        "name": "evaluation_service",
        "port": 8005,
        "cwd": r"c:\Users\USER\Desktop\fifth year\وظائف ومشاريع الجامعة\IR\IR-Project\Moli-Search-Engine\services\evaluation_service",
        "pythonpath": r"c:\Users\USER\Desktop\fifth year\وظائف ومشاريع الجامعة\IR\IR-Project\Moli-Search-Engine\services\evaluation_service"
    },
    {
        "name": "query_refinement_service",
        "port": 8006,
        "cwd": r"c:\Users\USER\Desktop\fifth year\وظائف ومشاريع الجامعة\IR\IR-Project\Moli-Search-Engine\services\query_refinement_service",
        "pythonpath": r"c:\Users\USER\Desktop\fifth year\وظائف ومشاريع الجامعة\IR\IR-Project\Moli-Search-Engine\services\query_refinement_service"
    },
    {
        "name": "frontend_service",
        "port": 8007,
        "cwd": r"c:\Users\USER\Desktop\fifth year\وظائف ومشاريع الجامعة\IR\IR-Project\Moli-Search-Engine\services\frontend_service",
        "pythonpath": r"c:\Users\USER\Desktop\fifth year\وظائف ومشاريع الجامعة\IR\IR-Project\Moli-Search-Engine\services\frontend_service"
    }
]

def print_safe(text):
    try:
        print(text)
    except UnicodeEncodeError:
        try:
            encoding = sys.stdout.encoding or 'ascii'
            print(text.encode(encoding, errors='replace').decode(encoding))
        except Exception:
            print(text.encode('ascii', errors='replace').decode('ascii'))

print_safe("=== VERIFYING STARTUP OF ALL LOCAL MICROSERVICES ===")

any_failed = False

for svc in services:
    name = svc["name"]
    port = svc["port"]
    cwd = svc["cwd"]
    
    print_safe(f"\n---> Starting {name} on port {port}...")
    
    env = os.environ.copy()
    env["PYTHONPATH"] = svc["pythonpath"]
    
    log_path = f"uvicorn_{name}_log.txt"
    lf = open(log_path, "w", encoding="utf-8", errors="replace")
    
    venv_python = r"c:\Users\USER\Desktop\fifth year\وظائف ومشاريع الجامعة\IR\IR-Project\Moli-Search-Engine\.venv\Scripts\python.exe"
    python_executable = venv_python if os.path.exists(venv_python) else sys.executable
    
    proc = subprocess.Popen(
        [python_executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(port)],
        cwd=cwd,
        env=env,
        stdout=lf,
        stderr=lf
    )
    
    # Poll health check for up to 90 seconds
    health_url = f"http://127.0.0.1:{port}/health"
    is_healthy = False
    last_error = None
    
    for attempt in range(90):
        try:
            res = requests.get(health_url, timeout=2.0)
            if res.status_code == 200:
                print_safe(f"[OK] {name} is healthy: {res.json()}")
                is_healthy = True
                break
            else:
                last_error = f"Status code: {res.status_code}"
        except Exception as e:
            last_error = str(e)
        time.sleep(1)
        
    if not is_healthy:
        print_safe(f"[FAILED] Could not connect to {name} after 30 seconds: {last_error}")
        any_failed = True
        
        lf.close()
        try:
            with open(os.path.join(cwd, log_path) if not os.path.exists(log_path) else log_path, "r", encoding="utf-8", errors="replace") as f:
                print_safe(f"--- LOGS for {name} ---")
                print_safe(f.read())
                print_safe("-----------------------")
        except Exception as log_ex:
            print_safe(f"Could not read logs for {name}: {log_ex}")
            
    # Terminate service
    proc.terminate()
    proc.wait()
    lf.close()
    
    # Clean up log file
    for p in [log_path, os.path.join(cwd, log_path)]:
        if os.path.exists(p):
            try:
                os.remove(p)
            except:
                pass

print_safe("\n=== STARTUP VERIFICATION COMPLETED ===")
if any_failed:
    sys.exit(1)
else:
    sys.exit(0)
