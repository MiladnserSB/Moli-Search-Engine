# import subprocess
# import time
# import requests
# import sys
# import os

# services = [
#     {
#         "name": "gateway_service",
#         "port": 8000,
#         "cwd": r"c:\Users\st\Desktop\Moli-Search-Engine\services\gateway_service",
#         "pythonpath": r"c:\Users\st\Desktop\Moli-Search-Engine\services\gateway_service"
#     },
#     {
#         "name": "preprocessing_service",
#         "port": 8001,
#         "cwd": r"c:\Users\st\Desktop\Moli-Search-Engine\services\preprocessing_service",
#         "pythonpath": r"c:\Users\st\Desktop\Moli-Search-Engine\services\preprocessing_service"
#     },
#     {
#         "name": "indexing_service",
#         "port": 8002,
#         "cwd": r"c:\Users\st\Desktop\Moli-Search-Engine\services\indexing_service",
#         "pythonpath": r"c:\Users\st\Desktop\Moli-Search-Engine\services\indexing_service"
#     },
#     {
#         "name": "retrieval_service",
#         "port": 8003,
#         "cwd": r"c:\Users\st\Desktop\Moli-Search-Engine\services\retrieval_service",
#         "pythonpath": r"c:\Users\st\Desktop\Moli-Search-Engine\services\retrieval_service"
#     },
#     {
#         "name": "clustering_service",
#         "port": 8004,
#         "cwd": r"c:\Users\st\Desktop\Moli-Search-Engine\services\clustering_service",
#         "pythonpath": r"c:\Users\st\Desktop\Moli-Search-Engine\services\clustering_service"
#     },
#     {
#         "name": "evaluation_service",
#         "port": 8005,
#         "cwd": r"c:\Users\st\Desktop\Moli-Search-Engine\services\evaluation_service",
#         "pythonpath": r"c:\Users\st\Desktop\Moli-Search-Engine\services\evaluation_service"
#     },
#     {
#         "name": "query_refinement_service",
#         "port": 8006,
#         "cwd": r"c:\Users\st\Desktop\Moli-Search-Engine\services\query_refinement_service",
#         "pythonpath": r"c:\Users\st\Desktop\Moli-Search-Engine\services\query_refinement_service"
#     },
#     {
#         "name": "frontend_service",
#         "port": 8007,
#         "cwd": r"c:\Users\st\Desktop\Moli-Search-Engine\services\frontend_service",
#         "pythonpath": r"c:\Users\st\Desktop\Moli-Search-Engine\services\frontend_service"
#     }
# ]

# def print_safe(text):
#     try:
#         print(text)
#     except UnicodeEncodeError:
#         try:
#             encoding = sys.stdout.encoding or 'ascii'
#             print(text.encode(encoding, errors='replace').decode(encoding))
#         except Exception:
#             print(text.encode('ascii', errors='replace').decode('ascii'))

# print_safe("==============================================================")
# print_safe("   STARTING ALL IR HUB SERVICES CONCURRENTLY   ")
# print_safe("==============================================================")

# running_processes = []

# try:
#     venv_python = r"c:\Users\st\Desktop\Moli-Search-Engine\.venv\Scripts\python.exe"
#     python_executable = venv_python if os.path.exists(venv_python) else sys.executable
    
#     for svc in services:
#         name = svc["name"]
#         port = svc["port"]
#         cwd = svc["cwd"]
        
#         print_safe(f"\n[LAUNCH] Starting {name} on port {port}...")
        
#         env = os.environ.copy()
#         env["PYTHONPATH"] = svc["pythonpath"]
        
#         log_path = f"uvicorn_{name}_log.txt"
#         lf = open(log_path, "w", encoding="utf-8", errors="replace")
        
#         proc = subprocess.Popen(
#             [python_executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(port)],
#             cwd=cwd,
#             env=env,
#             stdout=lf,
#             stderr=lf
#         )
#         running_processes.append((name, proc, lf, log_path, port))
        
#     print_safe("\nWaiting for services to initialize health checks...")
#     time.sleep(5)
    
#     # Check health status with retries (increased for model preloading)
#     all_healthy = True
#     for name, proc, lf, log_path, port in running_processes:
#         health_url = f"http://127.0.0.1:{port}/health"
#         healthy = False
#         for attempt in range(60):
#             try:
#                 res = requests.get(health_url, timeout=3.0)
#                 if res.status_code == 200:
#                     print_safe(f"  [OK] {name} (Port {port}): Healthy -> {res.json()}")
#                     healthy = True
#                     break
#             except Exception:
#                 pass
#             time.sleep(2)
#         if not healthy:
#             print_safe(f"  [ERROR] {name} (Port {port}): Could not connect or unhealthy after multiple retries")
#             all_healthy = False
            
#     if all_healthy:
#         print_safe("\n🚀 SUCCESS: All services are successfully running and verified healthy!")
#         print_safe("👉 Open your browser at: http://127.0.0.1:8007")
#     else:
#         print_safe("\n⚠️ WARNING: Some services had issues starting or answering health checks.")
#         print_safe("Check the respective uvicorn_*_log.txt files for details.")
        
#     print_safe("\nPress Ctrl+C to terminate all services and exit.")
    
#     # Keep the script alive
#     while True:
#         # Check if any process has died
#         for name, proc, lf, log_path, port in running_processes:
#             poll = proc.poll()
#             if poll is not None:
#                 print_safe(f"\n🚨 ALERT: Service '{name}' exited unexpectedly with code {poll}!")
#                 # Show last 10 lines of logs
#                 try:
#                     with open(log_path, "r", encoding="utf-8", errors="replace") as f:
#                         lines = f.readlines()
#                         print_safe(f"--- Last 10 log lines of {name} ---")
#                         for line in lines[-10:]:
#                             print_safe(line.strip())
#                         print_safe("-----------------------------------")
#                 except:
#                     pass
#         time.sleep(5)

# except KeyboardInterrupt:
#     print_safe("\n\n[SHUTDOWN] Ctrl+C detected. Terminating all microservices...")

# finally:
#     # Gracefully terminate all children
#     for name, proc, lf, log_path, port in running_processes:
#         print_safe(f"Terminating {name}...")
#         proc.terminate()
        
#     # Wait for them to stop
#     for name, proc, lf, log_path, port in running_processes:
#         try:
#             proc.wait(timeout=3)
#         except subprocess.TimeoutExpired:
#             print_safe(f"Force-killing {name}...")
#             proc.kill()
#         lf.close()
        
#         # Clean up log files if they are empty or small, or keep them for debugging.
#         # Let's delete them to avoid polluting the workspace
#         if os.path.exists(log_path):
#             try:
#                 os.remove(log_path)
#             except:
#                 pass
                
#     print_safe("\n[SHUTDOWN] All services terminated successfully.")


import subprocess
import time
import requests
import sys
import os

services = [
    {
        "name": "gateway_service",
        "port": 8000,
        "cwd": r"c:\Users\st\Desktop\Moli-Search-Engine\services\gateway_service",
        "pythonpath": r"c:\Users\st\Desktop\Moli-Search-Engine\services\gateway_service"
    },
    {
        "name": "preprocessing_service",
        "port": 8001,
        "cwd": r"c:\Users\st\Desktop\Moli-Search-Engine\services\preprocessing_service",
        "pythonpath": r"c:\Users\st\Desktop\Moli-Search-Engine\services\preprocessing_service"
    },
    {
        "name": "indexing_service",
        "port": 8002,
        "cwd": r"c:\Users\st\Desktop\Moli-Search-Engine\services\indexing_service",
        "pythonpath": r"c:\Users\st\Desktop\Moli-Search-Engine\services\indexing_service"
    },
    {
        "name": "retrieval_service",
        "port": 8003,
        "cwd": r"c:\Users\st\Desktop\Moli-Search-Engine\services\retrieval_service",
        "pythonpath": r"c:\Users\st\Desktop\Moli-Search-Engine\services\retrieval_service"
    },
    {
        "name": "clustering_service",
        "port": 8004,
        "cwd": r"c:\Users\st\Desktop\Moli-Search-Engine\services\clustering_service",
        "pythonpath": r"c:\Users\st\Desktop\Moli-Search-Engine\services\clustering_service"
    },
    {
        "name": "evaluation_service",
        "port": 8005,
        "cwd": r"c:\Users\st\Desktop\Moli-Search-Engine\services\evaluation_service",
        "pythonpath": r"c:\Users\st\Desktop\Moli-Search-Engine\services\evaluation_service"
    },
    {
        "name": "query_refinement_service",
        "port": 8006,
        "cwd": r"c:\Users\st\Desktop\Moli-Search-Engine\services\query_refinement_service",
        "pythonpath": r"c:\Users\st\Desktop\Moli-Search-Engine\services\query_refinement_service"
    },
    {
        "name": "frontend_service",
        "port": 8007,
        "cwd": r"c:\Users\st\Desktop\Moli-Search-Engine\services\frontend_service",
        "pythonpath": r"c:\Users\st\Desktop\Moli-Search-Engine\services\frontend_service"
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

print_safe("==============================================================")
print_safe("   STARTING ALL IR HUB SERVICES CONCURRENTLY   ")
print_safe("==============================================================")

running_processes = []

try:
    venv_python = r"c:\Users\st\Desktop\Moli-Search-Engine\.venv\Scripts\python.exe"
    python_executable = venv_python if os.path.exists(venv_python) else sys.executable
    
    for svc in services:
        name = svc["name"]
        port = svc["port"]
        cwd = svc["cwd"]
        
        print_safe(f"\n[LAUNCH] Starting {name} on port {port}...")
        
        env = os.environ.copy()
        env["PYTHONPATH"] = svc["pythonpath"]
        
        # Launch process WITHOUT redirecting stdout/stderr to files
        # This makes all Uvicorn output appear directly in this terminal.
        proc = subprocess.Popen(
            [python_executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(port)],
            cwd=cwd,
            env=env
        )
        running_processes.append((name, proc, port))
        
    print_safe("\nWaiting for services to initialize health checks...")
    time.sleep(5)
    
    # Check health status with retries (increased for model preloading)
    all_healthy = True
    for name, proc, port in running_processes:
        health_url = f"http://127.0.0.1:{port}/health"
        healthy = False
        for attempt in range(60):
            try:
                res = requests.get(health_url, timeout=3.0)
                if res.status_code == 200:
                    print_safe(f"  [OK] {name} (Port {port}): Healthy -> {res.json()}")
                    healthy = True
                    break
            except Exception:
                pass
            time.sleep(2)
        if not healthy:
            print_safe(f"  [ERROR] {name} (Port {port}): Could not connect or unhealthy after multiple retries")
            all_healthy = False
            
    if all_healthy:
        print_safe("\n🚀 SUCCESS: All services are successfully running and verified healthy!")
        print_safe("👉 Open your browser at: http://127.0.0.1:8007")
    else:
        print_safe("\n⚠️ WARNING: Some services had issues starting or answering health checks.")
        print_safe("Check the console output above for error details.")
        
    print_safe("\nPress Ctrl+C to terminate all services and exit.")
    
    # Keep the script alive and monitor processes
    while True:
        # Check if any process has died
        for name, proc, port in running_processes:
            poll = proc.poll()
            if poll is not None:
                print_safe(f"\n🚨 ALERT: Service '{name}' exited unexpectedly with code {poll}!")
                # No log files to read, but we can note that output appears in console
        time.sleep(5)

except KeyboardInterrupt:
    print_safe("\n\n[SHUTDOWN] Ctrl+C detected. Terminating all microservices...")

finally:
    # Gracefully terminate all children
    for name, proc, port in running_processes:
        print_safe(f"Terminating {name}...")
        proc.terminate()
        
    # Wait for them to stop
    for name, proc, port in running_processes:
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            print_safe(f"Force-killing {name}...")
            proc.kill()
                
    print_safe("\n[SHUTDOWN] All services terminated successfully.")