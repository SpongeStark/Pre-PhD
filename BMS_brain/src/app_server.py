import os
import sys
import json
import subprocess
import threading
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

# Paths
SRC_DIR = Path(__file__).parent.resolve()
ROOT_DIR = SRC_DIR.parent.resolve()
UI_DIR = SRC_DIR / "web_ui"
RESULTS_FORECAST_DIR = ROOT_DIR / "results_forecasting"
RESULTS_FORECAST_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_MPC_DIR = ROOT_DIR / "results_mpc"
RESULTS_MPC_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_RT_DIR = ROOT_DIR / "results_rt"
RESULTS_RT_DIR.mkdir(parents=True, exist_ok=True)

# Global state to keep track of running scripts
# Statuses: "idle", "running", "success", "error"
execution_state = {
    "pv_cleaner": {"status": "idle", "logs": "", "pid": None},
    "con_cleaner": {"status": "idle", "logs": "", "pid": None},
    "battery_optimizer": {"status": "idle", "logs": "", "pid": None},
    "battery_optimizer_ev": {"status": "idle", "logs": "", "pid": None},
    "battery_optimizer_caltech": {"status": "idle", "logs": "", "pid": None},
    "forecast_pv": {"status": "idle", "logs": "", "pid": None},
    "forecast_con": {"status": "idle", "logs": "", "pid": None},
    "forecast_ev": {"status": "idle", "logs": "", "pid": None},
    "forecast_caltech": {"status": "idle", "logs": "", "pid": None},
    "forecast_all": {"status": "idle", "logs": "", "pid": None},
    "mpc_supermarket": {"status": "idle", "logs": "", "pid": None},
    "mpc_ev": {"status": "idle", "logs": "", "pid": None},
    "mpc_caltech": {"status": "idle", "logs": "", "pid": None},
    "rt_supermarket": {"status": "idle", "logs": "", "pid": None},
    "rt_ev": {"status": "idle", "logs": "", "pid": None},
    "rt_caltech": {"status": "idle", "logs": "", "pid": None},
    "rt_all": {"status": "idle", "logs": "", "pid": None},
    "pipeline": {"status": "idle", "logs": "", "current_step": None, "pid": None}
}

state_lock = threading.Lock()

# Mapping script identifiers to command argument lists
SCRIPT_COMMANDS = {
    "pv_cleaner": [sys.executable, "-u", str(SRC_DIR / "data_cleaner.py")],
    "con_cleaner": [sys.executable, "-u", str(SRC_DIR / "data_cleaner_con.py")],
    "battery_optimizer": [sys.executable, "-u", str(SRC_DIR / "battery_optimizer.py")],
    "battery_optimizer_ev": [sys.executable, "-u", str(SRC_DIR / "battery_optimizer.py"), "--with-ev"],
    "battery_optimizer_caltech": [sys.executable, "-u", str(SRC_DIR / "battery_optimizer.py"), "--mode", "caltech_ev"],
    "forecast_pv": [sys.executable, "-u", str(SRC_DIR / "forecaster_pv.py")],
    "forecast_con": [sys.executable, "-u", str(SRC_DIR / "forecaster_con.py")],
    "forecast_ev": [sys.executable, "-u", str(SRC_DIR / "forecaster_ev.py")],
    "forecast_caltech": [sys.executable, "-u", str(SRC_DIR / "forecaster_caltech.py")],
    "forecast_all": [sys.executable, "-u", str(SRC_DIR / "forecaster_engine.py"), "--target", "all", "--model", "compare"],
    "mpc_supermarket": [sys.executable, "-u", str(SRC_DIR / "mpc_supermarket.py")],
    "mpc_ev": [sys.executable, "-u", str(SRC_DIR / "mpc_ev.py")],
    "mpc_caltech": [sys.executable, "-u", str(SRC_DIR / "mpc_caltech.py")],
    "rt_supermarket": [sys.executable, "-u", str(SRC_DIR / "rt_supermarket.py")],
    "rt_ev": [sys.executable, "-u", str(SRC_DIR / "rt_ev.py")],
    "rt_caltech": [sys.executable, "-u", str(SRC_DIR / "rt_caltech.py")],
    "rt_all": [sys.executable, "-u", str(SRC_DIR / "rt_controller.py"), "--mode", "all"]
}

def run_script_thread(script_key, on_complete=None):
    """Executes a single script, capturing logs line-by-line."""
    cmd = SCRIPT_COMMANDS.get(script_key)
    if not cmd:
        with state_lock:
            execution_state[script_key]["status"] = "error"
            execution_state[script_key]["logs"] = f"Error: No command defined for {script_key}\n"
        if on_complete:
            on_complete(False)
        return

    with state_lock:
        execution_state[script_key]["status"] = "running"
        execution_state[script_key]["logs"] = f"=== Starting execution of {script_key} ===\n"

    try:
        # Run with unbuffered python output (-u) in the root directory
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        process = subprocess.Popen(
            cmd,
            cwd=str(ROOT_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            bufsize=1
        )
        
        with state_lock:
            execution_state[script_key]["pid"] = process.pid

        # Read output line by line in real time
        for line in iter(process.stdout.readline, ''):
            with state_lock:
                execution_state[script_key]["logs"] += line

        process.stdout.close()
        return_code = process.wait()

        with state_lock:
            execution_state[script_key]["pid"] = None
            if return_code == 0:
                execution_state[script_key]["status"] = "success"
                execution_state[script_key]["logs"] += f"\n=== Execution completed successfully (Exit Code 0) ===\n"
                success = True
            else:
                execution_state[script_key]["status"] = "error"
                execution_state[script_key]["logs"] += f"\n=== Execution failed with exit code {return_code} ===\n"
                success = False

        if on_complete:
            on_complete(success)

    except Exception as e:
        with state_lock:
            execution_state[script_key]["status"] = "error"
            execution_state[script_key]["logs"] += f"\nException during execution: {e}\n"
            execution_state[script_key]["pid"] = None
        if on_complete:
            on_complete(False)

def run_pipeline_thread():
    """Runs the complete data flow pipeline sequentially."""
    steps = ["pv_cleaner", "con_cleaner", "battery_optimizer"]
    
    with state_lock:
        execution_state["pipeline"]["status"] = "running"
        execution_state["pipeline"]["logs"] = "=== Starting Complete Pipeline Optimization ===\n"
        execution_state["pipeline"]["current_step"] = steps[0]

    def log_to_pipeline(msg):
        with state_lock:
            execution_state["pipeline"]["logs"] += msg

    for i, step in enumerate(steps):
        with state_lock:
            execution_state["pipeline"]["current_step"] = step
            
        log_to_pipeline(f"\n[Step {i+1}/{len(steps)}] Running {step}...\n")
        
        # We run the script synchronously within this pipeline thread
        cmd = SCRIPT_COMMANDS[step]
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        try:
            process = subprocess.Popen(
                cmd,
                cwd=str(ROOT_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
                bufsize=1
            )
            
            with state_lock:
                execution_state["pipeline"]["pid"] = process.pid

            for line in iter(process.stdout.readline, ''):
                log_to_pipeline(f"[{step}] {line}")
                
            process.stdout.close()
            return_code = process.wait()
            
            if return_code != 0:
                log_to_pipeline(f"\n[ERROR] Pipeline aborted. {step} failed with exit code {return_code}.\n")
                with state_lock:
                    execution_state["pipeline"]["status"] = "error"
                    execution_state["pipeline"]["pid"] = None
                return
                
        except Exception as e:
            log_to_pipeline(f"\n[ERROR] Pipeline aborted. Exception in {step}: {e}\n")
            with state_lock:
                execution_state["pipeline"]["status"] = "error"
                execution_state["pipeline"]["pid"] = None
            return

    log_to_pipeline("\n=== Pipeline Executed Successfully! ===\n")
    with state_lock:
        execution_state["pipeline"]["status"] = "success"
        execution_state["pipeline"]["current_step"] = None
        execution_state["pipeline"]["pid"] = None


class BMSDashboardHTTPHandler(BaseHTTPRequestHandler):
    """Custom HTTP Handler to serve dashboard files and handle APIs."""

    def end_headers(self):
        # Allow CORS for development
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.end_headers()

    def do_GET(self):
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path

        # --- API: GET Status ---
        if path == "/api/status":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            with state_lock:
                self.wfile.write(json.dumps(execution_state).encode('utf-8'))
            return

        # --- API: GET Logs ---
        elif path == "/api/logs":
            query = urllib.parse.parse_qs(parsed_url.query)
            script = query.get("script", [None])[0]
            
            if script in execution_state:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                with state_lock:
                    response = {
                        "status": execution_state[script]["status"],
                        "logs": execution_state[script]["logs"]
                    }
                    if "current_step" in execution_state[script]:
                        response["current_step"] = execution_state[script]["current_step"]
                self.wfile.write(json.dumps(response).encode('utf-8'))
            else:
                self.send_error(400, "Invalid script name")
            return

        # --- API: GET Forecast Metrics ---
        elif path == "/api/forecast/metrics":
            metrics_file = RESULTS_FORECAST_DIR / "forecast_metrics.json"
            if metrics_file.exists():
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                with open(metrics_file, "rb") as mf:
                    self.wfile.write(mf.read())
            else:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({}).encode('utf-8'))
            return

        # --- API: GET MPC Metrics ---
        elif path == "/api/mpc/metrics":
            query = urllib.parse.parse_qs(parsed_url.query)
            mode = query.get("mode", [None])[0]
            if mode:
                mf = RESULTS_MPC_DIR / f"mpc_metrics_{mode}.json"
                if mf.exists():
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    with open(mf, "rb") as f:
                        self.wfile.write(f.read())
                    return
            # Aggregate all modes if none or unrecognized
            all_metrics = {}
            for m in ["supermarket", "ev", "caltech_ev"]:
                mf = RESULTS_MPC_DIR / f"mpc_metrics_{m}.json"
                if mf.exists():
                    try:
                        with open(mf, "r", encoding="utf-8") as f:
                            all_metrics[m] = json.load(f)
                    except Exception:
                        pass
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(all_metrics).encode('utf-8'))
            return

        # --- API: GET RT Metrics ---
        elif path == "/api/rt/metrics":
            query = urllib.parse.parse_qs(parsed_url.query)
            mode = query.get("mode", [None])[0]
            if mode:
                mf = RESULTS_RT_DIR / f"rt_metrics_{mode}.json"
                if mf.exists():
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    with open(mf, "rb") as f:
                        self.wfile.write(f.read())
                    return
            # Aggregate all modes if none or unrecognized
            all_metrics = {}
            for m in ["supermarket", "ev", "caltech_ev"]:
                mf = RESULTS_RT_DIR / f"rt_metrics_{m}.json"
                if mf.exists():
                    try:
                        with open(mf, "r", encoding="utf-8") as f:
                            all_metrics[m] = json.load(f)
                    except Exception:
                        pass
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(all_metrics).encode('utf-8'))
            return

        # --- API: Serve Plots ---
        elif path.startswith("/api/plots/"):
            plot_name = path.replace("/api/plots/", "")
            # Sanitize plot name to prevent traversal attacks
            plot_name = os.path.basename(plot_name)
            
            # Map of supported plots
            plot_files = {
                "data_cleaner_results.png": SRC_DIR / "data_cleaner_results.png",
                "data_cleaner_con_results.png": SRC_DIR / "data_cleaner_con_results.png",
                "optimization_results.png": SRC_DIR / "optimization_results.png",
                "optimization_results_ev.png": SRC_DIR / "optimization_results_ev.png",
                "optimization_results_caltech.png": SRC_DIR / "optimization_results_caltech.png",
                "forecast_pv.png": RESULTS_FORECAST_DIR / "forecast_pv.png",
                "forecast_con.png": RESULTS_FORECAST_DIR / "forecast_con.png",
                "forecast_ev.png": RESULTS_FORECAST_DIR / "forecast_ev.png",
                "forecast_caltech.png": RESULTS_FORECAST_DIR / "forecast_caltech.png",
                "forecast_scorecard.png": RESULTS_FORECAST_DIR / "forecast_scorecard.png",
                "mpc_schedule_supermarket.png": RESULTS_MPC_DIR / "mpc_schedule_supermarket.png",
                "mpc_schedule_ev.png": RESULTS_MPC_DIR / "mpc_schedule_ev.png",
                "mpc_schedule_caltech_ev.png": RESULTS_MPC_DIR / "mpc_schedule_caltech_ev.png",
                "rt_schedule_supermarket.png": RESULTS_RT_DIR / "rt_schedule_supermarket.png",
                "rt_schedule_ev.png": RESULTS_RT_DIR / "rt_schedule_ev.png",
                "rt_schedule_caltech_ev.png": RESULTS_RT_DIR / "rt_schedule_caltech_ev.png"
            }
            
            file_path = plot_files.get(plot_name)
            if file_path and file_path.exists():
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.end_headers()
                with open(file_path, 'rb') as img_f:
                    self.wfile.write(img_f.read())
            else:
                self.send_error(404, "Plot image not found")
            return

        # --- Serve Static UI Files ---
        else:
            # Clean path to map to index.html if request is root
            if path == "/" or path == "":
                ui_file = UI_DIR / "index.html"
            else:
                # Remove leading slash and construct file path
                ui_file = UI_DIR / path.lstrip("/")

            # Verify it's within UI_DIR to prevent directory traversal
            is_relative = False
            try:
                ui_file.resolve().relative_to(UI_DIR.resolve())
                is_relative = True
            except ValueError:
                pass

            if ui_file.exists() and is_relative:
                content_types = {
                    ".html": "text/html",
                    ".css": "text/css",
                    ".js": "application/javascript",
                    ".png": "image/png",
                    ".svg": "image/svg+xml",
                    ".json": "application/json"
                }
                suffix = ui_file.suffix.lower()
                content_type = content_types.get(suffix, "text/plain")

                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.end_headers()
                with open(ui_file, 'rb') as f:
                    self.wfile.write(f.read())
            else:
                self.send_error(404, f"File not found: {path}")

    def do_POST(self):
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path

        # --- API: Run Script ---
        if path == "/api/run":
            length = int(self.headers.get('content-length', 0))
            body = self.rfile.read(length).decode('utf-8')
            try:
                data = json.loads(body)
            except json.JSONDecodeError:
                self.send_error(400, "Bad JSON format")
                return

            script = data.get("script")
            
            if script == "pipeline":
                # Check if already running
                with state_lock:
                    is_running = any(execution_state[k]["status"] == "running" for k in execution_state)
                
                if is_running:
                    self.send_response(400)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": "A script or pipeline is already running."}).encode('utf-8'))
                    return
                
                # Start pipeline in a separate thread
                t = threading.Thread(target=run_pipeline_thread)
                t.daemon = True
                t.start()
                
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"message": "Pipeline started"}).encode('utf-8'))
                
            elif script in SCRIPT_COMMANDS:
                # Check if already running
                with state_lock:
                    is_running = any(execution_state[k]["status"] == "running" for k in execution_state)
                
                if is_running:
                    self.send_response(400)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": "Another script is already running."}).encode('utf-8'))
                    return
                
                # Start script execution in a background thread
                t = threading.Thread(target=run_script_thread, args=(script,))
                t.daemon = True
                t.start()
                
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"message": f"Script {script} started"}).encode('utf-8'))
            else:
                self.send_error(400, f"Unsupported script: {script}")
            return
        else:
            self.send_error(404)

def run_server(port=8050):
    UI_DIR.mkdir(parents=True, exist_ok=True)
    server_address = ('', port)
    httpd = HTTPServer(server_address, BMSDashboardHTTPHandler)
    print(f"\n========================================================")
    print(f"[INFO] BMS Brain Dashboard server started successfully!")
    print(f"[INFO] Access URL: http://localhost:{port}/")
    print(f"========================================================\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping dashboard server...")
        httpd.server_close()

if __name__ == "__main__":
    # Allow port override via command line arguments
    port = 8050
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            pass
    run_server(port)
