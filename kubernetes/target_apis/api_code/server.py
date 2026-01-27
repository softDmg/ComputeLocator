import logging
import os
import sys
import threading
import time
import datetime

from flask import Flask, request, g, json
from prometheus_flask_exporter import PrometheusMetrics
from prometheus_client import Gauge, Info
import psutil
from functions.ML import ml_method
from functions.pi_estimation import pi_estimation
from functions.dense_network import dense_network
from functions.video_encoding import video_encoding

app = Flask(__name__)
metrics = PrometheusMetrics(app, path=None)
logging.getLogger("werkzeug").setLevel(logging.ERROR)
metrics.start_http_server(9000)

# =============================================================================
# STRUCTURED LOGGING (individual request details)
# =============================================================================
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(levelname)s %(message)s')
handler.setFormatter(formatter)

app.logger.handlers.clear()
app.logger.addHandler(handler)
app.logger.setLevel(logging.INFO)
app.logger.propagate = False

@app.before_request
def before_request():
    g.start_time = time.time_ns()
    g.request_json = request.get_json()


@app.after_request
def after_request(response):
    end_time = time.time_ns()
    # Also log full details as JSON
    log_entry = {
        'type': "prev_request_history",
        'end_timestamp_ns': end_time,
        'start_timestamp_ns': g.start_time,
        'endpoint': request.path,
        'method': request.method,
        'payload': g.request_json,
        'status': response.status_code,
    }
    app.logger.info(json.dumps(log_entry))  #TODO could imply overhead if stdout gets bigger, a push pattern could be more efficient using loki client

    return response

# =============================================================================
# PROCESS METRICS
# =============================================================================

# CPU
PROCESS_CPU_PERCENT = Gauge('process_cpu_percent', 'Process CPU usage %')
PROCESS_CPU_USER = Gauge('process_cpu_user_seconds', 'User CPU time in seconds')
PROCESS_CPU_SYSTEM = Gauge('process_cpu_system_seconds', 'System CPU time in seconds')

# Memory
PROCESS_MEMORY_RSS = Gauge('process_memory_rss_bytes', 'Resident Set Size (physical memory)')
PROCESS_MEMORY_VMS = Gauge('process_memory_vms_bytes', 'Virtual Memory Size')
PROCESS_MEMORY_PERCENT = Gauge('process_memory_percent', 'Memory usage %')

# Threads & FDs
PROCESS_THREADS = Gauge('process_num_threads', 'Number of threads')

# I/O
PROCESS_IO_READ = Gauge('process_io_read_bytes', 'Bytes read')
PROCESS_IO_WRITE = Gauge('process_io_write_bytes', 'Bytes written')

# Context switches
PROCESS_CTX_VOLUNTARY = Gauge('process_ctx_switches_voluntary', 'Voluntary context switches')
PROCESS_CTX_INVOLUNTARY = Gauge('process_ctx_switches_involuntary', 'Involuntary context switches')

# Connections
PROCESS_CONNECTIONS = Gauge('process_connections', 'Number of network connections')

# Uptime
PROCESS_UPTIME = Gauge('process_uptime_seconds', 'Process uptime in seconds') #TODO make it nano seconds or ms

# Info
PROCESS_INFO = Info('process', 'Process information')


def collect_process_metrics():
    """Collect metrics for current Python process (like 'top')."""
    proc = psutil.Process(os.getpid())

    # Set process info once
    PROCESS_INFO.info({
        'pid': str(proc.pid),
        'name': proc.name(),
        'cmdline': ' '.join(proc.cmdline()[:3])  # First 3 args
    })

    while True:
        try:
            # CPU
            PROCESS_CPU_PERCENT.set(proc.cpu_percent())
            cpu_times = proc.cpu_times()
            PROCESS_CPU_USER.set(cpu_times.user)
            PROCESS_CPU_SYSTEM.set(cpu_times.system)

            # Memory
            mem = proc.memory_info()
            PROCESS_MEMORY_RSS.set(mem.rss)
            PROCESS_MEMORY_VMS.set(mem.vms)
            PROCESS_MEMORY_PERCENT.set(proc.memory_percent())

            # Threads
            PROCESS_THREADS.set(proc.num_threads())

            # I/O (Linux only)
            try:
                io = proc.io_counters()
                PROCESS_IO_READ.set(io.read_bytes)
                PROCESS_IO_WRITE.set(io.write_bytes)
            except (AttributeError, psutil.AccessDenied):
                pass

            # Context switches
            try:
                ctx = proc.num_ctx_switches()
                PROCESS_CTX_VOLUNTARY.set(ctx.voluntary)
                PROCESS_CTX_INVOLUNTARY.set(ctx.involuntary)
            except AttributeError:
                pass

            # Network connections
            try:
                PROCESS_CONNECTIONS.set(len(proc.net_connections()))
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                pass

            # Uptime
            PROCESS_UPTIME.set(time.time() - proc.create_time())

        except Exception as e:
            print(f"Metrics error: {e}")

        time.sleep(1)


# =============================================================================
# API ROUTES
# =============================================================================

@app.route('/ml', methods=['POST'])
def ml():
    ml_method(**request.json)
    return {"data": "endpoint OK\n"}

@app.route('/pi', methods=['POST', 'GET'])
def pi():
    if request.json:
        return pi_estimation(**request.json)
    return pi_estimation()


@app.route('/dense', methods=['POST', 'GET'])
def dense():
    if request.json:
        return dense_network(**request.json)
    return dense_network()


@app.route('/video', methods=['POST', 'GET'])
def video():
    if request.json:
        return video_encoding(**request.json)
    return video_encoding()


@app.route('/health')
def health():
    return {"status": "healthy"}


if __name__ == '__main__':
    threading.Thread(target=collect_process_metrics, daemon=True).start()
    print("API: http://0.0.0.0:8000")
    print("Metrics: http://0.0.0.0:9000/metrics")
    app.run(host='0.0.0.0', port=8000)