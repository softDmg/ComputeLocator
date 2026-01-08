import json
import time
from typing import Tuple

import requests
from flask import Flask
import pandas as pd

app = Flask(__name__)

class PrometheusClient:
    def __init__(self, base_url="http://localhost:9090"):
        self.base_url = base_url.rstrip("/")

    def _get(self, path, params):
        r = requests.get(f"{self.base_url}{path}", params=params)
        r.raise_for_status()
        return r.json()["data"]

    def get_label_associations(self, metric: str = "process_cpu_percent"): # fixed request "process_cpu_percent" is a bad practice
        return self._get(
            "/api/v1/series",
            params={"match[]": metric},
        )

    def query_all_metrics_for_pod(self, pod_name, start, end, step=None) -> Tuple[dict, int]:
        if step is None:
            step = max(0.001, (end - start)/4) # 4 is the amount of matrics outputted minus 1
        #TODO this function doesn't take into account the "no metrics returned" edge case
        data = self._get("/api/v1/query_range",
                  params={"query": f'{{pod="{pod_name}"}}',
                          "start": start,
                          "end": end,
                          "step": step})
        hidden_list = ["flask_http_request_duration_seconds_bucket", "python_gc_collections_total", 'python_gc_objects_collected_total', 'python_gc_objects_uncollectable_total']
        formatted_data = [{  "ip": series["metric"]["pod_ip"],
                            "metric_name": series["metric"]["__name__"],
                            "timestamp": pd.to_datetime(float(ts), unit="s"),
                            "value": float(value),}
                            for series in data["result"] for ts, value in series["values"] if series["metric"]["__name__"] not in hidden_list]

        return (pd.DataFrame(formatted_data).sort_values("timestamp").groupby("metric_name")["value"].apply(list).to_dict(),
                formatted_data[0]['ip'])

class LokiClient:
    def __init__(self, base_url="http://localhost:3100"):
        self.base_url = base_url.rstrip("/")

    def query_range(self, query,
                    start = int(time.time()*1e9)-int(20 * 3600 * 1e9), # 20h ago
                    end = int(time.time()*1e9),
                    limit=1000):
        url = f"{self.base_url}/loki/api/v1/query_range"
        params = {
            "query": query,
            "start": start,
            "end": end,
            "limit": limit
        }
        r = requests.get(url, params=params)
        r.raise_for_status()
        return r.json()["data"]["result"]

    def query_dinstinct_labels(self, tag:str="filename"):
        r = requests.get(f"{self.base_url}/loki/api/v1/label/{tag}/values")
        r.raise_for_status()
        return r.json()["data"]

loki_client = LokiClient()
prometheus_client = PrometheusClient()
@app.route('/history', methods=['GET'])
def history():
    global loki_client, prometheus_client
    final_logs = []
    for filename in loki_client.query_dinstinct_labels():
        for res in loki_client.query_range(f'{{filename="{filename}"}}'):
            if res["stream"]["detected_level"] != "info":
                continue
            pod_name = filename.split("/")[-1].split("_")[0]
            for log_entry in res["values"]:
                exec_result = json.loads(json.loads(log_entry[1])["log"][5:-1])
                end_ts = exec_result["end_timestamp_ns"] * 1e-9
                start_ts = exec_result["start_timestamp_ns"] * 1e-9
                metrics, pod_ip = prometheus_client.query_all_metrics_for_pod(pod_name, start_ts, end_ts)

                final_logs.append({"pod": pod_name,
                                   "ip": pod_ip,
                                   "input": exec_result["payload"],
                                   "function": exec_result["endpoint"][1:],
                                   "states": metrics})
    return json.dumps(final_logs)

@app.route('/endpoints', methods=['GET'])
def service_discovery():
    global prometheus_client
    return prometheus_client.get_label_associations()

if __name__ == '__main__':
    print("API: http://localhost:8000")
    app.run(host='0.0.0.0', port=8080, threaded=True)