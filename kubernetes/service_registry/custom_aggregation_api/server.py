import json
import time
from typing import Tuple, Any
import requests
from flask import Flask
import pandas as pd

app = Flask(__name__)

class PrometheusClient:
    def __init__(self, base_url="http://localhost:9090"): # or localhost if port forward prometheus-service
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
        hidden_list = ["flask_http_request_duration_seconds_bucket",
                       "python_gc_collections_total",
                       'python_gc_objects_collected_total',
                       'python_gc_objects_uncollectable_total',
                       'scrape_duration_seconds',
                       'scrape_samples_post_metric_relabeling',
                       'scrape_samples_scraped',
                       'scrape_series_added',
                       ]
        formatted_data = [{  "ip": series["metric"]["pod_ip"],
                            "metric_name": series["metric"]["__name__"],
                            "timestamp": pd.to_datetime(float(ts), unit="s"),
                            "value": float(value),}
                            for series in data["result"] for ts, value in series["values"] if series["metric"]["__name__"] not in hidden_list]

        return (pd.DataFrame(formatted_data).sort_values("timestamp").groupby("metric_name")["value"].apply(list).to_dict(),
                formatted_data[0]['ip'])

class LokiClient:
    def __init__(self, base_url="http://localhost:3100"): # or localhost if port forward loki
        self.base_url = base_url.rstrip("/")

    def query_range(self, query,
                    start = None, # 20h ago
                    end = None,
                    limit=1000):
        ts = int(time.time() * 1e9)
        if start is None:
            start = ts - int(24 * 3600 * 1e9) # starting 1d ago
        if end is None:
            end =   ts + int(24 * 3600 * 1e9) # until 1d later

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
    global loki_client
    final_logs = []
    for filename in loki_client.query_dinstinct_labels():
        for res in loki_client.query_range(f'{{filename="{filename}"}}'):
            if res["stream"]["detected_level"] != "info": #TODO so many manual filter because I don't know how to query properly loki
                continue
            pod_name = filename.split("/")[-1].split("_")[0]
            for log_entry in res["values"]:
                final_logs.append(format_log(pod_name, log_entry))
    return json.dumps(final_logs)

def format_log(pod_name, log_entry):
    global prometheus_client
    exec_result = json.loads(json.loads(log_entry[1])["log"][5:-1])
    end_ts = exec_result["end_timestamp_ns"] * 1e-9
    start_ts = exec_result["start_timestamp_ns"] * 1e-9
    metrics, pod_ip = prometheus_client.query_all_metrics_for_pod(pod_name, start_ts, end_ts)
    return {"pod": pod_name,
           "ip": pod_ip,
            "exec_duration_ns": exec_result["end_timestamp_ns"] - exec_result["start_timestamp_ns"],
            "input": exec_result["payload"],
           "function": exec_result["endpoint"][1:],
           "states": metrics}


@app.route('/endpoints', methods=['GET'])
def service_discovery():
    global prometheus_client
    return prometheus_client.get_label_associations()

if __name__ == '__main__':
    print("API: http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, threaded=True)

