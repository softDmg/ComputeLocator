import time

from tested_api_client import CustomApiClient
from cluster_data_client import ClusterApiClient

# get PODs
cluster_data_client = ClusterApiClient()
ENDPOINTS = cluster_data_client.getEndpoints()

ips = [e["pod_ip"] for e in ENDPOINTS]


# request test set
api_client = CustomApiClient(ips)
api_client.init_requests()

time.sleep(30) # waiting for loki and prometheus to get everything

print(cluster_data_client.getHistory())

# model training

# model testing

# plotting results

