import json

import requests


class ClusterApiClient:
    def __init__(self, port=5000, api_name="cluster-data-api-svc"):
        self.endpoints = api_name
        self.port = port

    def _get(self, endpoint):
        r = requests.get(endpoint)
        r.raise_for_status()
        return r.json()

    def getEndpoints(self):
        return self._get(f"http://{self.endpoints}:{self.port}/endpoints")

    def getHistory(self):
        return self._get(f"http://{self.endpoints}:{self.port}/history")
