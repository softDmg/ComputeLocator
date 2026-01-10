import requests


class ClusterApiClient:
    def __init__(self, port=5000):
        self.endpoints = "cluster-data-api-svc"
        self.port = port

    def _get(self, endpoint):
        r = requests.get(endpoint)
        r.raise_for_status()
        return r.json()

    def getEndpoints(self):
        return self._get(f"http://{self.endpoints}:{self.port}/endpoints")

    def getHistory(self):
        return self._get(f"http://{self.endpoints}:{self.port}/history")
