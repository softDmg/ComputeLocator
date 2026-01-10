from typing import List

import requests


class CustomApiClient:
    def __init__(self, ips:List[str], port=8000):
        self.endpoints = ips
        self.port = port

    def _post(self, endpoint, payload: dict[str, str|int|float]):
        r = requests.post(f"{endpoint}", json=payload)
        r.raise_for_status()

    def _init_enpoint(self, ip, endpoint, params: List[dict[str, str|int|float]]):
        for param in params:
            self._post(f"http://{ip}:{self.port}/{endpoint}", param)

    def init_ml(self, ip):
        PAYLOADS = [{"input": 1},
                     {"input": 10},
                     {"input": 100},
                     {"input": 1_000},
                     {"input": 10_000}]
        self._init_enpoint(ip, "ml", PAYLOADS)


    def init_requests(self):
        for endpoint in self.endpoints:
            self.init_ml(endpoint)