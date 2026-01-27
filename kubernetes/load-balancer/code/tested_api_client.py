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

    def _init_ml(self, ip):
        PAYLOADS = [{"input": 1},
                    {"input": 2},
                    {"input": 5},
                    {"input": 9},
                    {"input": 10},
                    {"input": 50},
                    {"input": 100},
                    {"input": 1_000},
                    {"input": 10_000},
                    {"input": 100_000}
        ]
        self._init_enpoint(ip, "ml", PAYLOADS)

    def _init_dense(self, ip):
        PAYLOADS = [{}]
        self._init_enpoint(ip, "dense", PAYLOADS)

    def _init_pi(self, ip):
        PAYLOADS = [{}]
        self._init_enpoint(ip, "pi", PAYLOADS)

    def _init_video(self, ip):
        PAYLOADS = [{}]
        self._init_enpoint(ip, "video", PAYLOADS)

    def init_requests(self):
        for endpoint in self.endpoints:
            self._init_ml(endpoint)
            self._init_dense(endpoint)
            self._init_pi(endpoint)
            self._init_video(endpoint)



