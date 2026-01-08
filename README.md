# ComputeLocator

## Kubernetes Metric Collector

This script collects specific CAdvisor metrics from each node in a Kubernetes cluster using Prometheus metrics endpoint.
It retrieves the following metrics:
##### CPU Metrics
- container_cpu_usage_seconds_total (Total CPU time consumed in seconds, used for CPU usage analysis)
- container_cpu_cfs_throttled_seconds_total
- container_cpu_cfs_throttled_periods_total
- container_cpu_cfs_periods_total
##### Memory Metrics
- container_memory_usage_bytes (Total memory usage in bytes, including cache)
- container_memory_working_set_bytes (Memory currently in use, excluding cache)
- container_memory_limit_bytes (Memory limit set for the container)
- container_memory_failcnt (Number of times memory usage hit the limit)
##### Disk Metrics
- container_fs_usage_bytes (Disk usage in bytes)
- container_fs_limit_bytes (Disk limit in bytes)
- container_fs_reads_bytes_total (Total bytes read from disk)
- container_fs_writes_bytes_total (Total bytes written to disk)
##### Network Metrics
- container_network_receive_bytes_total
- container_network_transmit_bytes_total
- container_network_receive_errors_total
container_network_transmit_errors_total

source : https://deepwiki.com/grafana/cadvisor/2.2-metrics-collection-and-exposition

### Deployment

To deploy the Metric Collector in your Kubernetes cluster, apply the provided YAML configuration files using kubectl:

```bash
kubectl apply -f kubernetes/containers.yaml
kubectl apply -f kubernetes/observer.yaml
```



# How to run 
start kube env
```
minikube start
```

build image :
```
docker build -t cluster_api ./kubernetes/service_registry/custom_aggregation_api/ && minikube image load cluster_api
```

Deploy cluster
```
kubectl apply -f ./kubernetes/target_apis/ && kubectl apply -f ./kubernetes/service_registry/ && kubectl get pods
```

Delete commands 
```
kubectl delete all --all
```

forwarding ports to test cluster on the host
```
kubectl port-forward Deployment/prometheus-server 8080:8080
```

envoyer des appels à l'api de test :
```
curl --request POST   --url http://localhost:8000/ml   --header 'content-type: application/json'   --data '{"input":15}'
```

# Architecture 
![img.png](project_resources/img.png)