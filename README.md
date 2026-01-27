# ComputeLocator

## Deployment

To deploy the Metric Collector in your Kubernetes cluster, apply the provided YAML configuration files using kubectl:

```bash
kubectl apply -f kubernetes/containers.yaml
kubectl apply -f kubernetes/observer.yaml
```



# How to run 
start kube env
```shell
minikube start
```

Delete commands 
```shell
kubectl delete all --all
```

build images :
```shell
clear && docker build -t client-gateway ./kubernetes/load-balancer/code/ && minikube image load client-gateway &&  docker build -t cluster_api ./kubernetes/service_registry/custom_aggregation_api/ && minikube image load cluster_api && docker build -t sample_api ./kubernetes/target_apis/api_code/ &&  minikube image load sample_api
```

Deploy cluster
```shell
kubectl apply -f ./kubernetes/target_apis/ && kubectl apply -f ./kubernetes/service_registry/ && kubectl get pods
```

Init data:
```shell
kubectl apply -f ./kubernetes/load-balancer/
```

Forwarding ports to test cluster on the host
```shell
kubectl port-forward deployment/prometheus-server 5000:5000
kubectl port-forward deployment/python-api-large 8000:8000
```

Récupérer les données formatées :
```shell
curl http://localhost:5000/history
```

envoyer des appels à l'api de test :
```shell
curl --request POST --url http://localhost:8000/ml --header 'content-type: application/json'   --data '{"input":15}'
```
    
complete commands sequence
````shell
kubectl apply -f ./kubernetes/target_apis/ && kubectl apply -f ./kubernetes/service_registry/ && kubectl get pods
kubectl apply -f ./kubernetes/load-balancer/
kubectl port-forward deployment/prometheus-server 5000:5000
````


# Architecture 
![img.png](project_resources/img.png)