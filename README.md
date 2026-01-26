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