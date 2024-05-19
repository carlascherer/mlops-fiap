
### buildando aplicação
docker-compose up --build -d



### como testar
```
curl -X POST http://localhost:8080/predict?model=default_propensity -H "Content-Type: application/json" -d @propensity-model/request_sample.json
```