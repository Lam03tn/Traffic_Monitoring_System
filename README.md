## System Architecture
![System Architecture](assets/kien-truc-tong-quan.png)

## Setup steps
Clone projects
```bash
git clone https://github.com/Lam03tn/Traffic_Monitoring_System.git
```
### Option 1: Running with Docker

Can run docker with either created images or create images for Backend FastAPI, Frontend React, Producer, Consumer
```bash
docker-compose up -d
```

Create init for cassandra and minio with camera_list.py and init_cassandra.py

After that, you can access
```bash
localhost:80
```
to interact with system through UI or
```
localhost:28000/redoc
```
for backend API docs

To build image, use Dockerfile to create. For example
```bash
docker-compose -f .\docker-compose-producer.yaml build 
```

### Option 2: Running with Kubernetes
The folder 'k8s' has divide into services, deployments, configmaps, and secrets. 
```bash
kubectl apply -f
```
to apply serrvices, deployments, configmaps, and secrets

Run Jobs to init database. Create VPN to tranmiss RTSP into cluster and Cloudflare to expose web through port 31000

# K8S - Components
### HAProxy and MediaMTX
![haproxy-mediamtx](https://github.com/user-attachments/assets/aebb2ee5-8545-479d-9460-ff4fdee15662)

### Kafka and Zookeeper
![kafka](https://github.com/user-attachments/assets/b595cdfd-1994-49e3-a98a-525ed8038d5c)

### Web component
![web](https://github.com/user-attachments/assets/5bdbed48-c9ee-4437-b5bb-816cb64cd873)

### AI component
![ai-component](https://github.com/user-attachments/assets/76fe2a0d-de6e-4d96-bb98-b7c8d7f45a9a)

# Results - Screen

![giam-sat-video](https://github.com/user-attachments/assets/f97e3fa0-49ee-4ef7-b166-b0658a262f8a)

![them-cau-hinh (1)](https://github.com/user-attachments/assets/42c15a8b-945e-4182-96e7-12eb695ecb4e)

![chi-tiet (1)](https://github.com/user-attachments/assets/ed46230a-d7c3-4c0c-a9be-f56991bfe276)




