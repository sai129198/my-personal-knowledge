# Kubernetes AI 工作负载部署

> **一句话定位**：在 K8s 上高效部署、管理和扩展 AI/ML 工作负载的完整指南。
>
> #status/canonical #topic/kubernetes #topic/deployment #year/2026

---

## 1. K8s AI 生态

### 1.1 核心组件

```
┌─────────────────────────────────────────────┐
│              User/Client                     │
└─────────────┬───────────────────────────────┘
              │
┌─────────────┴───────────────────────────────┐
│           K8s Control Plane                  │
│  ┌─────────┐ ┌─────────┐ ┌───────────────┐ │
│  │ API Server│ │Scheduler│ │Controller Mgr │ │
│  └─────────┘ └─────────┘ └───────────────┘ │
└─────────────┬───────────────────────────────┘
              │
┌─────────────┴───────────────────────────────┐
│              GPU Node(s)                     │
│  ┌─────────────────────────────────────┐    │
│  │  NVIDIA GPU Operator                │    │
│  │  ┌─────────┐ ┌─────────┐           │    │
│  │  │Device Plugin│ │DCGM Exporter│    │    │
│  │  └─────────┘ └─────────┘           │    │
│  └─────────────────────────────────────┘    │
│  ┌─────────────────────────────────────┐    │
│  │  KubeRay / Training Operator        │    │
│  │  ┌─────────┐ ┌─────────┐           │    │
│  │  │RayCluster│ │PyTorchJob│          │    │
│  │  └─────────┘ └─────────┘           │    │
│  └─────────────────────────────────────┘    │
│  ┌─────────────────────────────────────┐    │
│  │  Serving (vLLM/TGI/...)             │    │
│  │  ┌─────────┐ ┌─────────┐           │    │
│  │  │Deployment│ │Service  │           │    │
│  │  └─────────┘ └─────────┘           │    │
│  └─────────────────────────────────────┘    │
└─────────────────────────────────────────────┘
```

### 1.2 关键 Operator

| Operator | 功能 | 适用场景 |
|----------|------|----------|
| **NVIDIA GPU Operator** | GPU 驱动、Device Plugin、监控 | 所有 GPU 工作负载 |
| **KubeRay** | Ray 集群管理 | 分布式训练/推理 |
| **Training Operator** | TFJob/PyTorchJob/MPIJob | 分布式训练 |
| **Kubeflow** | ML 工作流平台 | 端到端 ML 流水线 |
| **KServe** | 模型服务 | 模型推理部署 |

---

## 2. GPU 节点配置

### 2.1 NVIDIA GPU Operator 安装

```bash
# 添加 Helm repo
helm repo add nvidia https://helm.ngc.io/nvidia
helm repo update

# 安装 GPU Operator
helm install gpu-operator nvidia/gpu-operator \
    --namespace gpu-operator \
    --create-namespace \
    --set driver.enabled=true \
    --set toolkit.enabled=true

# 验证
kubectl get pods -n gpu-operator
kubectl get nodes -o json | jq '.items[].status.capacity | select(."nvidia.com/gpu")'
```

### 2.2 GPU 节点池配置

```yaml
# AWS EKS GPU 节点组
apiVersion: eksctl.io/v1alpha5
kind: ClusterConfig
metadata:
  name: ai-cluster
  region: us-west-2

managedNodeGroups:
  - name: gpu-p4
    instanceType: p4d.24xlarge  # 8x A100
    desiredCapacity: 2
    minSize: 0
    maxSize: 10
    volumeSize: 500
    iam:
      withAddonPolicies:
        ebs: true
        efs: true
    labels:
      node-type: gpu-a100
      nvidia.com/gpu.present: "true"
    taints:
      - key: nvidia.com/gpu
        value: "true"
        effect: NoSchedule
```

### 2.3 MIG (Multi-Instance GPU)

```yaml
# MIG 配置 ConfigMap
apiVersion: v1
kind: ConfigMap
metadata:
  name: mig-config
data:
  config.yaml: |
    version: v1
    sharing:
      timeSlicing:
        resources:
        - name: nvidia.com/gpu
          replicas: 4
    # 或 MIG 配置
    mig:
      strategy: mixed
      devices:
        - device-filter: [0x20B210DE]
          devices: all
          mig-enabled: true
          mig-devices:
            "1g.10gb": 7
            "2g.20gb": 3
            "3g.40gb": 2
```

---

## 3. 训练工作负载

### 3.1 PyTorch Distributed Training

```yaml
# PyTorchJob (Training Operator)
apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  name: llm-finetune
spec:
  pytorchReplicaSpecs:
    Master:
      replicas: 1
      restartPolicy: OnFailure
      template:
        spec:
          containers:
          - name: pytorch
            image: my-registry/llm-train:latest
            command:
            - python
            - train.py
            - --nnodes=2
            - --nproc_per_node=8
            resources:
              limits:
                nvidia.com/gpu: 8
            volumeMounts:
            - name: data
              mountPath: /data
            - name: checkpoints
              mountPath: /checkpoints
          nodeSelector:
            node-type: gpu-a100
          tolerations:
          - key: nvidia.com/gpu
            operator: Exists
            effect: NoSchedule
          volumes:
          - name: data
            persistentVolumeClaim:
              claimName: training-data
          - name: checkpoints
            persistentVolumeClaim:
              claimName: checkpoint-storage
    Worker:
      replicas: 1
      restartPolicy: OnFailure
      template:
        spec:
          containers:
          - name: pytorch
            image: my-registry/llm-train:latest
            command:
            - python
            - train.py
            - --nnodes=2
            - --nproc_per_node=8
            resources:
              limits:
                nvidia.com/gpu: 8
          nodeSelector:
            node-type: gpu-a100
```

### 3.2 KubeRay 训练

```yaml
# RayCluster 配置
apiVersion: ray.io/v1
kind: RayCluster
metadata:
  name: llm-training
spec:
  headGroupSpec:
    rayStartParams:
      dashboard-host: "0.0.0.0"
      block: "true"
    template:
      spec:
        containers:
        - name: ray-head
          image: rayproject/ray-ml:2.9.0-gpu
          resources:
            limits:
              cpu: "16"
              memory: "128Gi"
              nvidia.com/gpu: "1"
          ports:
          - containerPort: 6379
            name: gcs
          - containerPort: 8265
            name: dashboard
          - containerPort: 10001
            name: client
  workerGroupSpecs:
  - replicas: 4
    minReplicas: 2
    maxReplicas: 10
    groupName: gpu-workers
    rayStartParams:
      block: "true"
    template:
      spec:
        containers:
        - name: ray-worker
          image: rayproject/ray-ml:2.9.0-gpu
          resources:
            limits:
              cpu: "32"
              memory: "256Gi"
              nvidia.com/gpu: "4"
          env:
          - name: RAY_memory_monitor_refresh_ms
            value: "0"
```

### 3.3 训练作业管理

```python
# 使用 KubeFlow SDK 提交训练
from kfp import dsl
from kfp import compiler

@dsl.component
 def train_model(
    data_path: str,
    model_output: str,
    epochs: int = 3,
    batch_size: int = 32
):
    return dsl.ContainerSpec(
        image='my-registry/llm-train:latest',
        command=['python', 'train.py'],
        args=[
            '--data', data_path,
            '--output', model_output,
            '--epochs', str(epochs),
            '--batch-size', str(batch_size)
        ]
    )

@dsl.pipeline
 def training_pipeline():
    train_task = train_model(
        data_path='s3://my-bucket/data',
        model_output='s3://my-bucket/models'
    )
    
    # 设置 GPU 资源
    train_task.set_gpu_limit(8)
    train_task.add_node_selector_constraint('node-type', 'gpu-a100')

compiler.Compiler().compile(training_pipeline, 'pipeline.yaml')
```

---

## 4. 推理服务部署

### 4.1 vLLM on K8s

```yaml
# vLLM Deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-llama2
spec:
  replicas: 1
  selector:
    matchLabels:
      app: vllm-llama2
  template:
    metadata:
      labels:
        app: vllm-llama2
    spec:
      nodeSelector:
        node-type: gpu-a100
      tolerations:
      - key: nvidia.com/gpu
        operator: Exists
        effect: NoSchedule
      containers:
      - name: vllm
        image: vllm/vllm-openai:latest
        args:
        - --model
        - meta-llama/Llama-2-70b
        - --tensor-parallel-size
        - "8"
        - --gpu-memory-utilization
        - "0.90"
        - --max-num-seqs
        - "256"
        - --quantization
        - awq
        ports:
        - containerPort: 8000
          name: http
        resources:
          limits:
            nvidia.com/gpu: "8"
            memory: "512Gi"
            cpu: "64"
        volumeMounts:
        - name: model-cache
          mountPath: /root/.cache/huggingface
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 5
      volumes:
      - name: model-cache
        persistentVolumeClaim:
          claimName: model-cache-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: vllm-llama2
spec:
  selector:
    app: vllm-llama2
  ports:
  - port: 8000
    targetPort: 8000
  type: ClusterIP
---
# HPA for autoscaling
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: vllm-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: vllm-llama2
  minReplicas: 1
  maxReplicas: 5
  metrics:
  - type: Pods
    pods:
      metric:
        name: gpu_utilization
      target:
        type: AverageValue
        averageValue: "70"
```

### 4.2 KServe 模型服务

```yaml
# KServe InferenceService
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: llm-service
spec:
  predictor:
    model:
      modelFormat:
        name: huggingface
      storageUri: s3://my-bucket/models/llama-2-7b
      resources:
        limits:
          nvidia.com/gpu: "1"
          memory: "32Gi"
        requests:
          nvidia.com/gpu: "1"
          memory: "32Gi"
      runtime: kserve-huggingfaceserver
    nodeSelector:
      node-type: gpu
    tolerations:
    - key: nvidia.com/gpu
      operator: Exists
      effect: NoSchedule
  transformer:
    containers:
    - name: transformer
      image: my-registry/llm-transformer:latest
      resources:
        limits:
          memory: "8Gi"
          cpu: "4"
```

### 4.3 多模型服务

```yaml
# 多模型路由 (Istio)
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: model-router
spec:
  hosts:
  - model-api
  http:
  - match:
    - uri:
        prefix: /v1/chat
    route:
    - destination:
        host: vllm-llama2
      weight: 70
    - destination:
        host: vllm-mistral
      weight: 30
    timeout: 30s
    retries:
      attempts: 3
      perTryTimeout: 10s
  - match:
    - uri:
        prefix: /v1/embeddings
    route:
    - destination:
        host: embedding-service
```

---

## 5. 存储配置

### 5.1 模型存储

```yaml
# 共享模型存储 (ReadWriteMany)
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-cache-pvc
spec:
  accessModes:
  - ReadWriteMany
  resources:
    requests:
      storage: 500Gi
  storageClassName: efs-sc  # AWS EFS
  # 或 azurefile-premium (Azure)
  # 或 filestore-premium (GCP)
```

### 5.2 高性能存储

```yaml
# 本地 SSD (训练数据)
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: training-data-fast
spec:
  accessModes:
  - ReadWriteOnce
  resources:
    requests:
      storage: 2Ti
  storageClassName: premium-rwo  # 高性能 SSD
---
# 数据加载 Job
apiVersion: batch/v1
kind: Job
metadata:
  name: data-loader
spec:
  template:
    spec:
      initContainers:
      - name: download
        image: amazon/aws-cli
        command:
        - aws
        - s3
        - sync
        - s3://my-bucket/training-data/
        - /data/
        volumeMounts:
        - name: data
          mountPath: /data
      containers:
      - name: main
        image: training-image
        volumeMounts:
        - name: data
          mountPath: /data
      volumes:
      - name: data
        persistentVolumeClaim:
          claimName: training-data-fast
```

---

## 6. 网络配置

### 6.1 RDMA/InfiniBand

```yaml
# RDMA 网络配置 (AWS EFA)
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: efa-device-plugin
spec:
  selector:
    matchLabels:
      name: efa-device-plugin
  template:
    metadata:
      labels:
        name: efa-device-plugin
    spec:
      hostNetwork: true
      containers:
      - name: efa-plugin
        image: amazon/aws-efa-k8s-device-plugin:latest
        securityContext:
          allowPrivilegeEscalation: false
          capabilities:
            drop: ["ALL"]
        volumeMounts:
        - name: device-plugin
          mountPath: /var/lib/kubelet/device-plugins
      volumes:
      - name: device-plugin
        hostPath:
          path: /var/lib/kubelet/device-plugins
---
# 使用 EFA 的 Pod
apiVersion: v1
kind: Pod
spec:
  containers:
  - name: training
    resources:
      limits:
        nvidia.com/gpu: "8"
        vpc.amazonaws.com/efa: "1"  # 请求 EFA 设备
```

### 6.2 网络策略

```yaml
# 限制 Pod 间通信
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: model-server-policy
spec:
  podSelector:
    matchLabels:
      app: vllm
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: api-gateway
    ports:
    - protocol: TCP
      port: 8000
  egress:
  - to:
    - namespaceSelector:
        matchLabels:
          name: monitoring
```

---

## 7. 监控与日志

### 7.1 GPU 监控 (DCGM)

```yaml
# DCGM Exporter DaemonSet
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: dcgm-exporter
spec:
  selector:
    matchLabels:
      app: dcgm-exporter
  template:
    metadata:
      labels:
        app: dcgm-exporter
    spec:
      containers:
      - name: dcgm-exporter
        image: nvcr.io/nvidia/k8s/dcgm-exporter:3.3.0
        ports:
        - name: metrics
          containerPort: 9400
        env:
        - name: DCGM_EXPORTER_LISTEN
          value: ":9400"
        volumeMounts:
        - name: nvidia-install-dir
          mountPath: /usr/local/nvidia
          readOnly: true
      volumes:
      - name: nvidia-install-dir
        hostPath:
          path: /usr/local/nvidia
---
# ServiceMonitor for Prometheus
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: dcgm-metrics
spec:
  selector:
    matchLabels:
      app: dcgm-exporter
  endpoints:
  - port: metrics
    interval: 15s
```

### 7.2 日志收集

```yaml
# Fluent Bit 配置
apiVersion: v1
kind: ConfigMap
metadata:
  name: fluent-bit-config
data:
  fluent-bit.conf: |
    [INPUT]
        Name              tail
        Tag               vllm.*
        Path              /var/log/containers/vllm*.log
        Parser            docker
    
    [FILTER]
        Name              kubernetes
        Match             vllm.*
        Kube_URL          https://kubernetes.default.svc:443
        Kube_CA_File      /var/run/secrets/kubernetes.io/serviceaccount/ca.crt
        Kube_Token_File   /var/run/secrets/kubernetes.io/serviceaccount/token
    
    [OUTPUT]
        Name              opensearch
        Match             vllm.*
        Host              opensearch-cluster
        Port              9200
        Index             vllm-logs
        Suppress_Type_Name On
```

---

## 8. 安全加固

### 8.1 Pod 安全

```yaml
# 安全上下文
apiVersion: v1
kind: Pod
spec:
  securityContext:
    runAsNonRoot: true
    runAsUser: 1000
    fsGroup: 1000
    seccompProfile:
      type: RuntimeDefault
  containers:
  - name: model-server
    securityContext:
      allowPrivilegeEscalation: false
      readOnlyRootFilesystem: true
      capabilities:
        drop:
        - ALL
    resources:
      limits:
        nvidia.com/gpu: "1"
```

### 8.2 镜像安全

```dockerfile
# 安全镜像构建
FROM vllm/vllm-openai:latest as base

# 使用非 root 用户
RUN useradd -m -u 1000 appuser
USER appuser

# 只读根文件系统
# 在 K8s 中设置 readOnlyRootFilesystem: true

# 扫描镜像
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
    aquasec/trivy image my-model-server:latest
```

---

## 9. 最佳实践

### 9.1 资源管理

```yaml
# 资源配额
apiVersion: v1
kind: ResourceQuota
metadata:
  name: gpu-quota
spec:
  hard:
    requests.nvidia.com/gpu: "16"
    limits.nvidia.com/gpu: "16"
    requests.memory: 2Ti
    limits.memory: 2Ti
---
# LimitRange
apiVersion: v1
kind: LimitRange
metadata:
  name: gpu-limits
spec:
  limits:
  - default:
      nvidia.com/gpu: "1"
    defaultRequest:
      nvidia.com/gpu: "1"
    type: Container
```

### 9.2 调度优化

```yaml
# Pod 拓扑分布约束
apiVersion: apps/v1
kind: Deployment
spec:
  template:
    spec:
      topologySpreadConstraints:
      - maxSkew: 1
        topologyKey: kubernetes.io/hostname
        whenUnsatisfiable: DoNotSchedule
        labelSelector:
          matchLabels:
            app: vllm
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchExpressions:
                - key: app
                  operator: In
                  values:
                  - vllm
              topologyKey: kubernetes.io/hostname
```

### 9.3 检查清单

```markdown
部署前检查：
□ GPU Operator 已安装并运行
□ 节点标签和污点已配置
□ 存储类已创建
□ 网络策略已定义
□ 资源配额已设置
□ 监控和日志已配置

部署后检查：
□ Pod 状态 Running
□ GPU 已正确挂载
□ 服务可访问
□ 监控指标正常
□ 日志收集正常
```

---

## 参考资源

- [NVIDIA GPU Operator Documentation](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/overview.html)
- [KubeRay Documentation](https://docs.ray.io/en/latest/cluster/kubernetes/index.html)
- [Kubeflow Training Operator](https://github.com/kubeflow/training-operator)
- [KServe Documentation](https://kserve.github.io/website/)
- [Kubernetes GPU Scheduling](https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/)
- [AWS EKS Best Practices for ML](https://aws.github.io/aws-eks-best-practices/)
- [GKE AI/ML Best Practices](https://cloud.google.com/kubernetes-engine/docs/best-practices/ai-ml)
