# AgenticX-GUIAgent Docker 部署指南

## 📋 概述

本目录包含了为 AgenticX-GUIAgent 项目配置的完整 Docker 部署方案，特别针对 NotetakerAgent 的多模态知识存储需求进行了优化。

## 🏗️ 存储架构

### 四层存储解决方案

```
┌─────────────────────────────────────────┐
│            AgenticX-GUIAgent             │
│            知识存储架构                  │
└─────────────────────────────────────────┘
                    │
        ┌───────────┼───────────┐
        │           │           │
   ┌────▼────┐ ┌───▼───┐ ┌────▼────┐
   │ 向量存储 │ │关系存储│ │ 对象存储 │
   │(Milvus) │ │(PgSQL)│ │(MinIO)  │
   └─────────┘ └───────┘ └─────────┘
```

#### 1. 向量存储层 (Vector Storage)
- **Milvus** (推荐): 企业级向量搜索引擎 - 端口 19530
- **Qdrant**: 实时向量搜索 - 端口 6333
- **Chroma**: 轻量级向量数据库 - 端口 8000
- **Weaviate**: 语义搜索引擎 - 端口 8080

#### 2. 关系存储层 (Structured Storage)
- **PostgreSQL**: 结构化数据存储 - 端口 5432
- **Redis**: 高性能缓存 - 端口 6379
- **MongoDB**: 文档型数据库 - 端口 27017

#### 3. 图存储层 (Graph Storage)
- **Neo4j**: 图数据库 - 端口 7474/7687
- **Nebula Graph**: 分布式图数据库 - 端口 9669

#### 4. 对象存储层 (Object Storage)
- **MinIO**: S3兼容对象存储 - 端口 9000/9001

## 🚀 快速开始

### 1. 环境准备

```bash
# 确保 Docker 和 Docker Compose 已安装
docker --version
docker-compose --version

# 进入 docker 目录
cd docker
```

### 2. 配置环境变量

```bash
# 复制环境变量模板
cp env.example .env

# 编辑环境变量 (可选)
vim .env
```

### 3. 启动服务

#### 启动 NotetakerAgent 核心服务
```bash
# 启动 NotetakerAgent 必需的存储服务
docker-compose up -d postgres redis milvus minio etcd

# 查看服务状态
docker-compose ps
```

#### 启动完整存储栈
```bash
# 启动所有存储服务
docker-compose up -d

# 查看所有服务状态
docker-compose ps
```

### 4. 验证服务

```bash
# 检查服务健康状态
docker-compose ps

# 查看服务日志
docker-compose logs milvus
docker-compose logs postgres
docker-compose logs redis
docker-compose logs minio
```

## 🔧 NotetakerAgent 配置

### Python 配置示例

```python
# NotetakerAgent 存储配置
STORAGE_CONFIG = {
    # 向量存储 - 用于知识嵌入和语义搜索
    "vector_storage": {
        "type": "milvus",
        "host": "localhost",
        "port": 19530,
        "collection_name": "reflection_knowledge",
        "dimension": 768  # 根据使用的嵌入模型调整
    },
    
    # 结构化存储 - 用于任务序列和元数据
    "structured_storage": {
        "type": "postgresql",
        "host": "localhost",
        "port": 5432,
        "database": "agenticx",
        "username": "postgres",
        "password": "password"
    },
    
    # 缓存存储 - 用于高频访问数据
    "cache_storage": {
        "type": "redis",
        "host": "localhost",
        "port": 6379,
        "password": "password",
        "db": 0
    },
    
    # 对象存储 - 用于截图和大文件
    "object_storage": {
        "type": "minio",
        "endpoint": "localhost:9000",
        "access_key": "minioadmin",
        "secret_key": "minioadmin",
        "bucket": "knowledge-screenshots",
        "secure": False
    }
}
```

### 连接测试代码

```python
# 测试 Milvus 连接
from pymilvus import connections, utility

try:
    connections.connect("default", host="localhost", port="19530")
    print(f"Milvus 版本: {utility.get_server_version()}")
    print("✅ Milvus 连接成功")
except Exception as e:
    print(f"❌ Milvus 连接失败: {e}")

# 测试 PostgreSQL 连接
import psycopg2

try:
    conn = psycopg2.connect(
        host="localhost",
        port=5432,
        database="agenticx",
        user="postgres",
        password="password"
    )
    print("✅ PostgreSQL 连接成功")
    conn.close()
except Exception as e:
    print(f"❌ PostgreSQL 连接失败: {e}")

# 测试 Redis 连接
import redis

try:
    r = redis.Redis(host="localhost", port=6379, password="password")
    r.ping()
    print("✅ Redis 连接成功")
except Exception as e:
    print(f"❌ Redis 连接失败: {e}")

# 测试 MinIO 连接
from minio import Minio

try:
    client = Minio(
        "localhost:9000",
        access_key="minioadmin",
        secret_key="minioadmin",
        secure=False
    )
    # 创建知识存储桶
    if not client.bucket_exists("knowledge-screenshots"):
        client.make_bucket("knowledge-screenshots")
    print("✅ MinIO 连接成功")
except Exception as e:
    print(f"❌ MinIO 连接失败: {e}")
```

## 📊 管理和监控

### Web 管理界面

- **MinIO Console**: http://localhost:9001
  - 用户名: minioadmin
  - 密码: minioadmin
  - 用途: 对象存储管理

- **Neo4j Browser**: http://localhost:7474
  - 用户名: neo4j
  - 密码: password
  - 用途: 图数据库管理

- **Grafana**: http://localhost:3000
  - 用户名: admin
  - 密码: admin
  - 用途: 监控面板

- **Prometheus**: http://localhost:9090
  - 用途: 监控指标

- **Jaeger**: http://localhost:16686
  - 用途: 分布式追踪

### 健康检查

```bash
# 检查所有服务健康状态
docker-compose ps

# 检查特定服务
curl http://localhost:9091/healthz  # Milvus
curl http://localhost:6333/health   # Qdrant
curl http://localhost:8000/api/v1/heartbeat  # Chroma
curl http://localhost:9000/minio/health/live # MinIO
```

## 🛠️ 常用操作

### 服务管理

```bash
# 启动服务
docker-compose up -d [service_name]

# 停止服务
docker-compose stop [service_name]

# 重启服务
docker-compose restart [service_name]

# 查看日志
docker-compose logs -f [service_name]

# 进入容器
docker-compose exec [service_name] bash
```

### 数据管理

```bash
# 备份数据
docker-compose exec postgres pg_dump -U postgres agenticx > backup.sql

# 清理数据
docker-compose down -v  # 警告: 会删除所有数据

# 重置特定服务
docker-compose stop milvus
docker volume rm docker_milvus_data
docker-compose up -d milvus
```

### 性能调优

```bash
# 查看资源使用
docker stats

# 调整内存限制 (在 docker-compose.yml 中)
services:
  milvus:
    deploy:
      resources:
        limits:
          memory: 4G
        reservations:
          memory: 2G
```

## 🔍 故障排除

### 常见问题

#### 1. Milvus 启动失败
```bash
# 检查依赖服务
docker-compose logs etcd
docker-compose logs minio

# 重启依赖服务
docker-compose restart etcd minio
docker-compose restart milvus
```

#### 2. 端口冲突
```bash
# 检查端口占用
lsof -i :19530  # Milvus
lsof -i :5432   # PostgreSQL
lsof -i :6379   # Redis

# 修改端口 (在 docker-compose.yml 中)
ports:
  - "19531:19530"  # 改为其他端口
```

#### 3. 数据持久化问题
```bash
# 检查数据目录权限
ls -la ./data/

# 修复权限
sudo chown -R $USER:$USER ./data/
```

#### 4. 内存不足
```bash
# 检查系统资源
free -h
docker system df

# 清理无用容器和镜像
docker system prune -a
```

## 📚 依赖安装

### Python 依赖

```bash
# 安装 NotetakerAgent 所需的 Python 包
pip install pymilvus psycopg2-binary redis minio

# 或使用 requirements.txt
echo "pymilvus>=2.4.0" >> requirements.txt
echo "psycopg2-binary>=2.9.0" >> requirements.txt
echo "redis>=5.0.0" >> requirements.txt
echo "minio>=7.0.0" >> requirements.txt
pip install -r requirements.txt
```

## 🔐 安全配置

### 生产环境建议

1. **修改默认密码**
```bash
# 编辑 .env 文件
POSTGRES_PASSWORD=your_secure_password
REDIS_PASSWORD=your_secure_password
NEO4J_PASSWORD=your_secure_password
```

2. **启用 SSL/TLS**
```yaml
# 在 docker-compose.yml 中配置 SSL
services:
  postgres:
    environment:
      POSTGRES_SSL_MODE: require
```

3. **网络隔离**
```yaml
# 创建专用网络
networks:
  agenticx-guiagent-network:
    driver: bridge
    internal: true
```

## 📈 扩展配置

### 集群部署

对于生产环境，可以考虑以下扩展：

1. **Milvus 集群模式**
2. **PostgreSQL 主从复制**
3. **Redis 集群**
4. **MinIO 分布式部署**

### 监控告警

配置 Prometheus + Grafana + AlertManager 实现完整的监控告警体系。

## 🆘 支持

如果遇到问题，请检查：

1. Docker 和 Docker Compose 版本
2. 系统资源 (内存、磁盘空间)
3. 网络连接
4. 防火墙设置
5. 服务日志

---

**🎉 现在您可以开始使用 AgenticX-GUIAgent 的完整存储解决方案了！**

NotetakerAgent 将能够利用这套企业级存储架构，实现真正的多模态知识管理和智能反思分析。