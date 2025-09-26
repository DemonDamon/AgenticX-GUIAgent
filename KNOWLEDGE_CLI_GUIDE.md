# 知识库管理CLI工具使用指南

## 概述

本工具提供了便捷的命令行接口来管理AgenticX-GUIAgent的知识库，包括查看状态、查询知识、导出数据等功能。

## 安装和配置

### 1. 环境要求

```bash
# Python 3.8+
python --version

# 安装依赖
cd /path/to/agenticx-guiagent
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
# 设置API密钥（根据使用的嵌入服务）
export BAILIAN_API_KEY="your_bailian_api_key"
export MOONSHOT_API_KEY="your_moonshot_api_key"

# 可选：设置向量数据库连接
export MILVUS_HOST="localhost"
export MILVUS_PORT="19530"
export MILVUS_USER="username"
export MILVUS_PASSWORD="password"
```

## 基本使用

### 1. 查看知识库状态

```bash
# 显示知识库的基本信息和统计数据
python cli_knowledge_manager.py status
```

输出示例：
```
📊 知识库状态信息
==================================================
📚 总知识数量: 1,234
🔍 总查询次数: 5,678
💾 缓存命中率: 85.30%
⏱️ 平均查询时间: 0.156秒

⚙️ 配置信息:
   - 存储类型: file
   - 向量化: 启用
   - 检索类型: hybrid

🗄️ 向量数据库信息:
   - 类型: MilvusVectorStore
   - 状态: 已连接
```

### 2. 查询知识

```bash
# 基本查询
python cli_knowledge_manager.py query "如何使用AgenticX框架"

# 限制结果数量
python cli_knowledge_manager.py query "移动端自动化" --limit 10
```

输出示例：
```
🔍 查询知识: 如何使用AgenticX框架
==================================================
✅ 找到 3 条相关知识:

1. AgenticX框架快速入门
   类型: tutorial
   来源: documentation
   内容: AgenticX是一个强大的智能体框架，支持多模态交互...
   相似度: 0.892

2. AgenticX核心概念
   类型: concept
   来源: manual
   内容: 本文介绍AgenticX的核心概念，包括智能体、工具、工作流...
   相似度: 0.845

⏱️ 查询耗时: 0.123秒
```

### 3. 列出知识

```bash
# 列出所有知识（默认10条）
python cli_knowledge_manager.py list

# 按类型筛选
python cli_knowledge_manager.py list --type tutorial

# 指定数量
python cli_knowledge_manager.py list --limit 20
```

### 4. 导出知识

```bash
# 导出为JSON格式
python cli_knowledge_manager.py export knowledge_backup.json

# 指定格式（目前支持json）
python cli_knowledge_manager.py export knowledge_backup.json --format json
```

导出的JSON文件结构：
```json
{
  "export_time": "2025-01-09T15:30:00",
  "total_count": 1234,
  "format": "json",
  "knowledge_items": [
    {
      "id": "knowledge_001",
      "title": "AgenticX框架介绍",
      "content": "...",
      "type": "tutorial",
      "source": "documentation",
      "domain": "ai",
      "tags": ["framework", "ai"],
      "created_at": "2025-01-01T10:00:00",
      "updated_at": "2025-01-01T10:00:00",
      "status": "active",
      "metadata": {}
    }
  ]
}
```

### 5. 测试连接

```bash
# 测试向量数据库和嵌入服务连接
python cli_knowledge_manager.py test
```

### 6. 清空知识库

```bash
# 清空所有知识（需要确认）
python cli_knowledge_manager.py clear --confirm
```

⚠️ **警告**: 此操作将删除所有知识数据，请谨慎使用！

## 高级功能

### 1. 批量操作脚本

创建批量查询脚本：

```bash
#!/bin/bash
# batch_query.sh

queries=(
    "AgenticX框架使用"
    "移动端自动化测试"
    "多模态智能体"
    "知识库管理"
)

for query in "${queries[@]}"; do
    echo "查询: $query"
    python cli_knowledge_manager.py query "$query" --limit 3
    echo "---"
done
```

### 2. 定期备份脚本

```bash
#!/bin/bash
# backup_knowledge.sh

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="knowledge_backup_${DATE}.json"

echo "开始备份知识库..."
python cli_knowledge_manager.py export "$BACKUP_FILE"

if [ $? -eq 0 ]; then
    echo "备份完成: $BACKUP_FILE"
    # 可选：上传到云存储
    # aws s3 cp "$BACKUP_FILE" s3://your-bucket/backups/
else
    echo "备份失败"
    exit 1
fi
```

### 3. 监控脚本

```bash
#!/bin/bash
# monitor_knowledge.sh

while true; do
    echo "$(date): 检查知识库状态"
    python cli_knowledge_manager.py status | grep "总知识数量\|缓存命中率\|平均查询时间"
    echo "---"
    sleep 300  # 每5分钟检查一次
done
```

## 向量数据库配置

### Milvus配置

1. **安装Milvus**:

```bash
# 使用Docker安装Milvus
docker run -d --name milvus-standalone \
  -p 19530:19530 -p 9091:9091 \
  -v $(pwd)/volumes/milvus:/var/lib/milvus \
  milvusdb/milvus:latest
```

2. **配置连接**:

在`config.py`中添加Milvus配置：

```python
MILVUS_CONFIG = {
    "host": "localhost",
    "port": 19530,
    "user": "",
    "password": "",
    "collection_name": "agenticx_guiagent_knowledge",
    "dimension": 1536,
    "metric_type": "COSINE"
}
```

3. **验证连接**:

```bash
# 测试Milvus连接
python cli_knowledge_manager.py test
```

### 其他向量数据库

- **Pinecone**: 配置API密钥和环境
- **Weaviate**: 配置服务器地址和认证
- **Qdrant**: 配置本地或云端实例

## 故障排除

### 常见问题

1. **ContentType错误**:
```
type object 'ContentType' has no attribute 'TEXT'
```
解决方案：使用`ContentType.PURE_TEXT`而不是`ContentType.TEXT`

2. **向量数据库连接失败**:
```
Failed to connect to vector database
```
解决方案：
- 检查数据库服务是否运行
- 验证连接配置
- 检查网络连接

3. **嵌入服务错误**:
```
Embedding provider not available
```
解决方案：
- 检查API密钥配置
- 验证网络连接
- 检查服务配额

### 日志调试

启用详细日志：

```bash
# 设置日志级别
export LOG_LEVEL=DEBUG

# 运行CLI工具
python cli_knowledge_manager.py status
```

查看日志文件：

```bash
# 查看最新日志
tail -f logs/knowledge_manager.log

# 搜索错误
grep -i error logs/knowledge_manager.log
```

## 性能优化

### 1. 缓存优化

```python
# 在配置中调整缓存设置
CACHE_CONFIG = {
    "enabled": True,
    "ttl": 3600,  # 1小时
    "max_entries": 10000,
    "cleanup_interval": 300  # 5分钟
}
```

### 2. 批量处理

```bash
# 批量查询时使用较小的limit
python cli_knowledge_manager.py query "关键词" --limit 5

# 分批导出大量数据
python cli_knowledge_manager.py export batch1.json --limit 1000
```

### 3. 索引优化

对于Milvus等向量数据库，确保创建适当的索引：

```python
# 示例：创建IVF_FLAT索引
index_params = {
    "index_type": "IVF_FLAT",
    "metric_type": "COSINE",
    "params": {"nlist": 1024}
}
```

## 最佳实践

1. **定期备份**: 设置自动备份任务
2. **监控性能**: 定期检查查询性能和缓存命中率
3. **清理数据**: 定期清理过期或无用的知识
4. **版本控制**: 对重要配置文件进行版本控制
5. **安全性**: 保护API密钥和数据库凭据

## 扩展开发

如需添加新功能，可以扩展`KnowledgeCLI`类：

```python
class KnowledgeCLI:
    async def custom_operation(self, params):
        """自定义操作"""
        # 实现自定义逻辑
        pass
```

然后在`main()`函数中添加对应的命令解析。

## 支持和反馈

如遇到问题或需要新功能，请：

1. 查看日志文件获取详细错误信息
2. 检查配置是否正确
3. 参考本文档的故障排除部分
4. 联系开发团队获取支持

---

**注意**: 本工具仍在持续开发中，某些功能可能需要根据实际需求进行调整和完善。