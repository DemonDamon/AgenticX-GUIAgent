# AgenticX-GUIAgent Knowledge Management Module

基于AgenticX框架的知识管理模块，使用AgenticX的storage和retrieval组件提供完整的知识存储、检索、管理和同步功能。

## 🏗️ 架构概览

本模块采用分层架构设计，完全基于AgenticX的存储和检索生态系统：

```
┌─────────────────────────────────────────────────────────────┐
│                    Knowledge Management                     │
├─────────────────────────────────────────────────────────────┤
│  KnowledgeManager (AgenticX Component)                     │
│  ├── AgenticXKnowledgeManager (核心管理器)                 │
│  ├── KnowledgeToVectorAdapter (适配器)                     │
│  ├── EventBus (事件总线)                                   │
│  └── Memory (缓存层)                                       │
├─────────────────────────────────────────────────────────────┤
│                 AgenticX Storage Layer                     │
│  ├── StorageManager (存储管理器)                           │
│  ├── MilvusStorage (向量存储)                              │
│  ├── PostgresStorage (关系存储)                            │
│  └── RedisStorage (缓存存储)                               │
├─────────────────────────────────────────────────────────────┤
│                AgenticX Retrieval Layer                    │
│  ├── VectorRetriever (向量检索)                            │
│  ├── HybridRetriever (混合检索)                            │
│  └── RetrievalQuery/Result (检索模型)                      │
├─────────────────────────────────────────────────────────────┤
│                    Data Models                              │
│  ├── KnowledgeItem (知识项)                                │
│  ├── KnowledgeType (知识类型)                              │
│  ├── QueryRequest/Result (查询模型)                        │
│  └── SyncRequest/Result (同步模型)                         │
└─────────────────────────────────────────────────────────────┘
```

## 概述

AgenticX-GUIAgent Knowledge模块提供了基于AgenticX框架的完整知识管理和共享系统。该模块已重构以充分利用AgenticX的核心组件，避免重复实现，提供现代化的多智能体知识协作机制。

## 重构说明

### 主要变更

1. **基于AgenticX Component**: 所有核心类都继承自`agenticx.core.component.Component`
2. **事件驱动架构**: 使用`agenticx.core.event.EventBus`进行知识共享和同步
3. **内存集成**: 使用`agenticx.memory.component.Memory`进行缓存管理
4. **存储优化**: 基于AgenticX的Storage组件重构存储层
5. **避免重复实现**: 移除与AgenticX重复的基础设施代码

### 核心组件

#### 1. KnowledgeTypes (knowledge_types.py)
- **功能**: 定义知识项、查询、同步等核心数据结构
- **重构**: 基于AgenticX的数据模型，集成事件和消息系统
- **特性**: 现代化的数据结构和类型注解

#### 2. KnowledgeStore (knowledge_store.py)
- **功能**: 实现知识的持久化存储、索引管理和查询
- **基类**: 基于AgenticX的Storage组件
- **实现**: InMemoryKnowledgeStore、SQLiteKnowledgeStore
- **特性**: 高性能存储和检索、事件监控

#### 3. KnowledgeManager (knowledge_manager.py)
- **功能**: 统一的知识管理、同步、缓存和生命周期管理
- **基类**: `agenticx.core.component.Component`
- **特性**: 事件驱动管理、Memory组件集成、现代化生命周期管理

#### 4. KnowledgePool (knowledge_pool.py)
- **功能**: 多智能体间的知识管理和共享机制
- **基类**: `agenticx.core.component.Component`
- **特性**: 访问控制、推荐系统、订阅机制、知识图谱

## 使用示例

### 基本使用

```python
from agenticx.core.event_bus import EventBus
from agenticx.memory.component import Memory
from knowledge import (
    KnowledgePool,
    KnowledgeManager,
    KnowledgeItem,
    KnowledgeType,
    KnowledgeSource
)

# 创建事件总线和内存组件
event_bus = EventBus()
memory = Memory()

# 创建知识管理器
knowledge_manager = KnowledgeManager(
    event_bus=event_bus,
    memory=memory,
    enable_validation=True,
    enable_lifecycle=True
)

# 创建知识池
knowledge_pool = KnowledgePool(
    knowledge_manager=knowledge_manager,
    event_bus=event_bus,
    memory=memory,
    enable_access_control=True,
    enable_recommendations=True,
    enable_subscriptions=True
)

# 启动服务
await knowledge_manager.start()
await knowledge_pool.start()
```

### 知识贡献

```python
from knowledge import (
    KnowledgeItem,
    KnowledgeType,
    KnowledgeSource,
    AccessLevel,
    ShareScope
)

# 创建知识项
knowledge = KnowledgeItem(
    type=KnowledgeType.PROCEDURAL,
    source=KnowledgeSource.AGENT_EXPERIENCE,
    title="移动应用操作经验",
    content={
        "action": "click_button",
        "element": "login_button",
        "success_rate": 0.95,
        "context": "登录页面"
    },
    description="成功的登录按钮点击操作经验",
    keywords={"click", "login", "button", "mobile"},
    domain="mobile_gui"
)

# 贡献知识
success = await knowledge_pool.contribute_knowledge(
    knowledge=knowledge,
    contributor_id="agent_001",
    access_level=AccessLevel.PUBLIC,
    share_scope=ShareScope.GLOBAL
)

print(f"知识贡献{'成功' if success else '失败'}")
```

### 知识查询

```python
from knowledge import QueryRequest

# 创建查询请求
query_request = QueryRequest(
    query_text="登录操作",
    query_type="semantic",
    filters={
        "type": ["procedural", "experiential"],
        "domain": "mobile_gui"
    },
    sort_by="relevance",
    limit=10
)

# 执行查询
result = await knowledge_pool.query_knowledge(
    request=query_request,
    requester_id="agent_002"
)

print(f"找到 {len(result.items)} 个相关知识项")
for knowledge in result.items:
    print(f"- {knowledge.title}: {knowledge.description}")
```

### 知识订阅

```python
from knowledge import KnowledgeType

# 订阅知识更新
def on_knowledge_update(knowledge):
    print(f"收到新知识: {knowledge.title}")

subscription_id = await knowledge_pool.subscribe_knowledge(
    subscriber_id="agent_003",
    knowledge_types={KnowledgeType.PROCEDURAL, KnowledgeType.EXPERIENTIAL},
    domains={"mobile_gui"},
    keywords={"click", "swipe", "input"},
    callback=on_knowledge_update
)

print(f"订阅ID: {subscription_id}")
```

### 知识分享

```python
from knowledge import ShareScope

# 分享知识给特定智能体
success = await knowledge_pool.share_knowledge(
    knowledge_id="knowledge_123",
    sharer_id="agent_001",
    target_agents={"agent_004", "agent_005"},
    share_scope=ShareScope.AGENT
)

print(f"知识分享{'成功' if success else '失败'}")
```

### 获取推荐

```python
# 获取知识推荐
recommendations = await knowledge_pool.get_recommendations(
    agent_id="agent_002",
    limit=5
)

for rec in recommendations:
    print(f"推荐知识: {rec.knowledge_id}")
    print(f"相关性分数: {rec.relevance_score:.2f}")
    print(f"推荐理由: {rec.reason}")
    
    # 提供反馈
    await knowledge_pool.provide_feedback(
        recommendation_id=rec.id,
        feedback_score=0.8,
        accepted=True
    )
```

## 数据结构

### KnowledgeItem
知识项是系统中的基本数据单元：

```python
@dataclass
class KnowledgeItem:
    id: str
    type: KnowledgeType
    source: KnowledgeSource
    status: KnowledgeStatus
    title: str
    content: Any
    description: str
    keywords: Set[str]
    context: Dict[str, Any]
    domain: str
    metadata: KnowledgeMetadata
    # ... 其他字段
```

### KnowledgeType
知识类型枚举：
- `FACTUAL`: 事实性知识
- `PROCEDURAL`: 程序性知识
- `EXPERIENTIAL`: 经验性知识
- `CONTEXTUAL`: 上下文知识
- `PATTERN`: 模式知识
- `RULE`: 规则知识
- `STRATEGY`: 策略知识

### KnowledgeSource
知识来源枚举：
- `AGENT_EXPERIENCE`: 智能体经验
- `USER_INPUT`: 用户输入
- `LEARNING_PROCESS`: 学习过程
- `REFLECTION`: 反思过程
- `COLLABORATION`: 协作过程

## 事件系统

知识模块通过AgenticX事件系统进行通信：

### 发布的事件
- `knowledge_stored`: 知识存储完成
- `knowledge_updated`: 知识更新完成
- `knowledge_shared`: 知识分享完成
- `knowledge_queried`: 知识查询完成
- `knowledge_notification`: 订阅通知
- `recommendation_generated`: 推荐生成

### 订阅事件
```python
# 订阅知识事件
def on_knowledge_stored(event):
    data = event.data
    print(f"新知识存储: {data['knowledge_id']}")

event_bus.subscribe("knowledge_stored", on_knowledge_stored)
```

## 存储选项

### 内存存储
```python
from knowledge import InMemoryKnowledgeStore

store = InMemoryKnowledgeStore()
```

### SQLite存储
```python
from knowledge import SQLiteKnowledgeStore

store = SQLiteKnowledgeStore(db_path="knowledge.db")
```

### 工厂模式
```python
from knowledge import KnowledgeStoreFactory

# 创建内存存储
store = KnowledgeStoreFactory.create_store("memory")

# 创建SQLite存储
store = KnowledgeStoreFactory.create_store("sqlite", db_path="knowledge.db")
```

## 配置选项

### KnowledgeManager配置
```python
knowledge_manager = KnowledgeManager(
    store=custom_store,
    event_bus=event_bus,
    memory=memory,
    cache_size=2000,        # 缓存大小
    cache_ttl=7200,         # 缓存TTL（秒）
    enable_validation=True,  # 启用知识验证
    enable_lifecycle=True    # 启用生命周期管理
)
```

### KnowledgePool配置
```python
knowledge_pool = KnowledgePool(
    knowledge_manager=knowledge_manager,
    event_bus=event_bus,
    memory=memory,
    enable_access_control=True,    # 启用访问控制
    enable_recommendations=True,   # 启用推荐系统
    enable_subscriptions=True      # 启用订阅机制
)
```

## 性能优化

1. **缓存策略**: 使用LRU缓存提高查询性能
2. **索引优化**: 多维度索引支持快速检索
3. **批量操作**: 支持批量存储和查询
4. **异步处理**: 全异步设计提高并发性能
5. **事件驱动**: 减少轮询，提高响应速度

## 最佳实践

1. **知识结构化**: 使用标准化的知识结构
2. **元数据丰富**: 提供详细的元数据信息
3. **访问控制**: 合理设置知识访问权限
4. **订阅管理**: 避免过度订阅影响性能
5. **反馈机制**: 积极提供推荐反馈改善系统

## 注意事项

1. **依赖关系**: 确保AgenticX框架已正确安装
2. **资源管理**: 注意内存和存储资源的使用
3. **并发安全**: 多智能体并发访问时的线程安全
4. **数据持久化**: 重要知识应使用持久化存储
5. **事件处理**: 确保事件处理器的异常安全

## 未来规划

1. **分布式存储**: 支持分布式知识存储
2. **智能推荐**: 基于机器学习的智能推荐
3. **知识图谱**: 增强知识图谱功能
4. **版本控制**: 知识版本管理和回滚
5. **性能监控**: 实时性能监控和优化