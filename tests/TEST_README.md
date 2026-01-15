# AgenticX-GUIAgent 测试套件

基于AgenticX框架的多模态智能体系统测试文件集合。

## 📋 测试文件说明

### 核心功能测试

**1. test_hybrid_embedding.py**
- **功能**: 混合embedding性能测试和调优
- **测试内容**: 文本embedding性能、多模态embedding性能、缓存性能、成本优化、批处理性能
- **启动命令**: `python test_hybrid_embedding.py`

**2. test_multimodal_only.py**
- **功能**: 专门测试多模态embedding路由
- **测试内容**: 验证多模态内容是否正确路由到多模态embedding模型
- **启动命令**: `python test_multimodal_only.py`

### 智能体功能测试

**3. test_multimodal_executor.py**
- **功能**: ExecutorAgent多模态操作执行测试
- **测试内容**: 多模态LLM驱动的操作执行、智能坐标定位、多模型降级策略
- **启动命令**: `python test_multimodal_executor.py`

**4. test_multimodal_manager.py**
- **功能**: ManagerAgent任务分解测试
- **测试内容**: 多模态任务分解、智能子任务生成、依赖关系分析
- **启动命令**: `python test_multimodal_manager.py`

**5. test_multimodal_reflector.py**
- **功能**: ActionReflectorAgent反思分析测试
- **测试内容**: 多模态反思分析、操作前后截图对比、成功性判断
- **启动命令**: `python test_multimodal_reflector.py`

**6. test_multimodal_notetaker.py**
- **功能**: NotetakerAgent知识管理测试
- **测试内容**: 智能知识捕获、知识分类标签、知识查询检索、知识组织关联
- **启动命令**: `python test_multimodal_notetaker.py`

### 集成协作测试

**7. test_multimodal_agents_integration.py**
- **功能**: 多智能体集成协作测试
- **测试内容**: 事件驱动协作、反思到知识流程、知识查询应用、知识演化学习
- **启动命令**: `python test_multimodal_agents_integration.py`

## 🚀 快速开始

### 环境准备

1. **安装依赖**
   ```bash
   pip install agenticx python-dotenv
   ```

2. **配置API密钥**
   
   在tests目录下创建 `.env` 文件：
   ```bash
   # 阿里云百炼API密钥（必需）
   BAILIAN_API_KEY=your_bailian_api_key_here
   
   # 可选：其他API密钥（用于降级策略）
   MOONSHOT_API_KEY=your_moonshot_api_key_here
   OPENAI_API_KEY=your_openai_api_key_here
   ```

3. **创建必要目录**
   ```bash
   mkdir -p screenshots knowledge_base results
   ```

### 运行测试

进入tests目录后，可以单独运行任何测试文件：

```bash
cd tests

# 运行embedding相关测试
python test_hybrid_embedding.py
python test_multimodal_only.py

# 运行智能体测试
python test_multimodal_executor.py
python test_multimodal_manager.py
python test_multimodal_reflector.py
python test_multimodal_notetaker.py

# 运行集成测试
python test_multimodal_agents_integration.py
```

## 🔧 测试配置

### 模型配置

测试使用的主要模型：
- **主要模型**: `qwen-vl-max` (多模态LLM)
- **备用模型**: `qwen-vl-plus`, `moonshot-v1-8k`
- **Embedding模型**: `multimodal-embedding-v1`, `text-embedding-v4`

### 测试数据目录

- **screenshots/**: 测试用截图文件
- **knowledge_base/**: 知识库存储目录
- **results/**: 测试结果输出目录

## 📊 测试结果说明

### 成功指标
- ✅ 多模态LLM调用成功
- ✅ Embedding路由正确
- ✅ 智能体协作正常
- ✅ 知识管理功能正常

### 预期结果
- **理想情况**: 所有测试通过，功能正常
- **可接受情况**: 80%以上测试通过，主要功能正常
- **需要关注**: 50%以下测试通过，需要检查配置

## 🛠️ 故障排除

### 常见问题

1. **API密钥错误**
   - 错误: `未设置BAILIAN_API_KEY` 或 `401 Unauthorized`
   - 解决: 在tests目录下的.env文件中设置正确的API密钥

2. **网络连接问题**
   - 错误: `连接超时` 或 `网络错误`
   - 解决: 检查网络连接，确保可以访问百炼API服务

3. **依赖包缺失**
   - 错误: `ModuleNotFoundError`
   - 解决: `pip install agenticx python-dotenv`

4. **文件路径问题**
   - 错误: `文件不存在`
   - 解决: 确保在正确的目录下运行测试，检查相对路径

### 调试建议

- 查看测试输出的详细日志信息
- 先运行简单的embedding测试验证基础配置
- 检查.env文件是否在正确位置且格式正确
- 确保API密钥有效且有足够的调用额度

## 📝 注意事项

- 测试会消耗API调用额度，请合理使用
- 部分测试需要网络连接，确保网络稳定
- 测试结果会保存在results目录中
- 建议在测试环境中运行，避免影响生产数据