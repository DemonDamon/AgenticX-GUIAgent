#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AgenticX-GUIAgent配置管理模块

基于AgenticX框架的配置管理系统，提供系统配置的数据模型和管理功能。

本模块已完全基于AgenticX框架重构：
- 集成AgenticX配置系统
- 支持AgenticX组件配置
- 兼容AgenticX事件系统配置
- 提供AgenticX工具系统配置

Author: AgenticX Team
Date: 2025
Version: 1.0.0 (基于AgenticX框架重构)
"""

import os
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path

from learning.learning_engine import LearningConfiguration


@dataclass
class AgenticXConfig:
    """AgenticX框架配置"""
    # 事件系统配置
    event_bus_enabled: bool = True
    event_bus_max_history: int = 1000
    event_bus_event_persistence: bool = False
    
    # 组件系统配置
    components_auto_initialize: bool = True
    components_lifecycle_management: bool = True
    components_dependency_injection: bool = True
    
    # 工具系统配置
    tools_timeout_default: float = 30.0
    tools_retry_count: int = 3
    tools_validation_enabled: bool = True
    tools_monitoring_enabled: bool = True
    
    # 内存系统配置
    memory_provider: str = "agenticx"
    memory_max_entries: int = 10000
    memory_ttl: int = 3600
    memory_embedding_model: str = "text-embedding-3-small"
    
    # 平台配置
    platform_name: str = "AgenticX-GUIAgent"
    platform_version: str = "1.0.0"
    platform_environment: str = "development"


@dataclass
class AgentAgenticXConfig:
    """智能体AgenticX配置"""
    max_iterations: int = 10
    memory_enabled: bool = True
    event_driven: bool = True
    component_based: bool = True


@dataclass
class WorkflowAgenticXConfig:
    """工作流AgenticX配置"""
    engine: str = "agenticx.core.workflow_engine.WorkflowEngine"
    execution_mode: str = "sequential"
    event_driven: bool = True
    state_management: bool = True
    error_handling: str = "retry_with_fallback"
    max_retries: int = 3


@dataclass
class NodeAgenticXConfig:
    """节点AgenticX配置"""
    timeout: int = 60
    memory_enabled: bool = True
    event_publishing: bool = True
    tool_validation: bool = False
    knowledge_persistence: bool = False


@dataclass
class EdgeAgenticXConfig:
    """边AgenticX配置"""
    event_trigger: str = ""
    data_flow: bool = True
    validation: bool = True
    result_validation: bool = False
    knowledge_extraction: bool = False
    loop_control: bool = False


@dataclass
class LLMConfig:
    """LLM配置"""
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    api_key: str = ""
    base_url: Optional[str] = None
    temperature: float = 0.3
    max_tokens: int = 4000
    timeout: int = 60
    
    def __post_init__(self):
        # Fallback logic if values are still empty
        if not self.api_key:
            if self.provider == "bailian":
                self.api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("BAILIAN_API_KEY", "")
            else:
                self.api_key = os.getenv("OPENAI_API_KEY", "")

        if not self.base_url:
            if self.provider == "openai":
                self.base_url = os.getenv("OPENAI_API_BASE")
            elif self.provider == "bailian":
                self.base_url = os.getenv("BAILIAN_API_BASE")




@dataclass
class AgentConfig:
    """智能体配置"""
    id: str
    name: str
    role: str
    goal: str
    backstory: str
    tools: List[str] = field(default_factory=list)
    learning_enabled: bool = True
    # AgenticX智能体配置
    agent_config: Optional[AgentAgenticXConfig] = None
    
    def __post_init__(self):
        if self.agent_config is None:
            self.agent_config = AgentAgenticXConfig()


@dataclass
class WorkflowNodeConfig:
    """工作流节点配置"""
    id: str
    type: str
    agent_id: str
    # AgenticX节点配置
    node_config: Optional[NodeAgenticXConfig] = None
    
    def __post_init__(self):
        if self.node_config is None:
            self.node_config = NodeAgenticXConfig()


@dataclass
class WorkflowEdgeConfig:
    """工作流边配置"""
    from_node: str
    to_node: str
    condition: Optional[str] = None
    # AgenticX边配置
    edge_config: Optional[EdgeAgenticXConfig] = None
    
    def __post_init__(self):
        if self.edge_config is None:
            self.edge_config = EdgeAgenticXConfig()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkflowEdgeConfig":
        edge_config_data = data.get("edge_config", {})
        return cls(
            from_node=data["from"],
            to_node=data["to"],
            condition=data.get("condition"),
            edge_config=EdgeAgenticXConfig(**edge_config_data) if edge_config_data else None
        )


@dataclass
class WorkflowConfig:
    """工作流配置"""
    id: str
    name: str
    nodes: List[WorkflowNodeConfig] = field(default_factory=list)
    edges: List[WorkflowEdgeConfig] = field(default_factory=list)
    # AgenticX工作流配置
    workflow_config: Optional[WorkflowAgenticXConfig] = None
    
    def __post_init__(self):
        if self.workflow_config is None:
            self.workflow_config = WorkflowAgenticXConfig()


@dataclass
class InfoPoolConfig:
    """InfoPool配置"""
    enabled: bool = True
    storage_type: str = "memory"  # memory, redis, database
    max_entries: int = 10000
    ttl: int = 3600  # 秒
    sync_interval: int = 5  # 秒
    component_config: Dict[str, Any] = field(default_factory=dict)
    storage: Dict[str, Any] = field(default_factory=dict)
    events: Dict[str, Any] = field(default_factory=dict)
    performance: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LearningStageConfig:
    """学习阶段配置"""
    enabled: bool = True
    config: Dict[str, Any] = field(default_factory=dict)


LearningConfig = LearningConfiguration


@dataclass
class MobileConfig:
    """移动设备配置"""
    platform: str = "android"  # android, ios
    device_id: str = "auto"  # 自动检测或指定设备ID
    appium_server: str = "http://localhost:4723"
    screenshot_dir: str = "./screenshots"
    max_wait_time: int = 30
    implicit_wait: int = 10
    adapter_config: Dict[str, Any] = field(default_factory=dict)
    connection: Dict[str, Any] = field(default_factory=dict)
    operations: Dict[str, Any] = field(default_factory=dict)
    tool_manager: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MonitoringConfig:
    """监控配置"""
    enabled: bool = True
    metrics: Dict[str, Any] = field(default_factory=dict)
    log_level: str = "INFO"
    log_file: str = "./logs/agenticx-guiagent.log"
    agenticx_observability: Dict[str, Any] = field(default_factory=dict)
    logging: Dict[str, Any] = field(default_factory=dict)
    alerts: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationConfig:
    """评估配置"""
    enabled: bool = True
    benchmark_suite: str = "mobile_agent_v3"
    metrics: Dict[str, Any] = field(default_factory=dict)
    report_interval: int = 100
    agenticx_evaluation: Dict[str, Any] = field(default_factory=dict)
    benchmarks: Dict[str, Any] = field(default_factory=dict)
    reporting: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgenticXGUIAgentConfig:
    """AgenticX-GUIAgent主配置类 - 基于AgenticX框架"""
    # AgenticX框架配置
    agenticx: AgenticXConfig = field(default_factory=AgenticXConfig)
    # 原有配置
    llm: LLMConfig = field(default_factory=LLMConfig)
    agents: List[AgentConfig] = field(default_factory=list)
    workflows: List[WorkflowConfig] = field(default_factory=list)
    info_pool: InfoPoolConfig = field(default_factory=InfoPoolConfig)
    learning: LearningConfig = field(default_factory=LearningConfig)
    mobile: MobileConfig = field(default_factory=MobileConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgenticXGUIAgentConfig":
        """从字典创建配置对象"""
        config = cls()
        
        # AgenticX配置
        if "agenticx" in data:
            agenticx_data = data["agenticx"]
            # 扁平化嵌套配置
            flat_config = {}
            for section, section_data in agenticx_data.items():
                if isinstance(section_data, dict):
                    for key, value in section_data.items():
                        flat_config[f"{section}_{key}"] = value
                else:
                    flat_config[section] = section_data
            config.agenticx = AgenticXConfig(**flat_config)
        
        # LLM配置
        if "llm" in data:
            llm_data = data["llm"]
            config.llm = LLMConfig(**llm_data)
        
        # 智能体配置
        if "agents" in data:
            agents = []
            for agent_data in data["agents"]:
                # 处理AgenticX agent_config
                agent_config_data = agent_data.pop("agent_config", {})
                agent = AgentConfig(**agent_data)
                if agent_config_data:
                    agent.agent_config = AgentAgenticXConfig(**agent_config_data)
                agents.append(agent)
            config.agents = agents
        
        # 工作流配置
        if "workflows" in data:
            workflows = []
            for workflow_data in data["workflows"]:
                # 处理AgenticX workflow_config
                workflow_config_data = workflow_data.get("workflow_config", {})
                
                # 处理节点配置
                nodes = []
                for node_data in workflow_data.get("nodes", []):
                    node_config_data = node_data.pop("node_config", {})
                    node = WorkflowNodeConfig(**node_data)
                    if node_config_data:
                        node.node_config = NodeAgenticXConfig(**node_config_data)
                    nodes.append(node)
                
                # 处理边配置
                edges = [
                    WorkflowEdgeConfig.from_dict(edge_data)
                    for edge_data in workflow_data.get("edges", [])
                ]
                
                # 创建工作流配置
                workflow = WorkflowConfig(
                    id=workflow_data["id"],
                    name=workflow_data["name"],
                    nodes=nodes,
                    edges=edges
                )
                if workflow_config_data:
                    workflow.workflow_config = WorkflowAgenticXConfig(**workflow_config_data)
                workflows.append(workflow)
            config.workflows = workflows
        
        # InfoPool配置
        if "info_pool" in data:
            config.info_pool = InfoPoolConfig(**data["info_pool"])
        
        # 学习配置
        if "learning" in data:
            learning_data = data["learning"]
            config.learning = LearningConfig(
                auto_learning_enabled=learning_data.get("auto_learning_enabled", True),
                learning_mode=learning_data.get("learning_mode", "adaptive"),
                max_concurrent_sessions=learning_data.get("max_concurrent_sessions", 3),
                max_learning_tasks=learning_data.get("max_learning_tasks", 10),
                learning_triggers=learning_data.get("learning_triggers", [
                    "automatic", "event_driven", "performance_based"
                ]),
            )
        
        # 移动设备配置
        if "mobile" in data:
            config.mobile = MobileConfig(**data["mobile"])
        
        # 监控配置
        if "monitoring" in data:
            config.monitoring = MonitoringConfig(**data["monitoring"])
        
        # 评估配置
        if "evaluation" in data:
            config.evaluation = EvaluationConfig(**data["evaluation"])
        
        return config
    
    def get_agent_config(self, agent_id: str) -> Optional[AgentConfig]:
        """获取指定智能体的配置"""
        for agent in self.agents:
            if agent.id == agent_id:
                return agent
        return None
    
    def get_workflow_config(self, workflow_id: str) -> Optional[WorkflowConfig]:
        """获取指定工作流的配置"""
        for workflow in self.workflows:
            if workflow.id == workflow_id:
                return workflow
        return None
    
    def validate(self) -> bool:
        """验证配置的有效性"""
        # 验证LLM配置
        if not self.llm.api_key:
            raise ValueError("LLM API密钥未配置")
        
        # 验证智能体配置
        if not self.agents:
            raise ValueError("至少需要配置一个智能体")
        
        agent_ids = {agent.id for agent in self.agents}
        
        # 验证工作流配置
        for workflow in self.workflows:
            for node in workflow.nodes:
                if node.agent_id not in agent_ids:
                    raise ValueError(f"工作流 {workflow.id} 中引用了不存在的智能体: {node.agent_id}")
        
        return True
    
    def setup_directories(self) -> None:
        """创建必要的目录"""
        directories = [
            Path(self.mobile.screenshot_dir),
            Path(self.monitoring.log_file).parent,
        ]
        
        # 如果启用了先验知识，创建知识库目录
        if self.learning.enabled and self.learning.prior_knowledge.get("enabled"):
            knowledge_path = self.learning.prior_knowledge.get("knowledge_base_path")
            if knowledge_path:
                directories.append(Path(knowledge_path))
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)