#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Config Loader - 配置加载器
从config.yaml加载embedding和知识管理配置

Author: AgenticX Team
Date: 2025
"""

import os
import yaml
from loguru import logger
from typing import Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv

from utils import setup_logger


class ConfigLoader:
    """配置加载器"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.logger = logger
        self._config_cache = None
        # 在初始化时加载.env文件，确保环境变量可用
        load_dotenv()
    
    def load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        if self._config_cache is not None:
            return self._config_cache
        
        config_file = Path(self.config_path)
        if not config_file.exists():
            logger.error(f"配置文件不存在: {config_file}")
            return {}
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # 替换环境变量
            config = self._replace_env_vars(config)
            
            self._config_cache = config
            logger.info(f"配置文件加载成功: {config_file}")
            return config
        
        except Exception as e:
            logger.error(f"配置文件加载失败: {e}")
            return {}
    
    def _replace_env_vars(self, obj: Any) -> Any:
        """递归替换环境变量"""
        if isinstance(obj, dict):
            return {k: self._replace_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._replace_env_vars(item) for item in obj]
        elif isinstance(obj, str) and obj.startswith('${') and obj.endswith('}'):
            # 提取环境变量名
            env_var = obj[2:-1]
            default_value = ""
            
            # 支持默认值语法: ${VAR_NAME:default_value}
            if ':' in env_var:
                env_var, default_value = env_var.split(':', 1)
            
            return os.getenv(env_var, default_value)
        else:
            return obj
    
    def get_embedding_config(self) -> Dict[str, Any]:
        """获取embedding配置"""
        config = self.load_config()
        return config.get('embedding', {})
    
    def get_knowledge_config(self) -> Dict[str, Any]:
        """获取知识管理配置"""
        config = self.load_config()
        knowledge_config = config.get('knowledge', {})
        
        # 合并embedding配置
        embedding_config = self.get_embedding_config()
        if embedding_config:
            knowledge_config['embedding_provider'] = embedding_config.get('provider', 'bailian')
            knowledge_config['embedding_config'] = embedding_config
        
        return knowledge_config
    
    def get_llm_config(self) -> Dict[str, Any]:
        """获取LLM配置"""
        config = self.load_config()
        return config.get('llm', {})
    
    def get_agenticx_config(self) -> Dict[str, Any]:
        """获取AgenticX配置"""
        config = self.load_config()
        return config.get('agenticx', {})
    
    def get_agent_config(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """获取特定智能体配置"""
        config = self.load_config()
        agents = config.get('agents', [])
        
        for agent in agents:
            if agent.get('id') == agent_id:
                return agent
        
        return None
    
    def validate_embedding_config(self) -> bool:
        """验证embedding配置"""
        embedding_config = self.get_embedding_config()
        
        if not embedding_config:
            logger.warning("未找到embedding配置")
            return False
        
        provider = embedding_config.get('provider')
        if not provider:
            logger.error("embedding配置中缺少provider")
            return False
        
        provider_config = embedding_config.get(provider, {})
        if not provider_config:
            logger.error(f"未找到{provider}提供者的配置")
            return False
        
        # 检查必要的配置项
        required_fields = ['api_key', 'model']
        for field in required_fields:
            if not provider_config.get(field):
                logger.error(f"{provider}配置中缺少{field}")
                return False
        
        logger.info(f"Embedding配置验证通过: {provider}")
        return True
    
    def get_env_template(self) -> str:
        """生成环境变量模板"""
        template = [
            "# AgenticX-GUIAgent Embedding 环境变量配置",
            "# 请根据您使用的embedding提供者设置相应的API密钥",
            "",
            "# 阿里云百炼配置",
            "BAILIAN_API_KEY=your_bailian_api_key_here",
            "BAILIAN_API_BASE=https://dashscope.aliyuncs.com",
            "BAILIAN_EMBEDDING_MODEL=text-embedding-v4",
            "",
            "# OpenAI配置 (备用)",
            "OPENAI_API_KEY=your_openai_api_key_here",
            "OPENAI_API_BASE=https://api.openai.com/v1",
            "",
            "# SiliconFlow配置 (备用)",
            "SILICONFLOW_API_KEY=your_siliconflow_api_key_here",
            "",
            "# DeepSeek配置 (备用)",
            "DEEPSEEK_API_KEY=your_deepseek_api_key_here",
            ""
        ]
        
        return "\n".join(template)
    
    def create_env_file(self, env_file_path: str = ".env") -> bool:
        """创建环境变量文件模板"""
        try:
            env_path = Path(env_file_path)
            if env_path.exists():
                logger.warning(f"环境变量文件已存在: {env_path}")
                return False
            
            with open(env_path, 'w', encoding='utf-8') as f:
                f.write(self.get_env_template())
            
            logger.info(f"环境变量模板文件创建成功: {env_path}")
            return True
        
        except Exception as e:
            logger.error(f"创建环境变量文件失败: {e}")
            return False
    
    def clear_cache(self):
        """清空配置缓存"""
        self._config_cache = None
        logger.debug("配置缓存已清空")


# 全局配置加载器实例
_config_loader = None

def get_config_loader(config_path: str = "config.yaml") -> ConfigLoader:
    """获取全局配置加载器实例"""
    global _config_loader
    if _config_loader is None:
        _config_loader = ConfigLoader(config_path)
    return _config_loader

def load_embedding_config() -> Dict[str, Any]:
    """快捷方法：加载embedding配置"""
    return get_config_loader().get_embedding_config()

def load_knowledge_config() -> Dict[str, Any]:
    """快捷方法：加载知识管理配置"""
    return get_config_loader().get_knowledge_config()

def validate_config() -> bool:
    """快捷方法：验证配置"""
    loader = get_config_loader()
    return loader.validate_embedding_config()