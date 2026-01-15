#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
M7: 多模态状态编码 - 深度状态表示学习

基于CLIP、ViT、BERT等多模态融合架构设计，提供统一的状态编码能力。
融合屏幕截图、任务描述、操作历史等多模态信息。

Author: AgenticX Team
Date: 2025
"""

from loguru import logger
import math
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torchvision.transforms as transforms
from PIL import Image
import numpy as np


@dataclass
class UIElement:
    """UI元素信息"""
    element_type: str  # button, text, image, input, etc.
    coordinates: Tuple[int, int, int, int]  # x1, y1, x2, y2
    text: Optional[str] = None
    confidence: float = 1.0
    attributes: Dict[str, Any] = None

    def __post_init__(self):
        if self.attributes is None:
            self.attributes = {}


class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]


class VisionEncoder(nn.Module):
    """视觉编码器 - 基于ViT架构"""
    
    def __init__(self, 
                 image_size: int = 224,
                 patch_size: int = 16,
                 embed_dim: int = 768,
                 num_heads: int = 12,
                 num_layers: int = 6,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.1):
        super().__init__()
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # 计算patch数量
        self.num_patches = (image_size // patch_size) ** 2
        
        # Patch嵌入
        self.patch_embed = nn.Conv2d(
            in_channels=3,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        
        # 位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        
        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Transformer编码器
        encoder_layer = TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            batch_first=True
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers)
        
        # 输出投影
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        
        # UI元素检测头
        self.ui_detector = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 64)  # 检测64种UI元素类型
        )
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.logger = logger
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def encode_screenshot(self, image: Union[Image.Image, torch.Tensor]) -> torch.Tensor:
        """编码屏幕截图"""
        if isinstance(image, Image.Image):
            image = self.transform(image)
        
        if image.dim() == 3:
            image = image.unsqueeze(0)  # 添加batch维度
        
        return self.forward(image)
    
    def extract_ui_features(self, image: Union[Image.Image, torch.Tensor]) -> torch.Tensor:
        """提取UI特征"""
        # 获取视觉特征
        visual_features = self.encode_screenshot(image)
        
        # 提取UI相关特征
        ui_features = self.ui_detector(visual_features)
        
        return ui_features
    
    def detect_ui_elements(self, image: Union[Image.Image, torch.Tensor]) -> List[UIElement]:
        """检测UI元素"""
        ui_features = self.extract_ui_features(image)
        
        # 简化的UI元素检测（实际应用中需要更复杂的检测算法）
        elements = []
        
        # 基于特征激活检测UI元素
        activations = torch.softmax(ui_features, dim=-1)
        top_activations = torch.topk(activations, k=5, dim=-1)
        
        for i, (score, idx) in enumerate(zip(top_activations.values[0], top_activations.indices[0])):
            if score > 0.5:  # 置信度阈值
                # 模拟UI元素位置（实际应用中需要更精确的定位）
                x1 = int(torch.rand(1) * 200)
                y1 = int(torch.rand(1) * 200)
                x2 = x1 + int(torch.rand(1) * 100 + 50)
                y2 = y1 + int(torch.rand(1) * 50 + 20)
                
                element = UIElement(
                    element_type=f"ui_type_{idx.item()}",
                    coordinates=(x1, y1, x2, y2),
                    confidence=score.item()
                )
                elements.append(element)
        
        return elements
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        B = x.shape[0]
        
        # Patch嵌入
        x = self.patch_embed(x)  # (B, embed_dim, H//patch_size, W//patch_size)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        
        # 添加CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # 添加位置编码
        x = x + self.pos_embed
        
        # Transformer编码
        x = self.transformer(x)
        
        # 取CLS token作为全局特征
        cls_feature = x[:, 0]
        
        # 输出投影
        output = self.output_proj(cls_feature)
        
        return output


class TextEncoder(nn.Module):
    """文本编码器 - 基于BERT架构"""
    
    def __init__(self,
                 vocab_size: int = 30000,
                 embed_dim: int = 768,
                 num_heads: int = 12,
                 num_layers: int = 6,
                 max_length: int = 512,
                 dropout: float = 0.1):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.max_length = max_length
        
        # 词嵌入
        self.word_embed = nn.Embedding(vocab_size, embed_dim)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(embed_dim, max_length)
        
        # Transformer编码器
        encoder_layer = TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers)
        
        # 输出投影
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        
        # 语义特征提取器
        self.semantic_extractor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.ReLU(),
            nn.Linear(embed_dim // 4, embed_dim)
        )
        
        self.logger = logger
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight, std=0.02)
    
    def encode_task_description(self, text: str) -> torch.Tensor:
        """编码任务描述"""
        # 简化的文本tokenization（实际应用中应使用专业的tokenizer）
        tokens = self._tokenize(text)
        token_ids = torch.tensor([tokens], dtype=torch.long)
        
        return self.forward(token_ids)
    
    def encode_action_sequence(self, actions: List[str]) -> torch.Tensor:
        """编码动作序列文本"""
        # 将动作序列转换为文本
        action_text = " ".join(actions)
        return self.encode_task_description(action_text)
    
    def extract_semantic_features(self, text: str) -> torch.Tensor:
        """提取语义特征"""
        # 获取文本编码
        text_encoding = self.encode_task_description(text)
        
        # 提取语义特征
        semantic_features = self.semantic_extractor(text_encoding)
        
        return semantic_features
    
    def _tokenize(self, text: str) -> List[int]:
        """简化的文本tokenization"""
        # 这里使用简化的tokenization，实际应用中应使用BERT tokenizer
        words = text.lower().split()
        
        # 简单的词汇映射
        vocab = {
            "<pad>": 0, "<unk>": 1, "<cls>": 2, "<sep>": 3,
            "click": 4, "tap": 5, "swipe": 6, "scroll": 7,
            "input": 8, "type": 9, "search": 10, "open": 11,
            "close": 12, "back": 13, "home": 14, "menu": 15
        }
        
        tokens = [vocab.get("<cls>", 2)]  # 开始token
        
        for word in words[:self.max_length - 2]:  # 保留空间给特殊token
            token_id = vocab.get(word, vocab["<unk>"])
            tokens.append(token_id)
        
        tokens.append(vocab.get("<sep>", 3))  # 结束token
        
        # 填充到固定长度
        while len(tokens) < self.max_length:
            tokens.append(vocab["<pad>"])
        
        return tokens[:self.max_length]
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 词嵌入
        x = self.word_embed(token_ids)
        
        # 位置编码
        x = self.pos_encoding(x.transpose(0, 1)).transpose(0, 1)
        
        # Transformer编码
        x = self.transformer(x)
        
        # 取第一个token（CLS）作为句子表示
        sentence_repr = x[:, 0]
        
        # 输出投影
        output = self.output_proj(sentence_repr)
        
        return output


class ActionHistoryEncoder(nn.Module):
    """动作历史编码器"""
    
    def __init__(self,
                 action_dim: int = 5,  # 动作向量维度
                 embed_dim: int = 768,
                 num_heads: int = 8,
                 num_layers: int = 4,
                 max_history_length: int = 50,
                 dropout: float = 0.1):
        super().__init__()
        
        self.action_dim = action_dim
        self.embed_dim = embed_dim
        self.max_history_length = max_history_length
        
        # 动作嵌入
        self.action_embed = nn.Linear(action_dim, embed_dim)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(embed_dim, max_history_length)
        
        # Transformer编码器
        encoder_layer = TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers)
        
        # 时序模式提取器
        self.temporal_extractor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, embed_dim)
        )
        
        # 输出投影
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        
        self.logger = logger
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def encode_action_sequence(self, actions: torch.Tensor) -> torch.Tensor:
        """编码动作序列"""
        return self.forward(actions)
    
    def extract_temporal_patterns(self, actions: torch.Tensor) -> torch.Tensor:
        """提取时序模式"""
        # 获取动作序列编码
        action_encoding = self.forward(actions)
        
        # 提取时序模式
        temporal_patterns = self.temporal_extractor(action_encoding)
        
        return temporal_patterns
    
    def compute_action_embeddings(self, actions: torch.Tensor) -> torch.Tensor:
        """计算动作嵌入"""
        # 动作嵌入
        action_embeds = self.action_embed(actions)
        
        return action_embeds
    
    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # actions shape: (batch_size, sequence_length, action_dim)
        
        # 动作嵌入
        x = self.action_embed(actions)
        
        # 位置编码
        x = self.pos_encoding(x.transpose(0, 1)).transpose(0, 1)
        
        # Transformer编码
        x = self.transformer(x)
        
        # 全局池化（平均池化）
        sequence_repr = torch.mean(x, dim=1)
        
        # 输出投影
        output = self.output_proj(sequence_repr)
        
        return output


class MultimodalStateEncoder(nn.Module):
    """多模态状态编码器主类"""
    
    def __init__(self,
                 vision_config: Optional[Dict] = None,
                 text_config: Optional[Dict] = None,
                 action_config: Optional[Dict] = None,
                 fusion_dim: int = 768,
                 num_fusion_layers: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        
        self.fusion_dim = fusion_dim
        
        # 初始化各模态编码器
        self.vision_encoder = VisionEncoder(**(vision_config or {}))
        self.text_encoder = TextEncoder(**(text_config or {}))
        self.action_encoder = ActionHistoryEncoder(**(action_config or {}))
        
        # 模态对齐层
        self.vision_proj = nn.Linear(self.vision_encoder.embed_dim, fusion_dim)
        self.text_proj = nn.Linear(self.text_encoder.embed_dim, fusion_dim)
        self.action_proj = nn.Linear(self.action_encoder.embed_dim, fusion_dim)
        
        # 多模态融合网络
        self.fusion_network = self._build_fusion_network(fusion_dim, num_fusion_layers, dropout)
        
        # 注意力融合
        self.attention_fusion = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim)
        )
        
        self.logger = logger
        
        # 初始化权重
        self._init_weights()
    
    def _build_fusion_network(self, dim: int, num_layers: int, dropout: float) -> nn.Module:
        """构建融合网络"""
        layers = []
        for i in range(num_layers):
            layers.extend([
                nn.Linear(dim, dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def encode(self, 
               screenshot: Union[Image.Image, torch.Tensor],
               task_text: str,
               action_history: torch.Tensor,
               agent_context: Optional[Dict] = None) -> torch.Tensor:
        """编码多模态状态"""
        # 各模态编码
        vision_features = self.vision_encoder.encode_screenshot(screenshot)
        text_features = self.text_encoder.encode_task_description(task_text)
        action_features = self.action_encoder.encode_action_sequence(action_history)
        
        # 模态对齐
        vision_aligned = self.vision_proj(vision_features)
        text_aligned = self.text_proj(text_features)
        action_aligned = self.action_proj(action_features)
        
        # 多模态融合
        fused_features = self.fuse_features(vision_aligned, text_aligned, action_aligned)
        
        return fused_features
    
    def fuse_features(self, 
                     vision: torch.Tensor, 
                     text: torch.Tensor, 
                     action: torch.Tensor) -> torch.Tensor:
        """融合多模态特征"""
        # 堆叠特征
        features = torch.stack([vision, text, action], dim=1)  # (batch, 3, fusion_dim)
        
        # 注意力融合
        fused_features, attention_weights = self.attention_fusion(
            features, features, features
        )
        
        # 全局池化
        global_features = torch.mean(fused_features, dim=1)
        
        # 通过融合网络
        enhanced_features = self.fusion_network(global_features)
        
        return enhanced_features
    
    def attention_fusion(self, features: List[torch.Tensor]) -> torch.Tensor:
        """基于注意力机制的特征融合"""
        # 将特征列表转换为张量
        stacked_features = torch.stack(features, dim=1)
        
        # 多头注意力
        attended_features, _ = self.attention_fusion(
            stacked_features, stacked_features, stacked_features
        )
        
        # 加权平均
        fused = torch.mean(attended_features, dim=1)
        
        return fused
    
    def get_state_embedding_dim(self) -> int:
        """获取状态嵌入维度"""
        return self.fusion_dim
    
    def forward(self, inputs: Dict[str, Any]) -> torch.Tensor:
        """神经网络前向传播"""
        screenshot = inputs.get("screenshot")
        task_text = inputs.get("task_text", "")
        action_history = inputs.get("action_history")
        agent_context = inputs.get("agent_context")
        
        if screenshot is None or action_history is None:
            raise ValueError("screenshot和action_history是必需的输入")
        
        # 编码多模态状态
        state_encoding = self.encode(screenshot, task_text, action_history, agent_context)
        
        # 输出层
        output = self.output_layer(state_encoding)
        
        return output


# 工具函数
def create_multimodal_encoder(config: Optional[Dict] = None) -> MultimodalStateEncoder:
    """创建多模态编码器"""
    default_config = {
        "vision_config": {
            "image_size": 224,
            "patch_size": 16,
            "embed_dim": 768,
            "num_heads": 12,
            "num_layers": 6
        },
        "text_config": {
            "vocab_size": 30000,
            "embed_dim": 768,
            "num_heads": 12,
            "num_layers": 6,
            "max_length": 512
        },
        "action_config": {
            "action_dim": 5,
            "embed_dim": 768,
            "num_heads": 8,
            "num_layers": 4,
            "max_history_length": 50
        },
        "fusion_dim": 768,
        "num_fusion_layers": 3,
        "dropout": 0.1
    }
    
    if config:
        # 递归更新配置
        def update_dict(d, u):
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = update_dict(d.get(k, {}), v)
                else:
                    d[k] = v
            return d
        
        default_config = update_dict(default_config, config)
    
    return MultimodalStateEncoder(**default_config)


def preprocess_image(image: Union[str, Image.Image], target_size: Tuple[int, int] = (224, 224)) -> torch.Tensor:
    """预处理图像"""
    if isinstance(image, str):
        image = Image.open(image)
    
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return transform(image)


def preprocess_action_history(actions: List[Dict], max_length: int = 50) -> torch.Tensor:
    """预处理动作历史"""
    action_tensors = []
    
    for action in actions[-max_length:]:  # 取最近的动作
        # 简化的动作编码
        action_vector = [
            action.get("type_id", 0) / 10.0,  # 动作类型ID（归一化）
            action.get("x", 0) / 1080.0,      # x坐标（归一化）
            action.get("y", 0) / 1920.0,      # y坐标（归一化）
            action.get("duration", 0) / 5.0,  # 持续时间（归一化）
            action.get("success", 0)          # 成功标志
        ]
        action_tensors.append(action_vector)
    
    # 填充到固定长度
    while len(action_tensors) < max_length:
        action_tensors.append([0.0] * 5)
    
    return torch.tensor(action_tensors[:max_length], dtype=torch.float32)