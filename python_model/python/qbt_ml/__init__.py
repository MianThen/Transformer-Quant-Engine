"""可选的离线模型研发工具；导入本包不会加载 PyTorch。"""

from .data.schemas import BAR_V1, FeatureSchema

__all__ = ["BAR_V1", "FeatureSchema"]
