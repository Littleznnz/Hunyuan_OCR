from .HunyuanOCR import HunyuanOCR_Node

# ----------------------------------------------------------------------
# 节点映射
# ----------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "HunyuanOCR_Node": HunyuanOCR_Node
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HunyuanOCR_Node": "Hunyuan OCR Predictor"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']