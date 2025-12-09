import torch
import numpy as np
import cv2
import re
import json
import os
from PIL import Image
from transformers import AutoProcessor, HunYuanVLForConditionalGeneration

# 全局缓存，防止每次运行都重新加载模型导致爆显存或速度慢
GLOBAL_MODEL_CACHE = {
    "model": None,
    "processor": None,
    "model_path": None
}

def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

# ----------------------------------------------------------------------
# 辅助函数区域 
# ----------------------------------------------------------------------

def clean_repeated_substrings(text):
    """Clean repeated substrings in text"""
    n = len(text)
    if n < 8000:
        return text
    for length in range(2, n // 10 + 1):
        candidate = text[-length:] 
        count = 0
        i = n - length
        while i >= 0 and text[i:i + length] == candidate:
            count += 1
            i -= length
        if count >= 10:
            return text[:n - length * (count - 1)]  
    return text

def parse_hunyuan_output(raw_text):
    """
    将模型输出的字符串解析为结构化字典。
    """
    pattern = r"(.*?)\((\d+),\s*(\d+)\),\s*\((\d+),\s*(\d+)\)"
    matches = re.findall(pattern, raw_text)
    formatted_data = {}
    
    for idx, match in enumerate(matches):
        text_content, x1, y1, x2, y2 = match
        text_content = text_content.strip()
        bbox = [int(x1), int(y1), int(x2), int(y2)]
        formatted_data[str(idx)] = {
            "text": text_content,
            "bbox": bbox
        }
    return formatted_data

def draw_mask_from_json(image_pil, formatted_data):
    """
    根据结构化的 JSON 数据绘制 Mask。
    """
    w, h = image_pil.size
    # 创建单通道黑色画布
    mask = np.zeros((h, w), dtype=np.uint8)
    
    for key, item in formatted_data.items():
        try:
            nx1, ny1, nx2, ny2 = item['bbox']
            # 坐标映射
            x1 = int((nx1 / 1000.0) * w)
            y1 = int((ny1 / 1000.0) * h)
            x2 = int((nx2 / 1000.0) * w)
            y2 = int((ny2 / 1000.0) * h)
            
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            # 绘制白色实心矩形
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        except Exception as e:
            print(f"绘制 ID {key} 出错: {e}")
            continue
            
    return mask

# ----------------------------------------------------------------------
# ComfyUI 节点类定义
# ----------------------------------------------------------------------

class HunyuanOCR_Node:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "model_path": ("STRING", {"default": "tencent/HunyuanOCR"}),
                "prompt": ("STRING", {
                    "multiline": True, 
                    "default": "检测并识别图片中的文字，将文本坐标格式化输出。"
                }),
                "max_new_tokens": ("INT", {"default": 16384, "min": 1024, "max": 32768}),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
            }
        }

    RETURN_TYPES = ("MASK", "STRING")
    RETURN_NAMES = ("mask", "json_output")
    FUNCTION = "run_ocr"
    CATEGORY = "Hunyuan OCR"

    def run_ocr(self, image, model_path, prompt, max_new_tokens, device):
        global GLOBAL_MODEL_CACHE

        # 1. 处理图像 (ComfyUI Batch Image -> List of PIL Images)
        # 这里的 image 是 [Batch, H, W, C] 的 Tensor
        batch_results_mask = []
        batch_results_json = []

        # 简单起见，如果模型路径变了，清理缓存
        if GLOBAL_MODEL_CACHE["model_path"] != model_path:
            print(f"HunyuanOCR: Loading model from {model_path}...")
            # 强制清理显存
            if GLOBAL_MODEL_CACHE["model"] is not None:
                del GLOBAL_MODEL_CACHE["model"]
                del GLOBAL_MODEL_CACHE["processor"]
                torch.cuda.empty_cache()
            
            # 加载新模型
            try:
                processor = AutoProcessor.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
                model = HunYuanVLForConditionalGeneration.from_pretrained(
                    model_path,
                    attn_implementation="eager",
                    dtype=torch.bfloat16,
                    device_map=device, # 强制指定设备
                    trust_remote_code=True
                )
                
                GLOBAL_MODEL_CACHE["model"] = model
                GLOBAL_MODEL_CACHE["processor"] = processor
                GLOBAL_MODEL_CACHE["model_path"] = model_path
            except Exception as e:
                raise RuntimeError(f"Failed to load HunyuanOCR model: {e}")

        model = GLOBAL_MODEL_CACHE["model"]
        processor = GLOBAL_MODEL_CACHE["processor"]

        # 处理 batch 中的每一张图片
        for i in range(image.shape[0]):
            img_tensor = image[i]
            # Tensor [H, W, C] -> PIL
            pil_img = tensor2pil(img_tensor.unsqueeze(0))

            # 2. 构建 Prompt
            messages = [[
                {"role": "system", "content": ""},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_img}, # Processor 可以直接处理 PIL
                        {"type": "text", "text": prompt},
                    ],
                }
            ]]

            texts = [
                processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                for msg in messages
            ]

            # 3. 预处理
            inputs = processor(
                text=texts,
                images=pil_img,
                padding=True,
                return_tensors="pt",
            )

            # 4. 推理
            with torch.no_grad():
                inputs = inputs.to(device)
                # 获取 input_ids 长度以便后续切分
                if "input_ids" in inputs:
                    input_ids = inputs.input_ids
                else:
                    input_ids = inputs.inputs
                
                generated_ids = model.generate(
                    **inputs, 
                    max_new_tokens=max_new_tokens, 
                    do_sample=False
                )

            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, generated_ids)
            ]

            raw_decoded = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

            # 5. 后处理
            cleaned_text = clean_repeated_substrings(raw_decoded)
            json_output = parse_hunyuan_output(cleaned_text)
            
            # 6. 生成 Mask
            mask_numpy = draw_mask_from_json(pil_img, json_output)
            
            # Mask Numpy (H, W) -> Tensor (1, H, W) -> Batch List
            # ComfyUI mask output needs to be float 0-1
            mask_tensor = torch.from_numpy(mask_numpy).float() / 255.0
            
            batch_results_mask.append(mask_tensor)
            
            # JSON 结果存储
            batch_results_json.append(json.dumps(json_output, indent=4, ensure_ascii=False))

        # 7. 整合输出
        # Mask: Stack to [Batch, H, W]
        final_mask = torch.stack(batch_results_mask, dim=0)
        
        # JSON: 如果 Batch > 1，返回 JSON 数组字符串，否则返回单个对象的 JSON 字符串
        if len(batch_results_json) == 1:
            final_json_str = batch_results_json[0]
        else:
            final_json_str = json.dumps([json.loads(x) for x in batch_results_json], indent=4, ensure_ascii=False)
            # json_output

        return (final_mask, final_json_str)

