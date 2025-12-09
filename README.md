Hunyuan_OCR项目https://github.com/Tencent-Hunyuan/HunyuanOCR 的ComfyUI复现

## System Requirements
- 🖥️ 操作系统：Linux
- 🐍 Python版本：3.12+（推荐）
- ⚡ CUDA版本：12.9
- 🔥 PyTorch版本：2.7.1
- 🎮 GPU：支持CUDA的NVIDIA显卡
- 🧠 GPU显存：20GB (for vLLM)
- 💾 磁盘空间：6GB

## Installation
```bash
pip install git+https://github.com/huggingface/transformers@82a06db03535c49aa987719ed0746a76093b1ec4
pip install -r requirements.txt
```

## 💬 推荐的OCR任务提示词
| 任务 | 中文提示词 | 英文提示词 |
|------|---------|---------|
| **文字检测识别** | 检测并识别图片中的文字，将文本坐标格式化输出。 | Detect and recognize text in the image, and output the text coordinates in a formatted manner. |
| **文档解析** | • 识别图片中的公式，用 LaTeX 格式表示。<br><br>• 把图中的表格解析为 HTML。<br><br>• 解析图中的图表，对于流程图使用 Mermaid 格式表示，其他图表使用 Markdown 格式表示。<br><br>• 提取文档图片中正文的所有信息用 markdown 格式表示，其中页眉、页脚部分忽略，表格用 html 格式表达，文档中公式用 latex 格式表示，按照阅读顺序组织进行解析。| • Identify the formula in the image and represent it using LaTeX format.<br><br>• Parse the table in the image into HTML.<br><br>• Parse the chart in the image; use Mermaid format for flowcharts and Markdown for other charts.<br><br>• Extract all information from the main body of the document image and represent it in markdown format, ignoring headers and footers. Tables should be expressed in HTML format, formulas in the document should be represented using LaTeX format, and the parsing should be organized according to the reading order.|
| **通用文字提取** | • 提取图中的文字。 | • Extract the text in the image. |
| **信息抽取** | • 输出 Key 的值。<br><br>• 提取图片中的: ['key1','key2', ...] 的字段内容，并按照 JSON 格式返回。<br><br>• 提取图片中的字幕。 | • Output the value of Key.<br><br>• Extract the content of the fields: ['key1','key2', ...] from the image and return it in JSON format.<br><br>• Extract the subtitles from the image. |
| **翻译** | 先提取文字，再将文字内容翻译为英文。若是文档，则其中页眉、页脚忽略。公式用latex格式表示，表格用html格式表示。 | First extract the text, then translate the text content into English. If it is a document, ignore the header and footer. Formulas should be represented in LaTeX format, and tables should be represented in HTML format. |

## 📚 引用
```
@misc{hunyuanvisionteam2025hunyuanocrtechnicalreport,
      title={HunyuanOCR Technical Report}, 
      author={Hunyuan Vision Team and Pengyuan Lyu and Xingyu Wan and Gengluo Li and Shangpin Peng and Weinong Wang and Liang Wu and Huawen Shen and Yu Zhou and Canhui Tang and Qi Yang and Qiming Peng and Bin Luo and Hower Yang and Xinsong Zhang and Jinnian Zhang and Houwen Peng and Hongming Yang and Senhao Xie and Longsha Zhou and Ge Pei and Binghong Wu and Kan Wu and Jieneng Yang and Bochao Wang and Kai Liu and Jianchen Zhu and Jie Jiang and Linus and Han Hu and Chengquan Zhang},
      year={2025},
      journal={arXiv preprint arXiv:2511.19575},
      url={https://arxiv.org/abs/2511.19575}, 
}
```

## 🙏 致谢
我们衷心感谢[HunYuanOCR](https://github.com/Tencent-Hunyuan/HunyuanOCR)、[ComfyUI](https://github.com/comfyanonymous/ComfyUI)、[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)、[MinerU](https://github.com/opendatalab/MinerU)、[MonkeyOCR](https://github.com/Yuliang-Liu/MonkeyOCR)、[DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR)、[dots.ocr](https://github.com/rednote-hilab/dots.ocr) 的作者和贡献者，感谢他们杰出的开源工作和宝贵的研究思路。

同时我们也感谢以下宝贵的开源数据集：[OminiDocBench](https://github.com/opendatalab/OmniDocBench)、[OCRBench](https://github.com/Yuliang-Liu/MultimodalOCR/tree/main/OCRBench)、[DoTA](https://github.com/liangyupu/DIMTDA)。

特别感谢vLLM和Hugging Face社区在推理部署方面所提供的即时支持。
