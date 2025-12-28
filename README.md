# Lightweight Hybrid Router for RAG (Unofficial Implementation)

这是一个基于跨模态注意力机制（Cross-Modal Attention）的轻量级 RAG 检索路由系统的**独立实现**。

本项目受论文 **Adaptive-RAG** 的启发，针对单步检索场景（Single-step Retrieval）实现了稀疏检索（BM25）与稠密检索（Vector Search）的智能动态路由。我们使用轻量级网络（<0.1M 参数）实现了对不同复杂度查询的自适应分流，在保持低延迟的同时显著提升了检索准确率。

## 📂 项目结构 (Project Structure)

- `main_final.py`: 核心主程序（包含模型定义、训练循环与路由推理逻辑）。
- `download_data.py`: 自动下载 SQuAD 和 HotpotQA 数据集并进行预处理。
- `plot_results.py`: 用于绘制实验结果图表（柱状图与饼图）的脚本。
- `requirements.txt`: 项目运行所需的 Python 依赖库。
- `result_hit_rate.png`: 检索命中率性能对比图。
- `result_distribution.png`: 路由策略决策分布图。

*(注：训练生成的大模型权重文件 .pth 和数据缓存 .npz 因文件过大未上传，运行代码后会自动生成)*
