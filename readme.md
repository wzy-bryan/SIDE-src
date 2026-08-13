SIDE 项目（简要）

- 运行攻击/消融实验脚本：`scripts/` 或 `bash scripts/attack_side.sh`（视项目结构）。
- 生成中间层与有害方向余弦相似度：`bash attention.sh`。
- 绘图脚本：`draw_score.py`（已改为使用 `SIDE_ROOT` 或相对路径）。

快速开始:

1. 安装依赖：

```bash
pip install -r requirements.txt
```

2. 可选环境变量：

```bash
export SIDE_ROOT=/path/to/SIDE
# export HF_HOME=/path/to/hf/cache
# export WANDB_MODE=offline
# export WANDB_API_KEY=your_key_if_needed
```

3. 运行示例：

```bash
bash scripts/attack_all_model.sh
```

更多详细说明请查看各模块源码。


### motivation 提取中间层方向 余弦相似度
README_prefix_harmful_similarity.md 
提取多个方法的中间层和有害方向的余弦相似度,画折线图：python main_harmful_similarity_multi_inputs.py 
结果保存在analysis output里

harmbench evaluate： SIDE/acl_src/baselines/eval.sh
keywords evaluate: SIDE/acl_src/baselines/eavl.sh

环境 side-nlpcc transformers==4.57.3 torch==2.8.0



