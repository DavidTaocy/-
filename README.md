简要说明

这是一个在 VS Code 中可运行的图书推荐系统示例工程，适用于隐式交互数据集（Goodbooks-10k 样式）。

目录结构
- data/                    # 放置赛方提供的数据：train/test/submission样例
- outputs/                 # 输出预测 submission.csv
- src/                     # 源代码
  - data_loader.py         # 读取/预处理数据
  - model_cf.py            # 基于用户的协同过滤（cosine 相似度）实现
  - model_mf.py            # 矩阵分解（TruncatedSVD）实现
  - main.py                # 运行入口，生成 submission.csv
- requirements.txt         # Python 依赖

快速开始
1. 在 `data/` 里放置赛方数据文件（支持 `train.csv` 或 `train_dataset.csv`，`test.csv` 或 `test_dataset.csv`，以及 `submission.csv` 示例）。
2. 安装依赖：

```bash
pip install -r requirements.txt
```

3. 运行（示例：使用矩阵分解，生成每用户 top-10）：

```bash
python src/main.py --method mf --top_k 10 --data_dir data --out outputs/submission.csv
```

生成文件 `outputs/submission.csv` 将包含两列：`user_id,item_id`（每个用户多行，按预测优先顺序）。

说明
- 支持两种模型：`cf`（用户基协同过滤）和 `mf`（矩阵分解 / SVD）。
- 仅使用赛方 `data/` 中的数据，不引入外部数据。
