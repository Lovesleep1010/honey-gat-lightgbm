# 蜂蜜高光谱分类：残差GAT＋LightGBM

这里包含残差GAT＋LightGBM、全局统计特征消融、k敏感性、波长归因、16/32/48波段实验、可见光/近红外区域实验，以及GraphSAGE、GCN、GIN和普通GAT对照。

## 直接运行

1. 用PyCharm打开整个仓库，选择已有PyTorch环境。
2. 其他依赖执行 `python -m pip install -r requirements.txt`。
3. 打开根目录的 `run.py`，选择 `EXPERIMENT`，点击运行。

数据已放在 `data/` 中，所有入口自动读取，不需要指定工作目录。默认入口是当前的主模型版本。

| 入口 | 内容 | 默认训练配置 |
| --- | --- | --- |
| `main` | 主模型及当前训练版本 | 最多50轮，连续5轮下游验证Bal-Acc未提高时早停 |
| `analysis` | 消融与光谱分析：消融、k、选带、区域比较和光谱LightGBM | 保留实际使用的5轮配置 |
| `baselines` | 表1的四个GNN对照 | 保留实际使用的5轮配置和各自原始模型设置 |

具体参数在 `experiments/对应入口/config.py` 中修改。主模型入口默认只运行全谱配置；将其中 `EXPERIMENTS` 改为 `["all"]` 可运行该版本的全部实验组。结果写入各入口下的 `results_*` 文件夹。

主模型与消融分析分别保留各自的训练配置，运行记录会保存对应的实际训练轮数和模型选择设置。数据、模型输入和结果应按各自配置对应使用。

目录结构：

```text
run.py                  # 总运行入口
experiments/
  main/                 # 残差GAT＋LightGBM主模型
  analysis/             # 消融、k敏感性、归因及波段实验
  baselines/            # 四个GNN基线
data/                   # 数据及来源说明
```

各实验目录中，`config.py` 修改运行设置，`model.py` 定义模型，`pipeline.py` 执行实验流程。四个基线的模型定义在 `experiments/baselines/models.py`。

## 自检

```bash
python check_installation.py
python check_installation.py --tests
```

第一条检查依赖、数据和三个入口；第二条额外运行已有的流程测试，使用模拟或小型合成样本。

## 数据与许可

CSV与Figshare原始文件一致，来源、作者、CC BY 4.0许可和校验值见 [data/README.md](data/README.md)。数据清洗规则由运行代码执行。代码暂按MIT许可整理，见 [LICENSE](LICENSE)。

完整的英文运行说明见 [README.md](README.md)。
