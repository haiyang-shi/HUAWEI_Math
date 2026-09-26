# 问题4复现说明

本目录重算问题4的动态辅助加热控制和 10–100 min 预冷温度场。参数不是手工补写：`问题4/run.py` 会先使用 `附件/附件1.xlsx`、`附件/附件2.xlsx` 重新执行问题1，生成根目录 `result/metrics.json`，再运行问题4。

## 一键运行

在项目根目录使用包含 NumPy、SciPy、Matplotlib 和 openpyxl 的 Python 环境执行：

```text
python 问题4/run.py
```

完整运行包含每片 84 网格、0.025 s 时间步的最终复核，以及 36 个动态候选和 19 个预冷扫描点，耗时取决于处理器性能。

## 选优口径

候选先满足功率、电压、冰量和启动成功硬约束，并要求动态启动时间不超过同工况问题3恒功率基线。随后在辅助能耗、启动时间、最大温差、最低单片电压和最大冰体积分数五个指标上构造 Pareto 集，再以能耗权重最高的归一化理想点距离选取折中解。这样不会再选出“零加热但明显更慢”的退化方案。

## 主要文件

- `dynamic_model.py`：预冷热网络、五片动态模型和四状态独立控制器；
- `optimize.py`：三工况候选搜索、Pareto 选优、表4、轨迹、温度场和图；
- `verify.py`：题面约束、终点撤热、候选和预冷扫描自动审计；
- `build_report.py`：从实际结果重建中文分析文档；
- `build_manifest.py`：对附件、模型、代码、结果和文档生成 SHA-256 清单；
- `问题4的解决方案与结果分析.md`：完整建模、结果与解释；
- `results/table_q4.csv`：表4六行结果；
- `results/multiobjective_candidates*.json`：全部候选、Pareto 标记和敏感性；
- `results/precooling_temperature_field*.csv`：带物理坐标的 $T(x,t)$；
- `results/verification.json`：自动验收结论；
- `results/run_manifest.json`：可复现文件清单。
