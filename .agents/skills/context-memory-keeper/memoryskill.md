# Context Memory (Active)

> Instructions for Agent:
> 1. Read this file before long MathModel workflows.
> 2. Keep long-term principles stable unless the user explicitly changes them.
> 3. Update the short-term workbench after major steps: problem parsing, data processing, QA, paper generation, and final delivery.
> 4. When this file becomes too long, move obsolete details to `memory_archive.md` and keep only durable conclusions here.

## 1. Long-Term Principles

- Role: mathematical modeling workflow assistant.
- Output language: Chinese academic style unless the user asks otherwise.
- Delivery target: keep Markdown and Word outputs aligned when a full paper is requested.
- Workflow rule: preserve the chain `problem parsing -> model selection -> data/code adaptation -> QA -> micro-unit generation -> merge`.
- Script rule: treat bundled `scripts/` as reusable code templates and code-level prompts; adapt them to the current problem before trusting outputs.

## 2. Short-Term Workbench

- Current problem: 2026中国研究生数学建模竞赛B题，当前仅完成问题一。
- Problem files: 根目录新版PDF为准；`problem_files/`中为PDF、附件1/2工作流副本。原Word仅用于交叉核对嵌入公式。
- Data sources: 附件1参数表；附件2两个工况各184点。附件2的电流和电流密度与附件1的25 cm²面积不一致，问题一输入以附件2电流密度列为准。
- Model route: `.agents/project_config.json`分配12个技能；Q1为五层一维多相传热传质及参数校准，不是通用时间序列回归。`paper_output/step1/`和`paper_output/plan/`已同步更正。
- Generated figures: `问题1/results/`中电压/温度验证、水冰动态、水量守恒和冰参数敏感性三幅图。
- QA status: Q1专用检查已通过，水/能量守恒、物理上下界、84/112网格和0.05/0.025 s时间步已检查；-25℃为冻结参数后的独立验证。尚未运行四问正式论文门禁。
- Final paper: 未生成四问完整论文；问题一结果报告为`问题1/问题1解决报告.md`，详细审题、数据口径、模型建立、数值算法与问题处理记录为`问题1/问题一建模思路与过程分析.md`。

## 3. External Resources / Literature

- None recorded yet.

## 4. Open Todos

- [x] 解析新版PDF、附件1和附件2，核对问题一具体要求。
- [x] 确定四问技能分工与Q1机理模型路线。
- [x] 针对问题一建立数据读取、数值模型与结果表。
- [x] 完成问题一专用QA和报告。
- [ ] 后续若用户要求四问完整论文，再扩展问题二至四，运行正式证据门禁与论文生成。
