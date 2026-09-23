# B题数学建模 Skills 下载记录

下载日期：2026-09-23。安装范围：当前 B 题项目。安装目录：`.agents/skills/`。

已安装 12 个 skills，共 74 个原始文件，全部与 GitHub 固定提交的 Git blob SHA-1 一致，且每个 SKILL.md 均含 name 与 description。详细校验见 [verification.json](verification.json)，逐文件来源见 [source-manifest.json](source-manifest.json)。

## 数学建模工作流

来源：[yushui2022/MathModel-Skill](https://github.com/yushui2022/MathModel-Skill/tree/0cc261d90d21e4ed540b02b0c71018cdcd47af58/packages/codex/.agents/skills)，standard 分支的 Codex 版；固定提交 `0cc261d90d21e4ed540b02b0c71018cdcd47af58`。

| Skill | 用途 |
| --- | --- |
| paper-workflow-orchestrator | 串联题意分析、建模、结果验证和论文工作流 |
| problem-doc-model-selector | 拆解各问的输入、输出、约束和模型路线 |
| modeling-paper-rubric-and-model-selector | 比较模型路线并规划验证证据 |
| authoritative-data-harvester | 按需获取公开数据并记录来源 |
| data-cleaning-and-visualization | 整理附件数据、规划与生成图表 |
| model-code-and-result-generator | 构建建模代码与结果证据框架 |
| quality-assurance-auditor | 核查计算结果、图表和结论的一致性 |
| paper-formal-writer | 整理正式论文与可编辑公式 |
| paper-micro-unit-generator | 按工作流需要修复局部论文内容 |
| context-memory-keeper | 记录项目工作流状态和产物 |

这些 skills 互相引用，因此完整安装同一个版本，避免缺失依赖。它们提供工作流和代码框架；本题的传热传质、水冰相变、冰堵、电压及加热控制方程仍需依据题目与文献建立，不能把通用框架输出当作求解结果。

## 符号推导与优化

来源：[K-Dense-AI/scientific-agent-skills](https://github.com/K-Dense-AI/scientific-agent-skills/tree/49c6e97775eaa18ba791bebe23162a70ae601c18/skills)，固定提交 `49c6e97775eaa18ba791bebe23162a70ae601c18`。

| Skill | 本题预期用途 |
| --- | --- |
| sympy | 方程整理、符号求导、雅可比矩阵及公式转数值函数 |
| pymoo | 电流加载参数、加热功率分配等受约束优化，以及能耗和启动时间的多目标比较 |

本次检索中，yaklang/control-theory-skill 和 DBvc/agent-skill-control-theory 的内容偏向 AI Agent 设计，与本题工程控制需求不够匹配，因此未安装。未找到并验证本题专用的氢燃料电池冷启动 skill。

## 使用状态

- Skills 已下载并安装；下一轮对话可使用。
- 尚未运行第三方建模脚本，尚未安装或升级 Python 依赖。
- MathModel 上游依赖列表已另存为 [mathmodel-requirements.txt](mathmodel-requirements.txt)，并非当前环境依赖已满足的证明。
- 后续运行时再检查所选 Python 环境的数值计算、绘图与文档依赖；pymoo skill 的下载不等于 pymoo Python 库的安装。
- 原始赛题及附件保持在原位置。本次仅根据题面文本筛选 skills，尚未完成公式、图片和附件参数的完整解析。

下一步可直接要求：使用已安装的 skills，读取题目与附件，先梳理四个问题的建模路线、参数需求和数值求解方案。
