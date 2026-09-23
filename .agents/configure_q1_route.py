"""Correct the generic Q1 classification after the upstream skill parses the PDF.

Run from the project root after analyze_problem.py and before
build_model_route.py.  The input contracts remain valid for all four questions.
"""

import json
import importlib.util
from pathlib import Path


ROOT = Path.cwd()
PATH = ROOT / "paper_output/step1/problem_analysis.json"
analysis = json.loads(PATH.read_text(encoding="utf-8"))
questions = analysis.get("questions", [])
q1 = next((q for q in questions if q.get("id") == "Q1"), None)
if q1 is None or len(questions) != 4:
    raise SystemExit("Expected Q1-Q4 from the current PDF before configuring Q1")

q1.update(
    task_type="机理/仿真与参数校准",
    inputs=["附件1几何和材料参数", "附件2两种温度的逐时电流密度、电压、温度"],
    outputs=["一维平均温度", "局部与最大冰体积分数", "单电池电压", "-20/-25摄氏度题定结果表"],
    constraints=[
        "五层一维膜电极区域，低温水蒸气/液态水/冰分相",
        "显式液态水/冰相变、潜热、冰占孔对扩散和催化面积的反馈",
        "使用附件1参数与附件2实验校准和跨温度验证",
        "相对误差按题目公式(1)计算；平均温度按厚度积分计算",
    ],
    recommended_models={
        "baseline": "备注1的常温一维传热传质与电压模型",
        "improved": "有限体积多相冷启动模型：水/冰相变、有效孔隙率与扩散、反应面积反馈",
    },
    validation_plan=[
        "-20摄氏度校准少量参数，-25摄氏度留作独立验证",
        "温度、电压逐时残差和题定相对误差",
        "水质量与能量守恒、非负性、网格收敛、相变参数敏感性",
    ],
    figure_suggestions=[
        "电压和温度实验/模型对照及残差",
        "最大冰体积分数时间曲线和厚度剖面",
        "冰相变参数敏感性与网格收敛对照",
    ],
)
analysis["project_review"] = {
    "primary_statement": "problem_files/氢燃料电池低温冷启动建模与控制策略研究.pdf",
    "q1_classification_corrected": True,
    "reason": "Generic keyword classification mistook a coupled physical simulation for forecasting regression.",
    "formula_review": "MathType/OLE equations checked against the new PDF pages 2-3 and 9-15.",
}
PATH.write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
selector_path = ROOT / ".agents/skills/problem-doc-model-selector/scripts/analyze_problem.py"
spec = importlib.util.spec_from_file_location("project_problem_selector", selector_path)
selector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(selector)
selector.write_markdown_outputs(analysis)
print("Q1 route configured as physical simulation and calibration; Q2-Q4 retained")
