"""Build the two Q1 Word reports and exact-source result tables."""

import csv
import json
import statistics
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor
from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parents[1]
Q1 = ROOT / "问题1"
OUT = ROOT / "result"
METRICS = json.loads((OUT / "metrics.json").read_text(encoding="utf-8"))


def write_csv(path, header, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def source_tables():
    wb = load_workbook(ROOT / "附件" / "附件1.xlsx", read_only=True, data_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    write_csv(OUT / "附件1参数原表.csv", rows[0], rows[1:])
    p = METRICS["parameters"]
    audit = [
        ("活化面积", "25 cm²", "附件1", "固定；一维方程按单位面积计算"),
        ("五层厚度 aGDL/aCL/PEM/cCL/cGDL", "150/3.4/12/11.3/150 μm", "附件1", "固定"),
        ("初始孔隙率 aGDL/aCL/cCL/cGDL", "0.8/0.3916/0.4207/0.8", "附件1", "固定"),
        ("冰密度/液水密度", "920/990 kg m⁻³", "附件1", "固定"),
        ("冰融化潜热", "333600 J kg⁻¹", "附件1", "固定"),
        ("双极板密度/比热/厚度", "1980 kg m⁻³ / 766 J kg⁻¹ K⁻¹ / 单侧2 mm", "附件1", "固定；热容缩放严格为1"),
        ("GDL/CL/PEM固相密度与比热", "见附件1参数原表", "附件1", "固定"),
        ("GDL/CL/PEM固相导热系数", "0.30/0.27/0.24 W m⁻¹ K⁻¹", "附件1", "固定"),
        ("对流换热系数", "40 W m⁻² K⁻¹", "附件1", "固定"),
        ("阴极冰覆盖活性面积指数", "3.5", "附件1", "固定"),
        ("初始膜含水量", "3", "附件1", "固定；随后由水量平衡演化"),
        ("膜密度/当量质量", "2150 kg m⁻³ / 1000 g mol⁻¹", "附件1", "固定"),
        ("总接触电阻", "0.01 Ω cm²", "题目备注1", "固定"),
        ("参照气体扩散系数 H₂/O₂", "1.10e−4/2.20e−5 m² s⁻¹", "题目备注1", "固定"),
        ("参照水蒸气扩散系数 阳极/阴极", "8.69e−5/2.48e−5 m² s⁻¹", "题目备注1", "固定"),
        ("热中性电压", "1.48 V", "题目备注1", "固定"),
        ("参考交换电流密度", f'{p["j0_ref_am2"]:.9g} A m⁻²', "备注1给0.01初始校准值", "只用−20℃标定"),
        ("产水膜吸收分配系数", f'{p["uptake_fraction"]:.9g}', "附件1未规定此等效系数", "只用−20℃标定"),
        ("水合活性指数", f'{p["hydration_activity_exponent"]:.9g}', "附件1未规定", "只用−20℃标定"),
        ("水合相关附加界面阻抗", f'{p["dry_interface_ohm_m2"]:.9g} Ω m²', "附件1未规定", "只用−20℃标定；不替代固定接触电阻"),
        ("冻结/融化动力学速率", "0.2/0.2 s⁻¹", "附件1未给有量纲速率", "假设值；做五倍与五分之一敏感性分析"),
        ("液水等效扩散系数", "GDL 2e−9、CL 2e−10 m² s⁻¹", "附件1仅给渗透率，无闭合所需毛细压函数", "等效假设；做敏感性分析"),
    ]
    write_csv(OUT / "参数使用与来源.csv", ("参数", "使用值", "来源", "处理"), audit)
    mrows = []
    for label, m in METRICS["metrics"].items():
        mrows.append((label, m["voltage_mae_v"], m["voltage_rmse_v"],
                      m["voltage_mean_relative_error_percent"], m["temperature_mae_c"],
                      m["temperature_rmse_c"], m["temperature_mean_relative_error_percent_celsius"],
                      m["max_ice_fraction_end"], m["max_pore_ice_saturation_whole_run"],
                      m["max_mass_balance_error_kg_m2"], m["max_energy_balance_error_j_m2"]))
    write_csv(OUT / "验证指标汇总.csv",
              ("工况", "电压MAE/V", "电压RMSE/V", "电压平均相对误差/%", "温度MAE/℃",
               "温度RMSE/℃", "温度平均相对误差/%", "末时最大冰体积分数",
               "最大孔隙冰饱和度", "水量守恒最大误差/kgm^-2", "能量守恒最大误差/Jm^-2"), mrows)
    exp_wb = load_workbook(ROOT / "附件" / "附件2.xlsx", read_only=True, data_only=True)
    consistency = []
    for label in ("-20℃", "-25℃"):
        values = [(float(r[1]), float(r[4])) for r in exp_wb[label].iter_rows(min_row=3, max_col=5, values_only=True)
                  if r[0] is not None and float(r[4]) > 0]
        inferred = [current / density for current, density in values]
        consistency.append((label, 25, round(statistics.median(inferred), 3),
                            round(min(inferred), 3), round(max(inferred), 3),
                            "模型使用电流密度列；25 cm² 不变"))
    write_csv(OUT / "电流面积口径核对.csv",
              ("工况", "附件1活化面积/cm2", "附件2电流除以电流密度中位数/cm2",
               "推算面积最小/cm2", "推算面积最大/cm2", "模型处理"), consistency)
    return audit


def set_cell_shading(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def make_doc(title, subtitle):
    d = Document()
    s = d.sections[0]
    s.page_height, s.page_width = Cm(29.7), Cm(21.0)
    s.top_margin, s.bottom_margin = Cm(2.1), Cm(2.0)
    s.left_margin, s.right_margin = Cm(2.1), Cm(2.1)
    normal = d.styles["Normal"]
    normal.font.name = "Microsoft YaHei"
    normal.font.size = Pt(9.5)
    normal.font.color.rgb = RGBColor(29, 42, 54)
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    normal.paragraph_format.space_after = Pt(5)
    normal.paragraph_format.line_spacing = 1.25
    for name, size, color in [("Title", 19, RGBColor(18, 43, 67)),
                              ("Heading 1", 13, RGBColor(17, 65, 97)),
                              ("Heading 2", 10.5, RGBColor(17, 65, 97))]:
        st = d.styles[name]
        st.font.name = "Microsoft YaHei"
        st.font.size = Pt(size)
        st.font.color.rgb = color
        st.font.bold = True
        st._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
        st.paragraph_format.space_before = Pt(10)
        st.paragraph_format.space_after = Pt(5)
    d.add_paragraph(title, "Title")
    p = d.add_paragraph(subtitle)
    p.runs[0].italic = True
    p.runs[0].font.color.rgb = RGBColor(90, 105, 116)
    return d


def para(d, text):
    return d.add_paragraph(text)


def heading(d, text, level=1):
    d.add_heading(text, level=level)


def equation(d, text):
    p = d.add_paragraph(style="Normal")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after = Pt(7)
    r = p.add_run(text)
    r.font.name = "Cambria Math"
    r.font.size = Pt(10)


def table(d, headers, rows, sizes=None):
    t = d.add_table(rows=1, cols=len(headers))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.style = "Table Grid"
    t.autofit = True
    for i, h in enumerate(headers):
        c = t.rows[0].cells[i]
        c.text = str(h)
        set_cell_shading(c, "E5EDF4")
    for row in rows:
        cells = t.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = str(val)
    for ridx, row in enumerate(t.rows):
        for c in row.cells:
            c.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            for p in c.paragraphs:
                p.paragraph_format.space_after = Pt(0)
                p.paragraph_format.line_spacing = 1.0
                for r in p.runs:
                    r.font.size = Pt(7.4 if len(headers) >= 7 else 8.4)
                    if ridx == 0:
                        r.bold = True
    d.add_paragraph()
    return t


def figure(d, path, caption, width=16.0):
    p = d.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(path), width=Cm(width))
    c = d.add_paragraph(caption)
    c.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for r in c.runs:
        r.font.size = Pt(8)
        r.font.color.rgb = RGBColor(89, 101, 112)


def model_doc(audit):
    d = make_doc("一维单电池瞬态自冷启动模型", "问题 1 建模文档  |  五层膜电极  |  附件 1 固定参数")
    para(d, "本模型沿膜电极厚度建立一维瞬态有限体积模型，在题目备注 1 的常温传热、传质和电压关系上加入液态水冻结与融化、冰占孔以及阴极反应面积衰减。用附件 2 的 −20℃ 工况标定未由附件 1 指定的少量等效参数，用 −25℃ 工况检验外推。输出为计算域平均温度、局部最大冰体积分数和单电池电压。")
    heading(d, "1 题意与证据来源")
    para(d, "题目 PDF 第 2–3 页要求三相水分解、相变控制方程、冰占孔反馈和两组实验表；第 8–14 页备注 1 给出五层几何、气体及水守恒、热平衡和电压方程；第 15 页备注 2 给出耦合关系。附件 1 提供固定物性，附件 2 提供每组 184 个、0–36.6 s 的观测。随题提供的 Luo 与 Jiao 综述阐述了膜水吸收、催化层结冰和气体传输阻塞的耦合机理 [2]；Jiao 与 Li 的多相模型支持把相变、孔隙和反应耦合求解 [3]。")
    para(d, "备注 3、备注 4 分别说明竞赛写作与代码附件的提交规范；它们不是问题 1 的物理参数或边界条件。本次交付按用户指定目录提供问题 1 的建模文档、数据分析和结果文件。")
    heading(d, "2 空间区域与建模假设")
    table(d, ["功能层", "厚度/μm", "初始孔隙率", "基础热导率/W m⁻¹ K⁻¹", "基础体积热容/J m⁻³ K⁻¹"],
          [["aGDL", "150", "0.8", "0.30", "185×545"],
           ["aCL", "3.4", "0.3916", "0.27", "970×240"],
           ["PEM", "12", "—", "0.24", "2150×1050"],
           ["cCL", "11.3", "0.4207", "0.27", "970×240"],
           ["cGDL", "150", "0.8", "0.30", "185×545"]])
    para(d, "氢气从阳极外表面、干空气从阴极外表面进入；入口绝对压力均取附件 1 的 101325 Pa。外加电流密度为附件 2 的逐时输入，阴极与阳极催化层内反应按层厚均匀分配。气体浓度采用准稳态通量解，温度与孔隙水、冰则瞬态求解。膜含水量用一维平均状态近似，双极板的单位面积热容并入热方程；未显式离散微孔层、流道和端板。")
    heading(d, "3 状态变量、三相水和冰占孔")
    equation(d, "mₚ = mᵥ + mₗ + mᵢ ;  εᵢ = mᵢ/ρᵢ ;  εg = ε₀ − εᵢ − mₗ/ρₗ")
    para(d, "mₚ、mᵥ、mₗ、mᵢ 均为单位控制体体积中的质量浓度（kg m⁻³）。局部冰体积分数 εᵢ 按题目式（9）定义，分母是整个控制体体积；孔隙冰饱和度 εᵢ/ε₀ 是另一指标，报告中不混用。取 ρᵢ=920、ρₗ=990 kg m⁻³。气体可通孔隙为 εg，非负下限仅用于数值稳定。")
    equation(d, "mₛₐₜ = (ε₀−εᵢ) Mᵥ pₛₐₜ(T)/(RT);  mᵥ=min(mₚ−mᵢ,mₛₐₜ);  mₗ=max(mₚ−mᵢ−mᵥ,0)")
    para(d, "饱和蒸气压按备注 1 式（28）的 Buck 分段关系，低于 0℃ 使用冰面分支；蒸气与液水作局部瞬时分相，不额外引入蒸发系数。")
    heading(d, "4 水迁移、结冰与膜吸水")
    equation(d, "∂mₚ/∂t = −∂(Nᵥ+Nₗ)/∂x + Sᵥ − Sₘ;   ∂mᵢ/∂t = r𝒇 − rₘ")
    equation(d, "Nᵥ = −Dᵥ,ref(T/Tref)¹·⁷⁵ εg¹·⁵ ∂(mᵥ/εg)/∂x;   Nₗ = −Dₗ,eq ∂mₗ/∂x")
    equation(d, "r𝒇 = k𝒇 [(273.15−T)/20]₊ mₗ (1−εᵢ/ε₀)₊;   rₘ = kₘ [(T−273.15)/5]₊ mᵢ")
    para(d, "产水仅在 cCL，Sᵥ=jMᵥ/(2FLcCL)。相变式表达了温度驱动、液水供应和有限孔容；冻结水仍计入总水，冻结潜热返回热方程。模型用膜水平均状态 λ 表示产水的一部分进入膜，λ 的初值严格取附件 1 的 3，膜密度和当量质量也按附件 1。膜吸水分配是未给定的等效闭合，在 −20℃ 工况标定；附件 1 的无量纲转化系数未被误当成有量纲冻结速率。")
    heading(d, "5 气体守恒和电压闭合")
    equation(d, "Dₖ,eff = Dₖ,ref (T/298.15)¹·⁷⁵ εg¹·⁵ ;  Sᴴ₂=−j/(2FL aCL),  Sᴼ₂=−j/(4FL cCL)")
    equation(d, "aᵢ,eff/aᵢ,0 = (1−εᵢ/ε₀)³·⁵ ;  V=Erev−ηact−ηohm−ηcon")
    para(d, "阳极和阴极参考气体扩散系数分别为 1.10×10⁻⁴ 与 2.20×10⁻⁵ m² s⁻¹，孔隙指数 1.5 与备注 1 一致。阴极冰覆盖面积指数 3.5 严格取附件 1。可逆电压采用备注 1 式（38），活化损失采用式（39）和 Arrhenius 型交换电流密度，欧姆损失保留 Springer 膜电导率以及固定 0.01 Ω cm² 接触电阻，浓差损失采用阴极等效扩散阻力计算极限电流密度。水合状态还影响有效反应面积和附加界面阻抗；这两项仅作为附件 1 未规定的等效校准闭合，不替代固定接触电阻。")
    heading(d, "6 能量方程及边界条件")
    equation(d, "Ceff ∂T/∂t = ∂[keff ∂T/∂x]/∂x + j(Eth−V)/L + L𝒇 ∂mᵢ/∂t")
    para(d, "Eth=1.48 V、冻结潜热 L𝒇=333600 J kg⁻¹；两侧均用 h=40 W m⁻² K⁻¹ 对流边界。基础层热容和导热率严格按附件 1，双极板热容 2×(0.002 m)×1980×766 J m⁻² K⁻¹ 按单位面积分摊到计算域，缩放系数固定为 1。初温取对应实验首个观测，初始孔隙液水与冰为零；膜初始含水量为 3。热传导在层界面以热阻串联，确保界面热流连续。")
    heading(d, "7 参数锁定、校准与可辨识性")
    para(d, "附件 1 中用于本模型的几何、孔隙率、物性、膜初值、冰覆盖指数和边界换热值均固定。详细逐项原表见 result/附件1参数原表.csv，使用与来源见 result/参数使用与来源.csv。参考交换电流密度在备注 1 中明确标为“初始校准值”，可识别；产水膜吸收分配、水合活性指数与附加界面阻抗均非附件 1 的给定参数，只在 −20℃ 的电压及平均温度数据上作最小二乘校准。")
    table(d, ["校准量", "识别值", "角色"],
          [["j₀,ref", f'{METRICS["parameters"]["j0_ref_am2"]:.6f} A m⁻²', "交换动力学，备注1允许校准"],
           ["膜吸水分配", f'{METRICS["parameters"]["uptake_fraction"]:.6f}', "等效产水分流"],
           ["水合活性指数", f'{METRICS["parameters"]["hydration_activity_exponent"]:.6f}', "水合对反应面积影响"],
           ["附加界面阻抗", f'{METRICS["parameters"]["dry_interface_ohm_m2"]:.6g} Ω m²', "水合相关附加阻抗"]])
    para(d, "−20℃ 校准目标把电压残差除以 0.020 V、温度残差除以 0.30℃ 后合并，并采用稳健 soft-L1 损失。−25℃ 的任何观测都不进入目标函数。冻结和等效液水迁移速率没有独立实测支撑，作为模型假设作敏感性分析，冰量数值不应解释为已被独立验证。")
    heading(d, "8 数值算法与输出定义")
    para(d, "五层按 8/3/4/5/8 个控制体划分，校准后采用每层三倍细分共 84 格。时步最大 0.05 s；每步先按输入电流计算产水和膜吸收，再以守恒隐式三对角线系统更新蒸气与液水，局部冻结/融化更新冰，最后以隐式热传导更新温度与电压。输出平均温度为 Σ(TᵢΔxᵢ)/ΣΔxᵢ；最大冰体积分数为 maxᵢ(mᵢ/ρᵢ)。在各测量时刻输出曲线，在 0、5、…、35 s 填表。")
    para(d, "水量核对：累计产水 = 孔隙留存 + 膜吸收 + 边界逸出；热量核对：电化学热 + 相变潜热 = 环境散热 + 热储存。检查非负气体浓度、孔隙占用小于 1、84/112 格差异与 0.05/0.025 s 时步差异。")
    heading(d, "9 适用范围与局限")
    para(d, "模型为单位面积一维简化模型，气体准稳态、膜水平均化，未完整解析毛细压、液水渗透率、膜内扩散和微孔层。附件 1 的接触角、渗透率和膜水转化系数由于缺少相应闭合方程未直接代入等效扩散式，未用的参数在来源表中区分。实验只到 36.6 s，平均温度始终低于 0℃，因此只能验证这一短时温升和电压预测，不能声称已验证解冻或成功启动。附件 2 电流/电流密度与 25 cm² 活化面积不一致；本模型以附件 2 已提供的电流密度作为单位面积输入，固定活化面积 25 cm² 不作反推。")
    heading(d, "参考文献")
    para(d, "[1] 2026 年中国研究生数学建模竞赛 B 题《氢燃料电池低温冷启动建模与控制策略研究》，备注 1、附件 1–2。")
    para(d, "[2] Luo Y, Jiao K. Cold start of proton exchange membrane fuel cell. Progress in Energy and Combustion Science, 2018, 64:29–61. DOI: 10.1016/j.pecs.2017.10.003。随题参考文献/main.pdf。")
    para(d, "[3] Jiao K, Li X. Three-dimensional multiphase modeling of cold start processes in polymer electrolyte membrane fuel cells. Electrochimica Acta, 2009, 54:6876–6891. DOI: 10.1016/j.electacta.2009.06.072。")
    para(d, "[4] Springer T E, Zawodzinski T A, Gottesfeld S. Polymer Electrolyte Fuel Cell Model. Journal of The Electrochemical Society, 1991, 138:2334–2342. DOI: 10.1149/1.2085971。")
    d.save(Q1 / "一维单电池瞬态自冷启动模型建模文档.docx")


def analysis_doc():
    d = make_doc("问题 1 数据结果分析", "−20℃ 参数校准与 −25℃ 独立验证  |  附件 1 固定参数")
    m20, m25 = METRICS["metrics"]["-20℃"], METRICS["metrics"]["-25℃"]
    para(d, f'采用附件 1 固定参数并仅在 −20℃ 上识别未规定的等效闭合后，−20℃ 的电压/温度 RMSE 分别为 {m20["voltage_rmse_v"]:.4f} V 和 {m20["temperature_rmse_c"]:.4f}℃；未参与校准的 −25℃ 工况分别为 {m25["voltage_rmse_v"]:.4f} V 和 {m25["temperature_rmse_c"]:.4f}℃。两组均为短时、未跨越 0℃ 的自冷启动过程，冰量只有模型预测，缺少实验直接校验。')
    heading(d, "1 数据口径与相对误差")
    para(d, "附件 2 的两张工作表各含 184 个采样点，时间为 0–36.6 s，字段为时间、电流、电压、温度、电流密度。−20℃ 用于校准；−25℃ 完全留出。模型驱动取附件 2 的电流密度列。附件 1 固定活化面积 25 cm²，但附件 2 电流/电流密度之比约为 300 cm²，故两列不能同时视为同一 25 cm² 电池的量；未擅自改动面积或重算电流密度。")
    equation(d, "εX(t) = 100% × |Xsim(t)−Xexp(t)| / |Xexp(t)|")
    para(d, "按题目式（1）分别对电压 V 和以摄氏度记录的温度 T 计算相对误差。温度相对误差是题目指定的表格指标，量纲与零点选择敏感；跨工况比较同时给出更稳定的 MAE 与 RMSE。")
    heading(d, "2 题目表 1 与表 2")
    headers = ["时间/s", "实验电压/V", "模型电压/V", "电压误差/%", "实验温度/℃",
               "模型温度/℃", "温度误差/%", "最大冰体积分数"]
    for stem, label in [("minus20", "−20℃（校准）"), ("minus25", "−25℃（验证）")]:
        with (OUT / f"table_{stem}.csv").open(encoding="utf-8-sig", newline="") as f:
            rows = [list(r.values()) for r in csv.DictReader(f)]
        heading(d, label, 2)
        table(d, headers, rows)
    heading(d, "3 全时序误差与预测表现")
    table(d, ["工况", "电压 MAE/V", "电压 RMSE/V", "电压平均相对误差/%", "温度 MAE/℃", "温度 RMSE/℃", "温度平均相对误差/%"],
          [[lab, f'{m["voltage_mae_v"]:.4f}', f'{m["voltage_rmse_v"]:.4f}',
            f'{m["voltage_mean_relative_error_percent"]:.3f}',
            f'{m["temperature_mae_c"]:.4f}', f'{m["temperature_rmse_c"]:.4f}',
            f'{m["temperature_mean_relative_error_percent_celsius"]:.3f}']
           for lab, m in [("−20℃", m20), ("−25℃", m25)]])
    para(d, "电压误差在 −25℃ 增大，前 15 s 的预测多偏低约 0.02 V，之后逐渐靠近实测值；这说明在更低温度下，未直接测量的膜水合及反应动力学温度依赖可能需要独立数据约束。温度在 −25℃ 后段略偏高，35 s 处约高 0.28℃。")
    figure(d, OUT / "validation_curves.png", "图 1  两组电压、平均温度、最大冰体积分数与电压残差", 16.2)
    heading(d, "4 水和冰的时空演化")
    para(d, f'在 36.6 s 末，−20℃ 最大局部冰体积分数为 {m20["max_ice_fraction_end"]:.5f}，−25℃ 为 {m25["max_ice_fraction_end"]:.5f}；对应最大孔隙冰饱和度分别为 {m20["max_pore_ice_saturation_whole_run"]:.5f} 和 {m25["max_pore_ice_saturation_whole_run"]:.5f}。较冷工况更早出现冰、末时冰量更高。两组最大总孔隙占用均小于 0.1，尚远未达到完全冰堵，不能用这些曲线判定未来启动是否成功。')
    figure(d, OUT / "water_ice_dynamics.png", "图 2  平均膜含水量与最大孔隙冰饱和度", 15.7)
    para(d, "35 s 的 84 个网格局部温度、水量和冰量分别保存于 field_minus20_35s.csv 与 field_minus25_35s.csv；完整 184 点轨迹保存在 trajectory_minus20.csv 与 trajectory_minus25.csv。")
    figure(d, OUT / "field_profiles_35s.png", "图 3  35 s 时膜电极厚度方向的温度与冰体积分数分布", 15.7)
    heading(d, "5 守恒、收敛和参数敏感性")
    table(d, ["检验", "−20℃", "−25℃"],
          [["水质量最大闭合误差/kg m⁻²", f'{m20["max_mass_balance_error_kg_m2"]:.2e}', f'{m25["max_mass_balance_error_kg_m2"]:.2e}'],
           ["能量最大闭合误差/J m⁻²", f'{m20["max_energy_balance_error_j_m2"]:.2e}', f'{m25["max_energy_balance_error_j_m2"]:.2e}'],
           ["最大孔隙总占用", f'{m20["max_pore_occupancy_whole_run"]:.5f}', f'{m25["max_pore_occupancy_whole_run"]:.5f}'],
           ["最小气体浓度/mol m⁻³", f'{m20["minimum_gas_concentration_mol_m3"]:.3f}', f'{m25["minimum_gas_concentration_mol_m3"]:.3f}']])
    n = METRICS["numerical_checks"]
    para(d, f'在 −20℃ 条件下，84 格增至 112 格的电压最大差 {n["grid_84_to_112_voltage_max_diff_v"]:.2e} V、温度最大差 {n["grid_84_to_112_temperature_max_diff_c"]:.2e}℃；最大时步 0.05 s 减半后电压最大差 {n["time_step_0p05_to_0p025_voltage_max_diff_v"]:.2e} V、温度最大差 {n["time_step_0p05_to_0p025_temperature_max_diff_c"]:.4f}℃。这些是数值稳定性证据，不等于物理参数已唯一识别。')
    figure(d, OUT / "balance_and_ice_sensitivity.png", "图 4  水量守恒残差与未实测相变参数的敏感性", 15.7)
    para(d, "冻结速率提高五倍时预测最大冰体积分数明显上升，液水等效迁移增强则冰积累下降。冰的绝对值依赖未独立标定的相变与液水闭合，需冰成像、局部含水或更长时间实验进一步检验。")
    heading(d, "6 结论及使用边界")
    para(d, "模型在固定附件 1 已给定参数的前提下，较好复现实验窗口内的温度和电压变化，并在留出工况上保持可用但误差变大。局部结冰趋势与低温物理机制一致，属于模型推断。实验终点的平均温度仍低于冰点，没有足够证据报告启动成功时刻、最低可启动温度或长期冰堵阈值。附件 2 的电流与电流密度口径冲突，在将此模型用于电堆优化前需要核实。")
    heading(d, "结果文件")
    para(d, "result/ 内提供题目两张填写表、184 点逐时数据、35 s 局部场数据、验证指标、参数来源清单、运行记录和三张图。问题1/run.py 可重新计算，问题1/checks.py 可复核数据、物理边界、守恒和数值收敛。")
    d.save(Q1 / "问题1数据结果分析文档.docx")


if __name__ == "__main__":
    audit = source_tables()
    model_markdown = Q1 / "一维单电池瞬态自冷启动模型建模文档.md"
    if not model_markdown.exists():
        model_doc(audit)
    analysis_doc()
    from docx_to_md import convert, MODEL_MATH, ANALYSIS_MATH
    if not model_markdown.exists():
        convert(Q1 / "一维单电池瞬态自冷启动模型建模文档.docx",
                model_markdown, MODEL_MATH, [])
    convert(Q1 / "问题1数据结果分析文档.docx",
            Q1 / "问题1数据结果分析文档.md", ANALYSIS_MATH,
            [("validation_curves.png", "电压、平均温度、最大冰体积分数与电压残差"),
             ("water_ice_dynamics.png", "平均膜含水量与最大孔隙冰饱和度"),
             ("field_profiles_35s.png", "35 秒时厚度方向的温度与冰分布"),
             ("balance_and_ice_sensitivity.png", "水量守恒残差与冰参数敏感性")])
    if (Q1 / "一维单电池瞬态自冷启动模型建模文档.docx").exists():
        (Q1 / "一维单电池瞬态自冷启动模型建模文档.docx").unlink()
    (Q1 / "问题1数据结果分析文档.docx").unlink()
    print("created 2 Markdown reports and 4 tabular outputs")
