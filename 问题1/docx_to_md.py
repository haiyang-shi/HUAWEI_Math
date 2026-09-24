"""Convert the authored Q1 reports to Markdown with TeX display equations."""
from pathlib import Path

from docx import Document
from docx.document import Document as _Document
from docx.table import Table
from docx.text.paragraph import Paragraph
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P

HERE = Path(__file__).resolve().parent

MODEL_MATH = iter([
    r"m_p=m_v+m_l+m_i,\qquad \varepsilon_i=\frac{m_i}{\rho_i},\qquad \varepsilon_g=\varepsilon_0-\varepsilon_i-\frac{m_l}{\rho_l}",
    r"m_{\mathrm{sat}}=\frac{(\varepsilon_0-\varepsilon_i)M_w p_{\mathrm{sat}}(T)}{RT},\quad m_v=\min(m_p-m_i,m_{\mathrm{sat}}),\quad m_l=\max(m_p-m_i-m_v,0)",
    r"\frac{\partial m_p}{\partial t}=-\frac{\partial(N_v+N_l)}{\partial x}+S_w-S_m,\qquad \frac{\partial m_i}{\partial t}=r_f-r_m",
    r"N_v=-D_{v,\mathrm{ref}}\left(\frac{T}{T_{\mathrm{ref}}}\right)^{1.75}\varepsilon_g^{1.5}\frac{\partial(m_v/\varepsilon_g)}{\partial x},\qquad N_l=-D_{l,\mathrm{eq}}\frac{\partial m_l}{\partial x}",
    r"r_f=k_f\left[\frac{273.15-T}{20}\right]_+m_l\left[1-\frac{\varepsilon_i}{\varepsilon_0}\right]_+,\qquad r_m=k_m\left[\frac{T-273.15}{5}\right]_+m_i",
    r"D_{k,\mathrm{eff}}=D_{k,\mathrm{ref}}\left(\frac{T}{298.15}\right)^{1.75}\varepsilon_g^{1.5},\quad S_{H_2}=-\frac{j}{2FL_{aCL}},\quad S_{O_2}=-\frac{j}{4FL_{cCL}}",
    r"\frac{a_{i,\mathrm{eff}}}{a_{i,0}}=\left(1-\frac{\varepsilon_i}{\varepsilon_0}\right)^{3.5},\qquad V=E_{\mathrm{rev}}-\eta_{\mathrm{act}}-\eta_{\mathrm{ohm}}-\eta_{\mathrm{con}}",
    r"C_{\mathrm{eff}}\frac{\partial T}{\partial t}=\frac{\partial}{\partial x}\left(k_{\mathrm{eff}}\frac{\partial T}{\partial x}\right)+\frac{j(E_{\mathrm{th}}-V)}{L}+L_f\frac{\partial m_i}{\partial t}",
])
ANALYSIS_MATH = iter([
    r"\varepsilon_X(t)=100\%\,\frac{|X_{\mathrm{sim}}(t)-X_{\mathrm{exp}}(t)|}{|X_{\mathrm{exp}}(t)|}",
])


def blocks(document):
    for child in document.element.body.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, document)
        elif isinstance(child, CT_Tbl):
            yield Table(child, document)


def convert(docx_path, md_path, math_lines, figures):
    d = Document(docx_path)
    figure_iter = iter(figures)
    lines = []
    for block in blocks(d):
        if isinstance(block, Table):
            rows = [[c.text.replace("|", "\\|").replace("\n", "<br>") for c in row.cells] for row in block.rows]
            lines.append("| " + " | ".join(rows[0]) + " |")
            lines.append("| " + " | ".join(["---"] * len(rows[0])) + " |")
            lines.extend("| " + " | ".join(row) + " |" for row in rows[1:])
            lines.append("")
            continue
        p = block
        if p._p.xpath(".//w:drawing"):
            name, alt = next(figure_iter)
            lines.extend([f"![{alt}](../result/{name})", ""])
            continue
        t = p.text.strip()
        if not t:
            continue
        if any(r.font.name == "Cambria Math" for r in p.runs):
            lines.extend(["$$", next(math_lines), "$$", ""])
        elif p.style.name == "Title":
            lines.extend([f"# {t}", ""])
        elif p.style.name == "Heading 1":
            lines.extend([f"## {t}", ""])
        elif p.style.name == "Heading 2":
            lines.extend([f"### {t}", ""])
        else:
            lines.extend([t, ""])
    md_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(md_path)


if __name__ == "__main__":
    convert(
        HERE / "一维单电池瞬态自冷启动模型建模文档.docx",
        HERE / "一维单电池瞬态自冷启动模型建模文档.md",
        MODEL_MATH, [],
    )
    convert(
        HERE / "问题1数据结果分析文档.docx",
        HERE / "问题1数据结果分析文档.md",
        ANALYSIS_MATH,
        [
            ("validation_curves.png", "电压、平均温度、最大冰体积分数与电压残差"),
            ("water_ice_dynamics.png", "平均膜含水量与最大孔隙冰饱和度"),
            ("field_profiles_35s.png", "35 秒时厚度方向的温度与冰分布"),
            ("balance_and_ice_sensitivity.png", "水量守恒残差与冰参数敏感性"),
        ],
    )
