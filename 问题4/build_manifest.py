"""Create a SHA-256 inventory after code, results and report are final."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    paths = [
        HERE / "dynamic_model.py",
        HERE / "optimize.py",
        HERE / "verify.py",
        HERE / "build_report.py",
        HERE / "scan_precooling.py",
        HERE / "run.py",
        HERE / "build_manifest.py",
        HERE / "README.md",
        HERE / "问题4的解决方案与结果分析.md",
        ROOT / "问题1" / "data.py",
        ROOT / "问题1" / "model.py",
        ROOT / "问题1" / "run.py",
        ROOT / "问题2" / "stack_model.py",
        ROOT / "问题3" / "aux_model.py",
        ROOT / "问题3" / "results" / "cooperative_search.json",
        ROOT / "result" / "metrics.json",
        ROOT / "附件" / "附件1.xlsx",
        ROOT / "附件" / "附件2.xlsx",
        ROOT / "氢燃料电池低温冷启动建模与控制策略研究.pdf",
    ]
    paths.extend(sorted((HERE / "results").glob("*")))
    paths = [path for path in paths
             if path.is_file() and path.name != "run_manifest.json"]
    manifest = {
        "status": "completed_q4_model_prediction",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "command": f"{sys.executable} 问题4/run.py",
        "python": sys.version,
        "platform": platform.platform(),
        "working_directory": str(ROOT),
        "model_lineage": [
            "问题1 calibrated 1-D multiphase cell",
            "问题2 five-cell/end-plate thermal network",
            "问题3 prescribed current ramp and constant-power baseline",
            "问题4 pre-cooling field and five-objective Pareto feedback",
        ],
        "files": [
            {
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "bytes": path.stat().st_size,
                "sha256": digest(path),
            }
            for path in paths
        ],
    }
    target = HERE / "results" / "run_manifest.json"
    target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    print(target)


if __name__ == "__main__":
    raise SystemExit(main())
