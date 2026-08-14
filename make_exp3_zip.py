import zipfile
import os

files = [
    ("trainer_mentalbert_daic.py", "trainer_mentalbert_daic.py"),
    ("daic_records.parquet", "daic_records.parquet"),
    ("trainer_outputs/baseline_cv/fold_manifest.json",
     "trainer_outputs/baseline_cv/fold_manifest.json"),
    ("trainer_outputs/baseline_cv/trivial_control_arm.csv",
     "trainer_outputs/baseline_cv/trivial_control_arm.csv"),
    ("trainer_outputs/baseline_cv/baseline_cv_summary.json",
     "trainer_outputs/baseline_cv/baseline_cv_summary.json"),
    ("trainer_outputs/baseline_cv/baseline_cv_summary.md",
     "trainer_outputs/baseline_cv/baseline_cv_summary.md"),
]

files += [
    (f"trainer_outputs/baseline_cv/runs/{f}",
     f"trainer_outputs/baseline_cv/runs/{f}")
    for f in os.listdir("trainer_outputs/baseline_cv/runs")
]

files += [
    (f"exp3_convergence/{f}",
     f"exp3_convergence/{f}")
    for f in os.listdir("exp3_convergence")
    if f.endswith(".py")
]

with zipfile.ZipFile("exp3_colab.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for src, arc in files:
        z.write(src, arcname=arc)

print(f"{len(files)} files -> exp3_colab.zip")