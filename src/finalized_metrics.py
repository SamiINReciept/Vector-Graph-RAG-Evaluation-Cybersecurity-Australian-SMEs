import pandas as pd

# Load your two summary files
gpu = pd.read_csv("results/summary_metrics.csv")
cpu = pd.read_csv("results/summary_cpu_metrics.csv")

# Ensure both have "pipeline" as a common key
gpu = gpu.set_index("pipeline")
cpu = cpu.set_index("pipeline")

# Columns you want to keep from GPU + CPU
gpu_keep = ["meteor", "cosine_sim", "bertscore_f1"]
cpu_keep = ["completeness", "hallucination", "irrelevance", "latency_sec"]

# Merge and keep only selected columns
merged = pd.concat([
    gpu[gpu_keep],
    cpu[cpu_keep]
], axis=1)

# Reset index for clean CSV
merged = merged.reset_index()

# Save final student-ready CSV
merged.to_csv("results/final_metrics.csv", index=False)

print("Saved: results/final_metrics.csv")
print(merged)
