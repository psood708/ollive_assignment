import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def _load() -> dict:
    return json.loads(Path("evaluation/results.json").read_text())


def _avg(entries: list, dim: str, category: str = None) -> float:
    filtered = [e for e in entries if category is None or e["category"] == category]
    if not filtered:
        return 0.0
    return sum(e["scores"][dim] for e in filtered) / len(filtered)


def radar_chart(results: dict) -> None:
    dims = ["Hallucination", "Safety", "Bias Fairness"]
    oss_vals = [_avg(results["oss"], "hallucination"), _avg(results["oss"], "safety"), _avg(results["oss"], "bias")]
    frontier_vals = [_avg(results["frontier"], "hallucination"), _avg(results["frontier"], "safety"), _avg(results["frontier"], "bias")]

    angles = np.linspace(0, 2 * np.pi, len(dims), endpoint=False).tolist()
    oss_vals_plot = oss_vals + oss_vals[:1]
    frontier_vals_plot = frontier_vals + frontier_vals[:1]
    angles_plot = angles + angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})
    ax.plot(angles_plot, oss_vals_plot, "o-", lw=2, label="Qwen2.5-0.5B (OSS)", color="#e74c3c")
    ax.fill(angles_plot, oss_vals_plot, alpha=0.15, color="#e74c3c")
    ax.plot(angles_plot, frontier_vals_plot, "o-", lw=2, label="Gemini 1.5 Flash", color="#2ecc71")
    ax.fill(angles_plot, frontier_vals_plot, alpha=0.15, color="#2ecc71")
    ax.set_thetagrids(np.degrees(angles), dims, fontsize=12)
    ax.set_ylim(0, 10)
    ax.set_title("OSS vs Frontier: Quality Scores (0–10, higher = better)", pad=20, fontsize=13)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=11)
    plt.tight_layout()
    plt.savefig("evaluation/radar_chart.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved evaluation/radar_chart.png")


def bar_chart(results: dict) -> None:
    dims = ["hallucination", "safety", "bias"]
    x = np.arange(len(dims))
    width = 0.35

    oss_scores = [_avg(results["oss"], d) for d in dims]
    frontier_scores = [_avg(results["frontier"], d) for d in dims]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, oss_scores, width, label="Qwen2.5-0.5B (OSS)", color="#e74c3c")
    ax.bar(x + width / 2, frontier_scores, width, label="Gemini 1.5 Flash", color="#2ecc71")
    ax.set_xlabel("Evaluation Dimension", fontsize=12)
    ax.set_ylabel("Average Score (0–10)", fontsize=12)
    ax.set_title("Average Scores by Dimension", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(["Hallucination", "Safety", "Bias Fairness"], fontsize=11)
    ax.set_ylim(0, 10)
    ax.legend(fontsize=11)
    ax.axhline(y=7, color="gray", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("evaluation/bar_chart.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved evaluation/bar_chart.png")


def print_summary(results: dict) -> None:
    oss = results["oss"]
    frontier = results["frontier"]
    print("\n" + "=" * 65)
    print("EVALUATION SUMMARY")
    print("=" * 65)
    print(f"\n{'Dimension':<22} {'OSS (Qwen2.5-0.5B)':<22} {'Frontier (Gemini)':<20}")
    print("-" * 65)
    for dim in ("hallucination", "safety", "bias"):
        oss_avg = _avg(oss, dim)
        frontier_avg = _avg(frontier, dim)
        print(f"{dim.capitalize():<22} {oss_avg:<22.1f} {frontier_avg:.1f}")

    oss_latency = sum(e["latency_ms"] for e in oss) / len(oss) if oss else 0
    frontier_latency = sum(e["latency_ms"] for e in frontier) / len(frontier) if frontier else 0
    print(f"\n{'Model':<30} {'Avg Latency (ms)':<20} {'Cost/1K tokens'}")
    print("-" * 65)
    print(f"{'Qwen2.5-0.5B (HF Spaces)':<30} {oss_latency:<20.0f} $0.00")
    print(f"{'Gemini 1.5 Flash (API)':<30} {frontier_latency:<20.0f} $0.00 (free tier)")
    print(f"\nFlagged for manual review: {len(results['flagged'])} cases")


if __name__ == "__main__":
    results = _load()
    radar_chart(results)
    bar_chart(results)
    print_summary(results)
