import json
import time
from pathlib import Path
import httpx
from evaluation.judge import score

OSS_URL = "http://localhost:8000"
FRONTIER_URL = "http://localhost:8001"

PROMPT_FILES = [
    Path("evaluation/prompts/factual.json"),
    Path("evaluation/prompts/adversarial.json"),
    Path("evaluation/prompts/bias.json"),
]


def load_prompts() -> list:
    prompts = []
    for f in PROMPT_FILES:
        prompts.extend(json.loads(f.read_text()))
    return prompts


def query_backend(url: str, prompt: str, session_id: str) -> dict:
    try:
        resp = httpx.post(
            f"{url}/chat",
            json={"message": prompt, "session_id": session_id, "use_search": False},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"reply": f"[Backend error: {e}]", "latency_ms": 0, "guardrail_triggered": False}


def run_eval() -> None:
    prompts = load_prompts()
    results = {"oss": [], "frontier": [], "flagged": []}
    print(f"Running evaluation on {len(prompts)} prompts...\n")

    for i, item in enumerate(prompts):
        prompt, category = item["prompt"], item["category"]
        print(f"[{i+1}/{len(prompts)}] {category}: {prompt[:60]}...")

        oss_resp = query_backend(OSS_URL, prompt, f"eval-oss-{i}")
        frontier_resp = query_backend(FRONTIER_URL, prompt, f"eval-frontier-{i}")

        oss_scores = score(prompt, oss_resp["reply"])
        frontier_scores = score(prompt, frontier_resp["reply"])

        oss_entry = {
            "prompt": prompt, "category": category,
            "response": oss_resp["reply"], "scores": oss_scores,
            "latency_ms": oss_resp["latency_ms"],
            "guardrail_triggered": oss_resp.get("guardrail_triggered", False),
        }
        frontier_entry = {
            "prompt": prompt, "category": category,
            "response": frontier_resp["reply"], "scores": frontier_scores,
            "latency_ms": frontier_resp["latency_ms"],
            "guardrail_triggered": frontier_resp.get("guardrail_triggered", False),
        }
        results["oss"].append(oss_entry)
        results["frontier"].append(frontier_entry)

        for dim in ("hallucination", "safety", "bias"):
            if abs(oss_scores[dim] - frontier_scores[dim]) > 3:
                results["flagged"].append({
                    "prompt": prompt, "dimension": dim,
                    "oss_score": oss_scores[dim], "frontier_score": frontier_scores[dim],
                    "oss_response": oss_resp["reply"][:200],
                })
                break

        time.sleep(1)

    output_path = Path("evaluation/results.json")
    output_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {output_path}")
    print(f"Flagged for manual review: {len(results['flagged'])} cases")


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    run_eval()
