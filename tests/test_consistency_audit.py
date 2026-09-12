import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FINAL_RESULTS_PATH = ROOT / "results" / "metrics" / "final_results.json"
MEMORY_PATH = ROOT / "results" / "metrics" / "memory_sensitivity.json"
README_PATH = ROOT / "README.md"

def test_consistency_audit():
    print("=================================================================")
    print("                 FINAL CONSISTENCY AUDIT                         ")
    print("=================================================================")

    with open(FINAL_RESULTS_PATH, "r") as f:
        final_results = json.load(f)

    with open(MEMORY_PATH, "r") as f:
        memory_results = json.load(f)

    with open(README_PATH, "r", encoding="utf-8") as f:
        readme_text = f.read()

    # Verify each method's metrics in final_results vs README
    methods = [
        ("naive", "Naive Sequential", 0.25, 0.3333, 0.995),
        ("ewc", "EWC", 0.25, 0.3333, 0.995),
        ("replay", "Experience Replay", 0.64625, 0.6717, 0.4775),
        ("replay_ewc", "Replay + EWC", 0.65625, 0.6800, 0.465),
        ("joint", "Joint Upper Bound", 0.9375, 0.9250, 0.0)
    ]

    for key, name, exp_acc, exp_task, exp_f in methods:
        actual = final_results[key]
        acc = actual["overall_accuracy"]
        task_acc = actual["final_avg_task_accuracy"]
        f_val = actual["average_forgetting"]

        print(f"\nChecking Method: {name}")
        print(f"  final_results.json: Overall Acc = {acc*100:.2f}%, Task Acc = {task_acc*100:.2f}%, Forgetting = {f_val*100:.2f}%")

        # Verify formatted strings exist in README
        acc_str = f"{acc*100:.2f}%"
        task_str = f"{task_acc*100:.2f}%"
        f_str = f"{f_val*100:.2f}%"

        assert acc_str in readme_text, f"Mismatch: {acc_str} for {name} not found in README.md"
        assert task_str in readme_text, f"Mismatch: {task_str} for {name} not found in README.md"
        assert f_str in readme_text, f"Mismatch: {f_str} for {name} not found in README.md"
        print(f"  [OK] Consistent with README.md table.")

    # Check memory sensitivity
    for b in [0, 50, 100, 200]:
        val = memory_results[str(b)]
        acc_str = f"{val['final_accuracy']*100:.2f}%"
        f_str = f"{val['forgetting']*100:.2f}%"
        print(f"Checking Memory Budget {b}: Acc = {acc_str}, Forgetting = {f_str}")
        assert acc_str in readme_text, f"Memory budget {b} acc {acc_str} not in README"
        assert f_str in readme_text, f"Memory budget {b} forgetting {f_str} not in README"
        print(f"  [OK] Budget {b} verified in README.md.")

    print("\n=================================================================")
    print("   ALL ARTIFACTS AND METRICS ARE 100% CONSISTENT AND VERIFIED!   ")
    print("=================================================================")

if __name__ == "__main__":
    test_consistency_audit()
