import subprocess
import sys
import os


def run_script(script_name):
    print(f"\n{'=' * 60}")
    print(f"🚀 EXECUTING: {script_name}")
    print(f"{'=' * 60}\n")

    try:
        # sys.executable ensures the runner uses your exact virtual environment
        subprocess.run([sys.executable, script_name], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] {script_name} failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"\n[ERROR] Could not find script: {script_name}")
        return False


if __name__ == "__main__":
    print("Booting Master Edge AI Benchmark Suite...\n")

    # Define the sequential execution pipeline
    test_pipeline = [
        "eval_summary.py", # Tests Model 1
        "evaluate_quantized.py",  # Tests Model 2 Accuracy & Latency (logs to CSV)
        "hardware_profiler.py",  # Tests Model 2 Memory constraints & P99 Latency
        "gpu_benchmark.py" #Tests GPU benchmark
    ]

    # Intelligently locate Model 3's evaluation script based on your file tree
    if os.path.exists("../Trash/Evaluation3.py"):
        test_pipeline.insert(0, "Evaluation3.py")
    elif os.path.exists(os.path.join("../training", "Evaluation3.py")):
        test_pipeline.insert(0, os.path.join("../training", "Evaluation3.py"))

    # Execute the pipeline
    suite_success = True
    for script in test_pipeline:
        if not run_script(script):
            suite_success = False
            break  # Halt the pipeline if a test crashes

    print(f"\n{'=' * 60}")
    if suite_success:
        print("🏆 SUITE COMPLETE: All automated benchmarks passed.")
        print("Check 'benchmarks/benchmark_results.csv' for the centralized logs.")
    else:
        print("❌ SUITE FAILED: Review the error logs above.")
    print(f"{'=' * 60}\n")