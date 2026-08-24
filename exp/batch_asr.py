"""Batch launcher for attack-success-rate experiments.

The single-run evaluation lives in ``evaluate_asr.py``. This file expands a
model/attack grid and starts one isolated process per setting.
"""

import argparse
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
SINGLE_RUN_ENTRY = Path(__file__).resolve().parent / "evaluate_asr.py"
DEFAULT_MODELS = "vicuna-7b,llama-2,mistral-7b,llama-3,vicuna-13b"
DEFAULT_ATTACKS = "gcg,saa,autodan,pair,drattack,template,deepinception,sap30"


def str2bool(value):
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "on"}:
        return True
    if text in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def comma_separated(value: str):
    return [item.strip() for item in str(value).split(",") if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser("Run PRI ASR evaluation over a model/attack grid.")
    parser.add_argument("--models", type=comma_separated, default=comma_separated(DEFAULT_MODELS))
    parser.add_argument("--attacks", type=comma_separated, default=comma_separated(DEFAULT_ATTACKS))
    parser.add_argument("--data-split", type=str, default="test", choices=["test", "all"])
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--do-sample", type=str2bool, default=False)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--auto-gpu-memory", type=str, default="22GiB")
    parser.add_argument("--auto-cpu-memory", type=str, default="64GiB")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", type=Path, default=ROOT_DIR / "results" / "ASR")
    parser.add_argument("--training-root", type=Path, default=ROOT_DIR / "training_results")
    parser.add_argument("--prefill-num-tokens", type=int, default=None)
    parser.add_argument("--td-topk-k", type=int, default=None)
    parser.add_argument("--pri-j-only", type=str2bool, default=True)
    parser.add_argument("--pri-j-consecutive-trigger", type=int, default=2)
    parser.add_argument("--eval-mode", type=str2bool, default=True)
    parser.add_argument("--disable-GPT-judge", action="store_true")
    parser.add_argument(
        "--GPT_API",
        type=str,
        default=None,
        help="GPT judge API key forwarded to each evaluation process.",
    )
    parser.add_argument("--judge-model", type=str, default="gpt-4o-mini")
    parser.add_argument("--multi-processing", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_command(args, model_name: str, attack: str):
    command = [
        sys.executable,
        str(SINGLE_RUN_ENTRY),
        "--model-name",
        model_name,
        "--attack",
        attack,
        "--defense",
        "pri",
        "--num-samples",
        str(args.num_samples),
        "--data-split",
        args.data_split,
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--do-sample",
        str(args.do_sample).lower(),
        "--device",
        args.device,
        "--auto-gpu-memory",
        args.auto_gpu_memory,
        "--auto-cpu-memory",
        args.auto_cpu_memory,
        "--seed",
        str(args.seed),
        "--output-root",
        str(args.output_root),
        "--training-root",
        str(args.training_root),
        "--pri-j-only",
        str(args.pri_j_only).lower(),
        "--pri-j-consecutive-trigger",
        str(args.pri_j_consecutive_trigger),
        "--eval-mode",
        str(args.eval_mode).lower(),
        "--judge-model",
        args.judge_model,
        "--multi-processing",
        str(args.multi_processing),
    ]
    if args.top_p is not None:
        command.extend(["--top-p", str(args.top_p)])
    if args.prefill_num_tokens is not None:
        command.extend(["--prefill-num-tokens", str(args.prefill_num_tokens)])
    if args.td_topk_k is not None:
        command.extend(["--td-topk-k", str(args.td_topk_k)])
    if args.disable_GPT_judge:
        command.append("--disable-GPT-judge")
    if args.GPT_API:
        command.extend(["--GPT_API", args.GPT_API])
    return command


def main():
    args = parse_args()
    for model_name in args.models:
        for attack in args.attacks:
            command = build_command(args, model_name, attack)
            printable_command = list(command)
            if "--GPT_API" in printable_command:
                key_index = printable_command.index("--GPT_API") + 1
                if key_index < len(printable_command):
                    printable_command[key_index] = "[REDACTED]"
            print("[PRI]", " ".join(printable_command), flush=True)
            if not args.dry_run:
                subprocess.run(command, cwd=ROOT_DIR, check=True)


if __name__ == "__main__":
    main()
