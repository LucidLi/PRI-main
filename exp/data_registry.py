import json
from pathlib import Path
from typing import Dict, Optional


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"


MODEL_FILE_ALIASES = {
    "llama2": "llama-2.json",
    "llama-2": "llama-2.json",
    "llama-2-7b-chat": "llama-2.json",
    "llama-2-7b-chat-hf": "llama-2.json",
    "llama3": "llama-3.json",
    "llama-3": "llama-3.json",
    "llama-3-8b": "llama-3.json",
    "llama-3-8b-instruct": "llama-3.json",
    "meta-llama-3-8b-instruct": "llama-3.json",
    "mistral": "mistral.json",
    "mistral-7b": "mistral.json",
    "mistral-7b-instruct-v0.2": "mistral.json",
    "vicuna": "vicuna-7b.json",
    "vicuna-7b": "vicuna-7b.json",
    "vicuna-7b-v1.5": "vicuna-7b.json",
    "vicuna-13b": "vicuna-13b.json",
    "vicuna-13b-v1.5": "vicuna-13b.json",
}


MODEL_DEPENDENT_ATTACKS = {
    "gcg": DATA_DIR / "gcg",
    "autodan": DATA_DIR / "autodan",
    "saa": DATA_DIR / "saa",
    "pair": DATA_DIR / "pair",
    "drattack": DATA_DIR / "drattack",
}


MODEL_INDEPENDENT_ATTACKS = {
    "advbench": DATA_DIR / "advbench_full_520.json",
    "advbench_full": DATA_DIR / "advbench_full_520.json",
    "advbench_full_520": DATA_DIR / "advbench_full_520.json",
    "hex_phi": DATA_DIR / "HEx_PHI_full_330.json",
    "hex-phi": DATA_DIR / "HEx_PHI_full_330.json",
    "hex_phi_full": DATA_DIR / "HEx_PHI_full_330.json",
    "hex_phi_full_330": DATA_DIR / "HEx_PHI_full_330.json",
    "hex-phi-full": DATA_DIR / "HEx_PHI_full_330.json",
    "hex-phi-full-330": DATA_DIR / "HEx_PHI_full_330.json",
    "hex_phi_full_330.json": DATA_DIR / "HEx_PHI_full_330.json",
    "hex-phi-full-330.json": DATA_DIR / "HEx_PHI_full_330.json",
    "deepinception": DATA_DIR / "deepinception_full_850.json",
    "sap30": DATA_DIR / "SAP30_full_210.json",
    "template": DATA_DIR / "Template_full_76.json",
    "gptfuzzer": DATA_DIR / "Template_full_76.json",
}


# Number of train+validation rows preceding the held-out attack split. These
# values mirror the split policy recorded in data/train/*.json. Callers select
# a named split and never need to handle source-file row indices directly.
JAILBREAK_TRAIN_VAL_COUNTS = {
    "gcg": 312,
    "saa": 312,
    "autodan": 312,
    "pair": 312,
    "drattack": 312,
    "deepinception": 312,
    "template": 46,
    "gptfuzzer": 46,
    "sap30": 126,
}


def _load_json(path: Path):
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "data" in data:
        data = data["data"]
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list or a dict with key 'data': {path}")
    return data


def resolve_local_path(path: Path):
    path = Path(path)
    if path.is_absolute():
        return path
    project_candidate = (ROOT_DIR / path).resolve()
    if project_candidate.exists():
        return project_candidate
    return (Path.cwd() / path).resolve()


def canonical_attack_name(attack: str):
    return str(attack).strip().lower().replace(" ", "_")


def model_file_for(model_name: str):
    key = str(model_name).strip().lower()
    if key not in MODEL_FILE_ALIASES:
        raise KeyError(
            f"Unsupported model name for model-dependent attack data: {model_name}. "
            f"Known aliases: {', '.join(sorted(MODEL_FILE_ALIASES))}"
        )
    return MODEL_FILE_ALIASES[key]


def resolve_attack_path(attack: str, model_name: Optional[str] = None):
    attack_key = canonical_attack_name(attack)
    if attack_key in MODEL_INDEPENDENT_ATTACKS:
        path = MODEL_INDEPENDENT_ATTACKS[attack_key]
    elif attack_key in MODEL_DEPENDENT_ATTACKS:
        if model_name is None:
            raise ValueError(f"--model-name is required for model-dependent attack {attack}.")
        path = MODEL_DEPENDENT_ATTACKS[attack_key] / model_file_for(model_name)
    else:
        candidate = Path(attack)
        if candidate.exists():
            path = candidate
        else:
            raise KeyError(f"Unknown attack {attack!r}.")
    if not path.exists():
        raise FileNotFoundError(f"Attack data file does not exist: {path}")
    return path


def normalize_sample(record: Dict, attack: str, model_name: Optional[str], source_path: Path, row_index: int = 0):
    h_prompt = record.get("H_prompt") or record.get("goal") or record.get("prompt") or ""
    j_prompt = record.get("J_prompt") or record.get("jailbreak") or None
    b_prompt = record.get("B_prompt")
    instruction = j_prompt or h_prompt
    if not instruction:
        raise ValueError(f"Sample has neither J_prompt nor H_prompt: {record}")

    sample_id = record.get("id", record.get("old_id"))
    if sample_id is None:
        sample_id = 0

    normalized = {
        "sample_uid": f"{canonical_attack_name(attack)}:{int(row_index)}",
        "row_index": int(row_index),
        "id": int(sample_id) if isinstance(sample_id, int) or str(sample_id).isdigit() else sample_id,
        "old_id": record.get("old_id"),
        "attack": canonical_attack_name(attack),
        "model_name": model_name,
        "method": record.get("method", canonical_attack_name(attack)),
        "model": record.get("model"),
        "category": record.get("category"),
        "target": record.get("target"),
        "H_prompt": h_prompt,
        "B_prompt": b_prompt,
        "J_prompt": j_prompt or h_prompt,
        "instruction": instruction,
        "source_path": str(source_path),
    }
    for key, value in record.items():
        if key not in normalized and key not in {"goal", "prompt", "jailbreak"}:
            normalized[key] = value
    return normalized


def load_attack_samples(
    attack: str,
    model_name: Optional[str],
    sample_index: int = 0,
    num_samples: Optional[int] = None,
    dataset_path: Optional[Path] = None,
    split: str = "all",
):
    source_path = resolve_local_path(dataset_path) if dataset_path is not None else resolve_attack_path(attack, model_name)
    raw_records = _load_json(source_path)
    split_name = str(split).strip().lower()
    if split_name not in {"all", "test"}:
        raise ValueError("split must be either 'all' or 'test'.")
    source_offset = 0
    attack_key = canonical_attack_name(attack)
    if split_name == "test" and attack_key in JAILBREAK_TRAIN_VAL_COUNTS:
        source_offset = int(JAILBREAK_TRAIN_VAL_COUNTS[attack_key])
        raw_records = raw_records[source_offset:]
    samples = [
        normalize_sample(record, attack, model_name, source_path, row_index=source_offset + index)
        for index, record in enumerate(raw_records)
    ]
    if sample_index < 0:
        raise ValueError("sample_index must be non-negative.")
    end = None if num_samples is None or int(num_samples) < 0 else sample_index + int(num_samples)
    return samples[sample_index:end]


def available_attacks():
    return sorted(set(MODEL_INDEPENDENT_ATTACKS) | set(MODEL_DEPENDENT_ATTACKS))
