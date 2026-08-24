import os
import json
import time
import uuid

from openai import OpenAI

# tqdm 可选，没装也能跑
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x

# 配置
MODEL = os.getenv("JUDGE_MODEL", "gpt-4o")
QUESTION_FILE = "data/mt_bench/question.jsonl"
OUT_FILE = f"data/mt_bench/reference_answer/{MODEL}_1.jsonl"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")



if not OPENAI_API_KEY:
    raise SystemExit("OPENAI_API_KEY is not set")

client_kwargs = {"api_key": OPENAI_API_KEY}
if OPENAI_BASE_URL:
    client_kwargs["base_url"] = OPENAI_BASE_URL
client = OpenAI(**client_kwargs)

# 与 MT-bench 一致的温度配置
temperature_config = {
    "writing": 0.7,
    "roleplay": 0.7,
    "extraction": 0.0,
    "math": 0.0,
    "coding": 0.0,
    "reasoning": 0.0,
    "stem": 0.1,
    "humanities": 0.1,
    "arena-hard-200": 0.0,
}

def get_temperature(q):
    if "required_temperature" in q:
        return q["required_temperature"]
    return temperature_config.get(q.get("category", ""), 0.7)

def chat(messages, temperature, max_retry=6):
    for i in range(max_retry):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=temperature,
            )
            return resp.choices[0].message.content
        except Exception:
            wait = min(2 ** i, 10)
            time.sleep(wait)
    raise RuntimeError("Failed to get completion after retries")

# 读取问题
with open(QUESTION_FILE, "r") as f:
    questions = [json.loads(line) for line in f if line.strip()]

os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

with open(OUT_FILE, "w") as fout:
    for q in tqdm(questions, desc="Generating reference answers", total=len(questions)):
        temp = get_temperature(q)
        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        turns = []

        for turn in q["turns"]:
            messages.append({"role": "user", "content": turn})
            out = chat(messages, temp)
            messages.append({"role": "assistant", "content": out})
            turns.append(out)

        ans = {
            "question_id": q["question_id"],
            "answer_id": str(uuid.uuid4()),
            "model_id": MODEL,
            "choices": [{"index": 0, "turns": turns}],
            "tstamp": time.time(),
        }
        fout.write(json.dumps(ans, ensure_ascii=False) + "\n")

print(f"Generated reference answers -> {OUT_FILE}")
