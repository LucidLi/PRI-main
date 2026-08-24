"""本脚本用于对模型答案执行自动评判（single / pairwise）。

典型用法：
python gen_judgment.py --model-list <模型列表> --parallel <并发数> --mode <single|pairwise-baseline|pairwise-all>

输入：
1) 题目文件 data/<bench_name>/question.jsonl
2) 候选答案目录 data/<bench_name>/model_answer
3) 参考答案目录 data/<bench_name>/reference_answer
4) 评审提示词 data/judge_prompts.jsonl

输出：
1) 评审结果文件 data/<bench_name>/model_judgment/*.jsonl
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from tqdm import tqdm


# 兼容补丁：适配 openai>=1.x 的调用方式，并替换 FastChat 内部旧接口。
try:
    import openai
    from openai import OpenAI
    import fastchat.llm_judge.common as _fc_common

    # OpenAI 访问配置
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")

    openai_key = OPENAI_API_KEY
    openai_base_url = OPENAI_BASE_URL

    if not openai_key:
        raise ValueError("OPENAI_API_KEY 未设置")

    # 兼容旧版异常命名：fastchat 旧逻辑可能访问 openai.error.OpenAIError。
    if not hasattr(openai, "error"):
        class _OpenAIErrorWrapper:
            OpenAIError = openai.OpenAIError

        openai.error = _OpenAIErrorWrapper

    # _openai_client：OpenAI 客户端实例。
    _client_kwargs = {"api_key": openai_key}
    if openai_base_url:
        _client_kwargs["base_url"] = openai_base_url
    _openai_client = OpenAI(**_client_kwargs)

    def _chat_compeletion_openai(model, conv, temperature=0, max_tokens=2048):
        """供 FastChat 调用的 chat completion 兼容函数。

        输入：
        - model: str，评审模型名（如 gpt-4o）。
        - conv: FastChat 对话对象，可转 OpenAI messages。
        - temperature: float，采样温度。
        - max_tokens: int，保留参数；当前调用保持与原逻辑一致，不强制传入。

        输出：
        - str，模型回复文本（choices[0].message.content）。
        """
        # messages：List[Dict[str, str]]，OpenAI Chat API 的标准消息格式。
        messages = conv.to_openai_api_messages()

        resp = _openai_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )

        return resp.choices[0].message.content

    # 用兼容函数覆盖 fastchat 内部同名入口。
    _fc_common.chat_compeletion_openai = _chat_compeletion_openai
    print("[PATCH] OpenAI 兼容补丁已加载")
except Exception as e:
    print("[PATCH ERROR]", e)


from fastchat.llm_judge.common import (
    load_questions,
    load_model_answers,
    load_judge_prompts,
    check_data,
    play_a_match_pair,
    play_a_match_single,
    get_model_list,
    Judge,
    MatchPair,
    MatchSingle,
    NEED_REF_CATS,
)


def make_match(questions, models, model_answers, judge, baseline_model, ref_answers=None, multi_turn=False):
    """构建 pairwise-baseline 模式下的对战列表。

    输入：
    - questions: List[Dict]，题目列表。
    - models: List[str]，待评测模型名列表。
    - model_answers: Dict[str, Dict[qid, answer]]，模型答案映射。
    - judge: Judge，评审器实例。
    - baseline_model: str，基线模型名。
    - ref_answers: Dict 或 None，参考答案映射（数学类会使用）。
    - multi_turn: bool，是否仅处理双轮题。

    输出：
    - List[MatchPair]。

    过程：
    - 每个问题中，每个模型与 baseline_model 组成一场对战。
    - 若模型名与 baseline 相同则跳过。
    - 若提供 ref_answers 则写入 ref_answer 字段。
    """
    # matches：List[MatchPair]
    matches = []

    for q in questions:
        # multi_turn=True 时仅接受恰好两轮的题目。
        if multi_turn and len(q["turns"]) != 2:
            continue

        for i in range(len(models)):
            q_id = q["question_id"]  # 类型：int 或 str（取决于数据文件）
            m_1 = models[i]           # 待评模型名，类型：str
            m_2 = baseline_model      # 基线模型名，类型：str

            if m_1 == m_2:
                continue

            a_1 = model_answers[m_1][q_id]  # 模型1答案对象
            a_2 = model_answers[baseline_model][q_id]  # 基线答案对象

            if ref_answers is not None:
                ref = ref_answers[judge.model_name][q_id]
                match = MatchPair(
                    dict(q),
                    m_1,
                    m_2,
                    a_1,
                    a_2,
                    judge,
                    ref_answer=ref,
                    multi_turn=multi_turn,
                )
            else:
                match = MatchPair(dict(q), m_1, m_2, a_1, a_2, judge, multi_turn=multi_turn)

            matches.append(match)

    return matches


def make_match_all_pairs(
    questions,
    models,
    model_answers,
    judge,
    baseline_model=None,
    ref_answers=None,
    multi_turn=False,
):
    """构建 pairwise-all 模式下的全模型两两对战列表。

    输入/输出与 make_match 类似，不同点：
    - 不依赖 baseline_model。
    - 在 models 内做组合 C(n,2)。
    """
    matches = []

    for q in questions:
        if multi_turn and len(q["turns"]) != 2:
            continue

        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                q_id = q["question_id"]
                m_1 = models[i]
                m_2 = models[j]

                a_1 = model_answers[m_1][q_id]
                a_2 = model_answers[m_2][q_id]

                if ref_answers is not None:
                    ref = ref_answers[judge.model_name][q_id]
                    match = MatchPair(
                        dict(q),
                        m_1,
                        m_2,
                        a_1,
                        a_2,
                        judge,
                        ref_answer=ref,
                        multi_turn=multi_turn,
                    )
                else:
                    match = MatchPair(dict(q), m_1, m_2, a_1, a_2, judge, multi_turn=multi_turn)

                matches.append(match)

    return matches


def make_match_single(
    questions,
    models,
    model_answers,
    judge,
    baseline_model=None,
    ref_answers=None,
    multi_turn=False,
):
    """构建 single 模式下的评分任务列表。

    输入：
    - 与 make_match 基本一致；baseline_model 参数保留但不使用。

    输出：
    - List[MatchSingle]。
    """
    matches = []

    for q in questions:
        if multi_turn and len(q["turns"]) != 2:
            continue

        for i in range(len(models)):
            q_id = q["question_id"]
            m = models[i]
            a = model_answers[m][q_id]

            if ref_answers is not None:
                ref = ref_answers[judge.model_name][q_id]
                matches.append(MatchSingle(dict(q), m, a, judge, ref_answer=ref, multi_turn=multi_turn))
            else:
                matches.append(MatchSingle(dict(q), m, a, judge, multi_turn=multi_turn))

    return matches


def make_judge_pairwise(judge_model, judge_prompts):
    """构建 pairwise 模式所需的 4 类评审器。

    输入：
    - judge_model: str，评审模型名。
    - judge_prompts: Dict[str, prompt_template]，提示词模板映射。

    输出：
    - Dict[str, Judge]，键包括：
      default / math / default-mt / math-mt。
    """
    judges = {}
    judges["default"] = Judge(judge_model, judge_prompts["pair-v2"])
    judges["math"] = Judge(judge_model, judge_prompts["pair-math-v1"], ref_based=True)
    judges["default-mt"] = Judge(judge_model, judge_prompts["pair-v2-multi-turn"], multi_turn=True)
    judges["math-mt"] = Judge(
        judge_model,
        judge_prompts["pair-math-v1-multi-turn"],
        ref_based=True,
        multi_turn=True,
    )
    return judges


def make_judge_single(judge_model, judge_prompts):
    """构建 single 模式所需的 4 类评审器。"""
    judges = {}
    judges["default"] = Judge(judge_model, judge_prompts["single-v1"])
    judges["math"] = Judge(judge_model, judge_prompts["single-math-v1"], ref_based=True)
    judges["default-mt"] = Judge(judge_model, judge_prompts["single-v1-multi-turn"], multi_turn=True)
    judges["math-mt"] = Judge(
        judge_model,
        judge_prompts["single-math-v1-multi-turn"],
        ref_based=True,
        multi_turn=True,
    )
    return judges


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # 基本路径参数
    parser.add_argument("--bench-name", type=str, default="mt_bench", help="题库名称。")
    parser.add_argument("--judge-file", type=str, default="data/judge_prompts.jsonl", help="评审提示词文件。")

    # 评审模型与模式参数
    parser.add_argument("--judge-model", type=str, default="gpt-4o")
    parser.add_argument("--baseline-model", type=str, default="gpt-3.5-turbo")
    parser.add_argument(
        "--mode",
        type=str,
        default="single",
        help=(
            "评测模式："
            "pairwise-baseline=与基线两两比较；"
            "pairwise-all=全部两两比较；"
            "single=单答案评分。"
        ),
    )

    # model-list 使用 nargs='+'，因此命令行中可传多个模型名。
    parser.add_argument("--model-list", type=str, default=["vicuna-7b_Self-Exam"], nargs="+", help="待评模型列表。")
    parser.add_argument("--model-id", type=str, default="vicuna-7b_Self-Exam", help="输出文件中的模型标识。")

    # 执行控制参数
    parser.add_argument("--parallel", type=int, default=1, help="并发 API 调用数量。")
    parser.add_argument("--first-n", type=int, default=80, help="调试参数：仅评测前 n 个问题。")

    args = parser.parse_args()

    # 路径变量类型：str
    question_file = f"data/{args.bench_name}/question.jsonl"
    answer_dir = f"data/{args.bench_name}/model_answer"
    ref_answer_dir = f"data/{args.bench_name}/reference_answer"

    # questions：List[Dict]
    questions = load_questions(question_file, None, None)

    # model_answers/ref_answers：Dict[model_name, Dict[question_id, answer_record]]
    model_answers = load_model_answers(answer_dir)
    ref_answers = load_model_answers(ref_answer_dir)

    # judge_prompts：Dict[str, Dict]（键为模板名）
    judge_prompts = load_judge_prompts(args.judge_file)

    if args.first_n:
        questions = questions[: args.first_n]

    # models：List[str]
    if args.model_list is None:
        models = get_model_list(answer_dir)
    else:
        models = args.model_list

    # 根据 mode 决定：
    # 1) judge 构建函数
    # 2) match 执行函数
    # 3) 输出文件后缀
    # 4) 基线模型是否启用
    if args.mode == "single":
        judges = make_judge_single(args.judge_model, judge_prompts)
        play_a_match_func = play_a_match_single
        output_file = f"data/{args.bench_name}/model_judgment/{args.model_id}_{args.judge_model}_single.jsonl"
        make_match_func = make_match_single
        baseline_model = None
    else:
        judges = make_judge_pairwise(args.judge_model, judge_prompts)
        play_a_match_func = play_a_match_pair
        output_file = f"data/{args.bench_name}/model_judgment/{args.model_id}_{args.judge_model}_pair.jsonl"

        if args.mode == "pairwise-all":
            make_match_func = make_match_all_pairs
            baseline_model = None
        else:
            make_match_func = make_match
            baseline_model = args.baseline_model

    # 数据一致性检查：确认题目、模型答案、参考答案、评审器可对应。
    check_data(questions, model_answers, ref_answers, models, judges)

    # 按题目类别分为：
    # - question_math：需要参考答案的类别（NEED_REF_CATS）
    # - question_default：其他类别
    question_math = [q for q in questions if q["category"] in NEED_REF_CATS]
    question_default = [q for q in questions if q["category"] not in NEED_REF_CATS]

    # matches：List[MatchSingle] 或 List[MatchPair]（取决于 mode）
    matches = []

    # 单轮普通题
    matches += make_match_func(
        question_default,
        models,
        model_answers,
        judges["default"],
        baseline_model,
    )

    # 单轮数学/参考题
    matches += make_match_func(
        question_math,
        models,
        model_answers,
        judges["math"],
        baseline_model,
        ref_answers,
    )

    # 多轮普通题（仅处理 turn 数为 2 的问题）
    matches += make_match_func(
        question_default,
        models,
        model_answers,
        judges["default-mt"],
        baseline_model,
        multi_turn=True,
    )

    # 多轮数学/参考题
    matches += make_match_func(
        question_math,
        models,
        model_answers,
        judges["math-mt"],
        baseline_model,
        ref_answers,
        multi_turn=True,
    )

    # match_stat：Dict[str, Any]，用于打印执行概览。
    match_stat = {}
    match_stat["bench_name"] = args.bench_name
    match_stat["mode"] = args.mode
    match_stat["judge"] = args.judge_model
    match_stat["baseline"] = baseline_model
    match_stat["model_list"] = models
    match_stat["total_num_questions"] = len(questions)
    match_stat["total_num_matches"] = len(matches)
    match_stat["output_path"] = output_file

    print("Stats:")
    print(json.dumps(match_stat, indent=4))

    # 执行评审：
    # - parallel=1：串行
    # - parallel>1：线程池并行
    if args.parallel == 1:
        for match in tqdm(matches):
            play_a_match_func(match, output_file=output_file)
    else:
        def play_a_match_wrapper(match):
            play_a_match_func(match, output_file=output_file)

        # 固定随机种子后打乱任务顺序，减小并发热点。
        np.random.seed(0)
        np.random.shuffle(matches)

        with ThreadPoolExecutor(args.parallel) as executor:
            for _ in tqdm(executor.map(play_a_match_wrapper, matches), total=len(matches)):
                pass
