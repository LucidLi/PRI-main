"""Automatically judge model answers in single or pairwise mode.

Example:
python gen_judgment.py --model-list <model_list> --parallel <workers> --mode <single|pairwise-baseline|pairwise-all>

Inputs:
1) Question file: data/<bench_name>/question.jsonl
2) Candidate-answer directory: data/<bench_name>/model_answer
3) Reference-answer directory: data/<bench_name>/reference_answer
4) Judge prompts: data/judge_prompts.jsonl

Output:
1) Judgment file: data/<bench_name>/model_judgment/*.jsonl
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from tqdm import tqdm


# Compatibility patch for openai>=1.x and the legacy FastChat interface.
try:
    import openai
    from openai import OpenAI
    import fastchat.llm_judge.common as _fc_common

    # OpenAI client configuration.
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")

    openai_key = OPENAI_API_KEY
    openai_base_url = OPENAI_BASE_URL

    if not openai_key:
        raise ValueError("OPENAI_API_KEY is not set.")

    # Provide the legacy exception namespace expected by older FastChat code.
    if not hasattr(openai, "error"):
        class _OpenAIErrorWrapper:
            OpenAIError = openai.OpenAIError

        openai.error = _OpenAIErrorWrapper

    # OpenAI client instance.
    _client_kwargs = {"api_key": openai_key}
    if openai_base_url:
        _client_kwargs["base_url"] = openai_base_url
    _openai_client = OpenAI(**_client_kwargs)

    def _chat_compeletion_openai(model, conv, temperature=0, max_tokens=2048):
        """Compatibility wrapper for FastChat chat-completion calls.

        The conversation object is converted to OpenAI messages and the
        generated text is returned.
        """
        # Standard OpenAI Chat API message format.
        messages = conv.to_openai_api_messages()

        resp = _openai_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )

        return resp.choices[0].message.content

    # Replace the matching FastChat entry point with the compatibility wrapper.
    _fc_common.chat_compeletion_openai = _chat_compeletion_openai
    print("[PATCH] OpenAI compatibility patch loaded")
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
    """Build matches for pairwise comparison against a baseline model.

    Each model is compared with the baseline for every question. Matches with
    the same model name as the baseline are skipped, and reference answers are
    attached when provided.
    """
    # List of MatchPair objects.
    matches = []

    for q in questions:
        # In multi-turn mode, accept only questions with exactly two turns.
        if multi_turn and len(q["turns"]) != 2:
            continue

        for i in range(len(models)):
            q_id = q["question_id"]  # int or str, depending on the data file.
            m_1 = models[i]           # Candidate model name.
            m_2 = baseline_model      # Baseline model name.

            if m_1 == m_2:
                continue

            a_1 = model_answers[m_1][q_id]  # Candidate model answer record.
            a_2 = model_answers[baseline_model][q_id]  # Baseline answer record.

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
    """Build all pairwise matches among the supplied models.

    Unlike ``make_match``, this function does not require a baseline model and
    creates all C(n, 2) model pairs.
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
    """Build single-answer scoring tasks.

    The arguments are similar to ``make_match``; ``baseline_model`` is kept
    for interface compatibility but is not used.
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
    """Build the four judge variants required for pairwise evaluation.

    The returned mapping contains default, math, default-mt, and math-mt
    judges.
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
    """Build the four judge variants required for single-answer evaluation."""
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

    # Basic path parameters.
    parser.add_argument("--bench-name", type=str, default="mt_bench", help="Benchmark name.")
    parser.add_argument("--judge-file", type=str, default="data/judge_prompts.jsonl", help="Judge prompt file.")

    # Judge model and mode parameters.
    parser.add_argument("--judge-model", type=str, default="gpt-4o")
    parser.add_argument("--baseline-model", type=str, default="gpt-3.5-turbo")
    parser.add_argument(
        "--mode",
        type=str,
        default="single",
        help=(
            "Evaluation mode: "
            "pairwise-baseline=compare against a baseline; "
            "pairwise-all=compare all model pairs; "
            "single=score individual answers."
        ),
    )

    # model-list uses nargs='+', allowing multiple model names on the command line.
    parser.add_argument("--model-list", type=str, default=["vicuna-7b_Self-Exam"], nargs="+", help="Models to evaluate.")
    parser.add_argument("--model-id", type=str, default="vicuna-7b_Self-Exam", help="Model identifier in the output file.")

    # Execution control parameters.
    parser.add_argument("--parallel", type=int, default=1, help="Number of concurrent API calls.")
    parser.add_argument("--first-n", type=int, default=80, help="Debug option: evaluate only the first n questions.")

    args = parser.parse_args()

    # Path variables are strings.
    question_file = f"data/{args.bench_name}/question.jsonl"
    answer_dir = f"data/{args.bench_name}/model_answer"
    ref_answer_dir = f"data/{args.bench_name}/reference_answer"

    # questions：List[Dict]
    questions = load_questions(question_file, None, None)

    # model_answers/ref_answers：Dict[model_name, Dict[question_id, answer_record]]
    model_answers = load_model_answers(answer_dir)
    ref_answers = load_model_answers(ref_answer_dir)

    # Judge prompt templates keyed by template name.
    judge_prompts = load_judge_prompts(args.judge_file)

    if args.first_n:
        questions = questions[: args.first_n]

    # models：List[str]
    if args.model_list is None:
        models = get_model_list(answer_dir)
    else:
        models = args.model_list

    # Select the following according to mode:
    # 1) judge-construction function
    # 2) match-execution function
    # 3) output-file suffix
    # 4) whether a baseline model is used
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

    # Verify that questions, answers, references, and judges are compatible.
    check_data(questions, model_answers, ref_answers, models, judges)

    # Split questions into reference-answer categories and other categories.
    question_math = [q for q in questions if q["category"] in NEED_REF_CATS]
    question_default = [q for q in questions if q["category"] not in NEED_REF_CATS]

    # List[MatchSingle] or List[MatchPair], depending on mode.
    matches = []

    # Single-turn general questions.
    matches += make_match_func(
        question_default,
        models,
        model_answers,
        judges["default"],
        baseline_model,
    )

    # Single-turn math or reference-answer questions.
    matches += make_match_func(
        question_math,
        models,
        model_answers,
        judges["math"],
        baseline_model,
        ref_answers,
    )

    # Multi-turn general questions; process only questions with two turns.
    matches += make_match_func(
        question_default,
        models,
        model_answers,
        judges["default-mt"],
        baseline_model,
        multi_turn=True,
    )

    # Multi-turn math or reference-answer questions.
    matches += make_match_func(
        question_math,
        models,
        model_answers,
        judges["math-mt"],
        baseline_model,
        ref_answers,
        multi_turn=True,
    )

    # Execution summary for logging.
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

    # Run judgments serially when parallel=1 and with a thread pool otherwise.
    if args.parallel == 1:
        for match in tqdm(matches):
            play_a_match_func(match, output_file=output_file)
    else:
        def play_a_match_wrapper(match):
            play_a_match_func(match, output_file=output_file)

        # Shuffle after fixing the seed to reduce concurrent request hotspots.
        np.random.seed(0)
        np.random.shuffle(matches)

        with ThreadPoolExecutor(args.parallel) as executor:
            for _ in tqdm(executor.map(play_a_match_wrapper, matches), total=len(matches)):
                pass
