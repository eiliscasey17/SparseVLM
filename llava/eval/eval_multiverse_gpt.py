import argparse
import json
import os
import re
import statistics
import time
import urllib.error
import urllib.request

from tqdm import tqdm


DEFAULT_SYSTEM_PROMPT = (
    "You are a careful evaluator for a multimodal dialogue benchmark. "
    "Score the candidate assistant answer against the gold reference answer, "
    "using the provided scenario, user turn, and checklist. Be strict about "
    "factual correctness and whether the candidate addresses the user's actual request."
)


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def json_dumps(data):
    return json.dumps(data, ensure_ascii=False, sort_keys=True)


def normalize_checklist(checklist):
    if checklist is None:
        return []
    if isinstance(checklist, list):
        return checklist
    if isinstance(checklist, dict):
        return [f"{k}: {v}" for k, v in checklist.items()]
    return [str(checklist)]


def build_user_prompt(answer):
    checklist_lines = normalize_checklist(answer.get("checklist"))
    checklist_text = "\n".join(f"- {item}" for item in checklist_lines) if checklist_lines else "(none)"

    return (
        "Evaluate the candidate answer for this MultiVerse benchmark turn.\n\n"
        f"Character:\n{answer.get('character', '')}\n\n"
        f"Scenario:\n{answer.get('scenario', '')}\n\n"
        f"Goal:\n{answer.get('goal', '')}\n\n"
        f"User prompt:\n{answer.get('prompt') or answer.get('user_prompt', '')}\n\n"
        f"Checklist:\n{checklist_text}\n\n"
        f"Gold reference answer:\n{answer.get('reference', '')}\n\n"
        f"Candidate answer:\n{answer.get('text') or answer.get('prediction', '')}\n\n"
        "Return only a JSON object with this exact schema:\n"
        "{\n"
        '  "overall_score": integer from 1 to 10,\n'
        '  "reference_alignment": integer from 1 to 5,\n'
        '  "instruction_following": integer from 1 to 5,\n'
        '  "checklist_coverage": integer from 1 to 5,\n'
        '  "hallucination": integer from 1 to 5,\n'
        '  "verdict": "pass" or "fail",\n'
        '  "strengths": ["short string", "..."],\n'
        '  "issues": ["short string", "..."],\n'
        '  "explanation": "2-4 sentence explanation"\n'
        "}\n\n"
        "Scoring guidance:\n"
        "- 9-10: excellent, materially matches the reference and satisfies the user need.\n"
        "- 7-8: good, minor omissions or phrasing issues.\n"
        "- 5-6: partially correct but misses important specifics.\n"
        "- 3-4: weak, generic, or noticeably incorrect.\n"
        "- 1-2: fails to answer, hallucinates badly, or contradicts the reference.\n"
        "For hallucination: 5 means no meaningful hallucination; 1 means severe hallucination."
    )


def extract_json_object(text):
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            candidate = repair_json_string(candidate)
            return json.loads(candidate)

    raise ValueError("Could not parse JSON object from model response.")


def repair_json_string(text):
    # Replace raw newlines/tabs inside quoted strings with escaped forms.
    text = _escape_control_chars_inside_strings(text)
    # Remove invalid backslash escapes like \_ that models occasionally emit.
    text = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', text)
    # Remove trailing commas before closing braces/brackets.
    text = re.sub(r",\s*([}\]])", r"\1", text)
    return text


def _escape_control_chars_inside_strings(text):
    out = []
    in_string = False
    escape = False

    for ch in text:
        if in_string:
            if escape:
                out.append(ch)
                escape = False
                continue
            if ch == "\\":
                out.append(ch)
                escape = True
                continue
            if ch == '"':
                out.append(ch)
                in_string = False
                continue
            if ch == "\n":
                out.append("\\n")
                continue
            if ch == "\r":
                out.append("\\r")
                continue
            if ch == "\t":
                out.append("\\t")
                continue
            out.append(ch)
            continue

        out.append(ch)
        if ch == '"':
            in_string = True

    return "".join(out)


def call_openai_chat(api_key, model, system_prompt, user_prompt, max_tokens, temperature, base_url):
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=180) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body["choices"][0]["message"]["content"]


def score_answer(answer, args):
    user_prompt = build_user_prompt(answer)
    last_error = None

    for attempt in range(args.max_retries):
        try:
            raw_review = call_openai_chat(
                api_key=args.api_key,
                model=args.gpt_model,
                system_prompt=DEFAULT_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                base_url=args.api_base,
            )
            parsed_review = extract_json_object(raw_review)
            return raw_review, parsed_review
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ValueError, KeyError) as exc:
            last_error = exc
            if attempt + 1 < args.max_retries:
                time.sleep(args.retry_sleep)

    raise RuntimeError(f"Failed to score answer after {args.max_retries} attempts: {last_error}")


def summarize_reviews(reviews):
    numeric_fields = [
        "overall_score",
        "reference_alignment",
        "instruction_following",
        "checklist_coverage",
        "hallucination",
    ]
    summary = {
        "num_reviews": len(reviews),
        "num_successful_reviews": 0,
        "num_failed_reviews": 0,
        "pass_rate": 0.0,
        "means": {},
    }

    if not reviews:
        return summary

    successful_reviews = [review for review in reviews if isinstance(review.get("evaluation"), dict)]
    failed_reviews = [review for review in reviews if not isinstance(review.get("evaluation"), dict)]
    summary["num_successful_reviews"] = len(successful_reviews)
    summary["num_failed_reviews"] = len(failed_reviews)

    if successful_reviews:
        summary["pass_rate"] = sum(
            1 for review in successful_reviews if review["evaluation"].get("verdict", "").lower() == "pass"
        ) / len(successful_reviews)

    for field in numeric_fields:
        values = [
            review["evaluation"][field]
            for review in successful_reviews
            if isinstance(review["evaluation"].get(field), (int, float))
        ]
        if values:
            summary["means"][field] = round(statistics.mean(values), 4)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate MultiVerse generations with GPT.")
    parser.add_argument("--answers-file", type=str, required=True)
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument("--summary-file", type=str, required=True)
    parser.add_argument("--gpt-model", type=str, default="gpt-4.1-mini")
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--retry-sleep", type=float, default=3.0)
    parser.add_argument("--api-base", type=str, default=os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1"))
    parser.add_argument("--api-key", type=str, default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--skip-bad-items", action="store_true")
    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit("OPENAI_API_KEY is not set. Export it or pass --api-key.")

    answers = load_jsonl(os.path.expanduser(args.answers_file))

    output_path = os.path.expanduser(args.output_file)
    summary_path = os.path.expanduser(args.summary_file)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)

    existing_reviews = []
    completed_ids = set()
    if os.path.exists(output_path):
        existing_reviews = load_jsonl(output_path)
        completed_ids = {
            review.get("question_id") or f"{review.get('sample_id')}_{review.get('assistant_turn_id')}"
            for review in existing_reviews
        }

    new_reviews = []
    with open(output_path, "a", encoding="utf-8") as out_f:
        for answer in tqdm(answers):
            question_id = answer.get("question_id") or f"{answer['sample_id']}_{answer['assistant_turn_id']}"
            if question_id in completed_ids:
                continue

            try:
                raw_review, parsed_review = score_answer(answer, args)
                record = {
                    "question_id": question_id,
                    "sample_id": answer.get("sample_id"),
                    "turn_id": answer.get("turn_id"),
                    "assistant_turn_id": answer.get("assistant_turn_id"),
                    "answer_id": answer.get("answer_id"),
                    "model_id": answer.get("model_id"),
                    "status": "ok",
                    "evaluation": parsed_review,
                    "raw_review": raw_review,
                }
            except Exception as exc:
                if not args.skip_bad_items:
                    raise
                record = {
                    "question_id": question_id,
                    "sample_id": answer.get("sample_id"),
                    "turn_id": answer.get("turn_id"),
                    "assistant_turn_id": answer.get("assistant_turn_id"),
                    "answer_id": answer.get("answer_id"),
                    "model_id": answer.get("model_id"),
                    "status": "error",
                    "error": str(exc),
                    "evaluation": None,
                    "raw_review": None,
                }

            out_f.write(json_dumps(record) + "\n")
            out_f.flush()
            new_reviews.append(record)

    all_reviews = existing_reviews + new_reviews
    summary = summarize_reviews(all_reviews)
    summary.update({
        "answers_file": args.answers_file,
        "output_file": args.output_file,
        "gpt_model": args.gpt_model,
    })

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(new_reviews)} new reviews to {output_path}")
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
