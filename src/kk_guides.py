"""Write a fill-in solving guide for every KK train puzzle: one model call per puzzle.

    uv run --env-file .env src/kk_guides.py                          # all 2,674 items with anthropic/claude-haiku-5-5
    uv run --env-file .env src/kk_guides.py --limit 10               # a quick look first
    uv run --env-file .env src/kk_guides.py --model-args '{"max_tokens": 32000}'

The model sees fixed instructions and one worked example (the 3ppl_100 puzzle, its answer and the guide for it,
data/kk_guide_example.txt), then the puzzle and its answer, and replies with the guide. The system part is identical for every
request, so Anthropic models cache it. Nothing is validated: each response is written as it arrives to
--out (default data/kk_guides.jsonl: idx, id, people, names, answer, guide, finish_reason, usage), idx being the row in train.jsonl.
Rerun the same command to resume; items already in the file are skipped. --force starts over.
"""
import argparse, json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
import litellm
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
litellm.suppress_debug_info = True

INSTRUCTIONS = """You write solving guides for Knights and Knaves logic puzzles. A guide is a fill-in-the-blanks worksheet for a small student model. The student never sees the answer: it copies your template and replaces each ___ by following your steps. A good guide makes every blank a small, local check, so a weak model following it line by line reaches the correct answer.

You are given the puzzle and its correct answer. Use the answer to make sure every step leads to it, but never state anyone's real type outside a blank.

Write the guide in exactly this layout, with these section headings and nothing before or after. In the Instructions section, N is the number of your last step.

Instructions
Solve the puzzle below by filling in the template. Copy every line of the template and replace each ___ with your answer. Do not skip any line. Do not use a different method. Do not change the order of the steps. Write the CONCLUSION only after you finish Step N.
Question
The puzzle, with every statement written as <Name> said, "<statement>." Keep each statement's exact meaning and words; only the speech verbs change.
Rules
The same three rules as in the example, word for word.
What each statement means
One bullet per person, in puzzle order:
* <Name>'s statement means: "<plain restatement>" It is TRUE if ... It is FALSE if ...
- Translate into plain words: "A is a knight iff B is a knight" and "A is a knave iff B is a knave" mean "A and B are the same type"; "A is a knight iff B is a knave" means "A and B are different types"; "A or B" means "at least one ..." and is FALSE only if both parts fail; "A and B" means "both ..." and is FALSE if either part fails; "if A then B" is FALSE only when A happens and B does not; "It is not the case that X is a knave" means "X is a knight".
- Read the original wording carefully: knight vs knave, and vs or, the direction of if-then, who is talked about.
- Always say exactly when the statement is TRUE and when it is FALSE.
- Name the likely traps, as the example does ("It does NOT say that Michael is a knave. It does not talk about Ethan.").
- A statement that can never be true ("X is a knight and X is a knave", "X is a knight iff X is a knave") is ALWAYS FALSE, so its speaker must be a knave; a statement that is always true ("X is a knight or X is a knave") is ALWAYS TRUE, so its speaker must be a knight. Say so: these are the best places to start.
- Do not state anyone's actual type in this section.
Template
One line per step, each starting "Step k:".
- Every step except the last finds exactly one person: "Step k: Find <Name>. ..." Order the steps so each step uses only people found in earlier steps, plus at most one case split ("Case A: Suppose X is a knight ... Case B: Suppose X is a knave ...") whose cases either agree, as in the example, or where one case reaches a mismatch ("Does it match? ___" answered no, then "So Case B is impossible.").
- Start from the easiest anchor: an always-FALSE or always-TRUE statement, a statement only about the speaker, or a person whose two cases agree.
- To test a person, show both tries: "Try <Name> as a knight: ... So <Name>'s statement is ___. A knight must say something TRUE. Does it match? ___. Try <Name> as a knave: ... A knave must say something FALSE. Does it match? ___. Keep the try that matches. So <Name> is a ___."
- Split each statement into its parts and give each part its own blank (""Zoey is a knight" is ___."), then combine them ("So Zoey's statement is ___."). Before using an earlier result, restate it: "From Step j, <Name> is a ___."
- The last step is always "Step N: Check every person. You must write all <number> lines." Then, for every person in puzzle order: "<Name> is a ___. <the parts of the statement, each with a blank>, so <Name>'s statement is ___. A ___ must say something ___. Does it match? ___." End with: "If any answer to "Does it match?" is no, your answer is wrong. Go back to Step 1 and find the mistake. Write the CONCLUSION only when all <number> answers are yes."
- Blanks hold only short answers: knight, knave, same, different, TRUE, FALSE, yes, no.
- The Template is EMPTY: every answer is a ___. You must not fill in any blank yourself. Every word the student has to work out (a type, same/different, TRUE/FALSE, yes/no) is a ___, everywhere, including inside cases and tries: write "So Ethan is a ___.", never "So Ethan is a knight.", even inside "Case B".
- Never write a person's real type outside a blank (never "So Ethan is a knight."). Hypotheticals ("Suppose Ethan is a knight", "Try Ethan as a knight") are fine.
- Every sentence, once filled in correctly, must be true in its context (in the real solution, or inside the case it belongs to).
- Short, plain sentences. No symbols, no logic notation, no extra commentary.
Answer format
Fill each ___ with "knight" or "knave". Keep this exact order: <names in puzzle order>.
CONCLUSION: (1) <Name> is a ___ (2) <Name> is a ___ ...

Puzzles have 3 to 5 people, so a guide has one "Find" step per person plus the final check."""

REQUEST = """Puzzle:
{quiz}

Correct answer (for you only; never state it outside a blank):
{answer}

Write the guide. Remember: the Template contains only ___ in place of answers; do not fill any of them in."""


def load_items(src):
    items = []
    for idx, line in enumerate(open(src / "rewrites.jsonl")):  # same order as train.jsonl
        r = json.loads(line)
        items.append({"idx": idx, "id": r["id"], "people": int(r["people"]), "quiz": r["quiz"],
                      "names": r["names"], "answer": r["answer"]})
    return items


def request(item):
    answer = "\n".join(f"({i}) {n} is a {a}" for i, (n, a) in enumerate(zip(item["names"], item["answer"]), 1))
    return REQUEST.format(quiz=item["quiz"], answer=answer)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=ROOT.parent / "kk_claude-haiku-5-5",
                    help="folder with rewrites.jsonl (aligned with train.jsonl)")
    ap.add_argument("--model", default="anthropic/claude-haiku-5-5", help="LiteLLM model string")
    ap.add_argument("--model-args", type=json.loads, default={}, help="JSON passed to litellm.completion")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--limit", type=int, help="first N items only")
    ap.add_argument("--ids", nargs="+", help="only these item ids")
    ap.add_argument("--out", type=Path, default=ROOT / "data/kk_guides.jsonl")
    ap.add_argument("--force", action="store_true", help="delete --out and start over")
    args = ap.parse_args()

    # LiteLLM's HTTP client keeps httpx's default pool of 100 connections; match it to the thread count
    limits = httpx.Limits(max_connections=args.concurrency, max_keepalive_connections=args.concurrency)
    litellm.sync_transport = httpx.HTTPTransport(limits=limits)
    kwargs = {"max_tokens": 16000, "num_retries": 3, **args.model_args}

    items = load_items(args.src)
    if args.ids:
        items = [it for it in items if it["id"] in set(args.ids)]
    items = items[: args.limit]
    example = next(it for it in load_items(args.src) if it["id"] == "3ppl_100")
    system = (f"{INSTRUCTIONS}\n\n## Example\n\n<request>\n{request(example)}\n</request>\n\n"
              f"<guide>\n{(ROOT / 'data/kk_guide_example.txt').read_text().strip()}\n</guide>\n\n"
              "Reply with the guide only, in the same layout, without the <guide> tags.")
    system_msg = {"role": "system", "content": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]}

    path = args.out
    path.parent.mkdir(parents=True, exist_ok=True)
    if args.force:
        path.unlink(missing_ok=True)
    done = set()
    if path.exists():
        rows = []
        for line in open(path):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:  # a line cut short by an interruption
                continue
        rows = [r for r in rows if r.get("guide")]  # failed requests are retried
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        done = {r["id"] for r in rows}
    todo = [it for it in items if it["id"] not in done]
    print(f"{len(items) - len(todo)} items already done, {len(todo)} to go -> {path}")

    def call(item):
        msgs = [system_msg, {"role": "user", "content": request(item)}]
        try:
            r = litellm.completion(model=args.model, messages=msgs, **kwargs)
        except Exception as e:  # noqa: BLE001 - keep going; the item is retried on the next run
            return {"error": f"{type(e).__name__}: {str(e)[:300]}"}
        text = (r.choices[0].message.content or "").strip()
        text = text.removeprefix("<guide>").removesuffix("</guide>").strip()
        text = text.replace("\n(N is the number of your last step.)", "")  # a note from the instructions, sometimes copied
        details = getattr(r.usage, "prompt_tokens_details", None)
        usage = {"prompt": r.usage.prompt_tokens, "completion": r.usage.completion_tokens,
                 "cached": getattr(details, "cached_tokens", None) or 0}
        return {"guide": text, "finish_reason": r.choices[0].finish_reason, "usage": usage}

    no_blanks = []
    tokens, failed, truncated = {"prompt": 0, "cached": 0, "completion": 0}, 0, 0
    with open(path, "a") as f, ThreadPoolExecutor(args.concurrency) as pool:
        futures = {pool.submit(call, it): it for it in todo}
        try:
            for fut in tqdm(as_completed(futures), total=len(futures), desc="guides"):
                it, res = futures[fut], fut.result()
                if "error" in res:
                    failed += 1
                    tqdm.write(f"{it['id']}: {res['error']}")
                    continue
                truncated += res["finish_reason"] == "length"
                if "___" not in res["guide"].split("Template")[-1]:
                    no_blanks.append(it["id"])
                for k in tokens:
                    tokens[k] += res["usage"][k]
                row = {k: it[k] for k in ("idx", "id", "people", "names", "answer")} | res
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
        except KeyboardInterrupt:
            pool.shutdown(wait=False, cancel_futures=True)
            print("interrupted; rerun to resume")
    print(f"done: {len(todo) - failed} written, {failed} failed (rerun to retry), {truncated} cut off at max_tokens; "
          f"tokens {tokens}")
    if no_blanks:
        print(f"{len(no_blanks)} guides have no ___ in their Template (answers filled in): {' '.join(no_blanks)}")


if __name__ == "__main__":
    main()
