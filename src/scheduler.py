"""One thread pool for every request of a script run, shared by evaluate.py, guided_generation.py and self_rewrite.py.

A finished request is handled on the main thread right away, and whatever follow-up requests that produces (the next try
of an item that failed) go straight into the same pool. Runs and tries therefore overlap: the pool stays full until the
very last request, instead of draining at the end of every run while the slowest requests finish.
"""
import json
import queue
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm


def drive(jobs, call, handle, concurrency, desc="items", status=None):
    """Run call(job) for every job on `concurrency` threads, in submission order.

    handle(job, result) runs on the main thread for each finished job (scoring and file writes stay there: the math grader
    uses SIGALRM) and returns the follow-up jobs to queue, if any. An exception from call() is raised here; it, like
    Ctrl-C, cancels every request still queued, so a resumed run picks up from what was written.

    The progress bar counts the initial jobs (items): one is done when a result comes back with no follow-up (solved, or
    out of tries), so its total stays fixed. Beside it: whatever status() returns (e.g. {"solved": n}) and the number of
    requests sent.
    """
    jobs = list(jobs)
    pool, finished, pending, requests = ThreadPoolExecutor(concurrency), queue.Queue(), 0, 0
    bar = tqdm(total=len(jobs), desc=desc)

    def submit(job):
        nonlocal pending, requests
        pool.submit(call, job).add_done_callback(lambda fut: finished.put((job, fut)))
        pending += 1
        requests += 1

    try:
        for job in jobs:
            submit(job)
        while pending:
            try:
                job, fut = finished.get(timeout=1)  # a timeout keeps Ctrl-C responsive
            except queue.Empty:
                continue
            pending -= 1
            follow_ups = list(handle(job, fut.result()) or ())
            for follow_up in follow_ups:
                submit(follow_up)
            if not follow_ups:
                bar.update(1)
            bar.set_postfix({**(status() if status else {}), "requests": requests}, refresh=False)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)  # Ctrl-C shouldn't wait for every queued request
        bar.close()


def read_jsonl(path):
    """Records of a .jsonl, if it exists; a last line cut short (an interrupted run) is dropped."""
    rows = []
    if path.exists():
        with open(path) as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def reopen(path, rows):
    """Rewrite path with rows only (dropping a cut-short last line), atomically, and return it opened for appending."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    tmp.replace(path)
    return open(path, "a")


def append(f, row):
    f.write(json.dumps(row, ensure_ascii=False) + "\n")
    f.flush()


def answer(response):
    """(text, finish_reason, reasoning) of a completion; only the part after </think> is the answer."""
    message = response.choices[0].message
    text = message.content or ""
    text = "" if text.lstrip().startswith("<think>") and "</think>" not in text else text  # cut off while thinking
    return text.rpartition("</think>")[2].strip(), response.choices[0].finish_reason, getattr(message, "reasoning_content", None)

