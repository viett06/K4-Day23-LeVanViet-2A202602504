"""research.py - STUDENT IMPLEMENTS.  The main script.   Guide: GUIDE.md, part 3.

Usage:  python research.py "survey about world model"
Result: reports/<slug>.md   reports/<slug>.sources.json   reports/<slug>.meta.json
"""
import json  # noqa: F401
import os  # noqa: F401
import re  # noqa: F401
import sys
import time  # noqa: F401
import traceback
from collections import Counter  # noqa: F401
from pathlib import Path

from agents import FINALIZER_PATH, REPORT_PATH, SOURCES_PATH, VALIDATOR_PATH, WORKDIR, build_lead_agent  # noqa: F401
from model import make_model  # noqa: F401
from sandbox import download, open_sandbox, upload  # noqa: F401

ROOT = Path(__file__).parent
REPORTS = ROOT / "reports"
VALIDATOR_SOURCE = ROOT / "check_citations.py"
FINALIZER_SOURCE = ROOT / "finalize_citations.py"   # provided: uploaded next to your validator


def slugify(topic):
    """Turn a topic into a safe file name: lower case, runs of non-word characters become one "-", max 60 chars,
    never empty (fall back to "topic"). The topic is user input: "../../x" must not escape reports/."""
    slug = re.sub(r"[^\w]+", "-", (topic or "").strip().lower(), flags=re.UNICODE)
    slug = re.sub(r"-+", "-", slug).strip("-")
    return (slug or "topic")[:60].strip("-") or "topic"


def build_prompt(topic):
    """The user message sent to the lead agent."""
    return (
        f"Research topic: {topic}\n\n"
        "Create a citation-checked survey report for this topic. Follow the required workflow in your system prompt: "
        "plan, delegate at least three researcher tasks in parallel, merge sources, write the report body, run the "
        "finalizer, run the validator until OK, and spot-check claims. Stop searching once you have about 8 to 14 "
        "sources. sources.json must include at least three of arxiv, hf-daily, hf-search, and web, including at "
        "least one hf-daily paper whose URL is https://huggingface.co/papers/<id>. Cite that paper in the body. "
        "Then write and validate the report."
    )


def summarize(messages, elapsed, model_name):
    """Return {"model", "elapsed_s", "subagent_calls", "tool_calls": {name: count}, "tokens": {"input", "output"}}.

    PSEUDO-CODE: walk the lead's messages; for every message with tool_calls count call["name"] (subagent_calls = the
    count of "task"); add the input/output token counts from each message's usage_metadata when present.
    (Lead messages only: subagent tokens are not included, so this undercounts the real cost.)
    elapsed_s rounded to 0.1.
    """
    tool_calls = Counter()
    tokens = Counter()
    for message in messages or []:
        calls = _message_get(message, "tool_calls", []) or []
        for call in calls:
            name = call.get("name") if isinstance(call, dict) else getattr(call, "name", None)
            if name:
                tool_calls[name] += 1
        usage = _message_get(message, "usage_metadata", None) or {}
        if isinstance(usage, dict):
            tokens["input"] += int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
            tokens["output"] += int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
    return {
        "model": model_name,
        "elapsed_s": round(elapsed, 1),
        "subagent_calls": tool_calls.get("task", 0),
        "tool_calls": dict(sorted(tool_calls.items())),
        "tokens": {"input": tokens["input"], "output": tokens["output"]},
    }


def _message_get(message, key, default=None):
    if isinstance(message, dict):
        return message.get(key, default)
    return getattr(message, key, default)


def save_outputs(backend, topic, messages, elapsed, model_name, reports_dir=REPORTS):
    """Download the report from the sandbox and write the three files into reports_dir. Return the report path.

    PSEUDO-CODE:
      files = download(backend, [REPORT_PATH, SOURCES_PATH])
      if the report is missing/empty or sources.json is missing/invalid JSON: raise RuntimeError and WRITE NOTHING
          (a failed run must never leave an empty or half-written report behind)
      write <slug>.sources.json, <slug>.meta.json (topic + summarize(...) + n_sources + source_families: the sorted
      distinct "source" values of sources.json) and <slug>.md
    """
    files = download(backend, [REPORT_PATH, SOURCES_PATH])
    report_bytes = files.get(REPORT_PATH)
    sources_bytes = files.get(SOURCES_PATH)
    if not report_bytes or not report_bytes.strip():
        raise RuntimeError("report.md missing or empty")
    if not sources_bytes or not sources_bytes.strip():
        raise RuntimeError("sources.json missing or empty")
    try:
        sources = json.loads(sources_bytes.decode("utf-8"))
    except ValueError as exc:
        raise RuntimeError(f"sources.json is invalid JSON: {exc}") from exc
    if not isinstance(sources, list) or not sources:
        raise RuntimeError("sources.json must be a non-empty JSON array")

    slug = slugify(topic)
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / f"{slug}.md"
    sources_path = reports_dir / f"{slug}.sources.json"
    meta_path = reports_dir / f"{slug}.meta.json"
    meta = {
        "topic": topic,
        **summarize(messages, elapsed, model_name),
        "n_sources": len(sources),
        "source_families": sorted({str(entry.get("source", "")) for entry in sources if isinstance(entry, dict)}),
    }
    report_text = report_bytes.decode("utf-8")
    sources_text = json.dumps(sources, ensure_ascii=False, indent=2) + "\n"
    meta_text = json.dumps(meta, ensure_ascii=False, indent=2) + "\n"
    sources_path.write_text(sources_text, encoding="utf-8")
    meta_path.write_text(meta_text, encoding="utf-8")
    report_path.write_text(report_text, encoding="utf-8")
    return report_path


def main(topic):
    """Return the process exit code (0 ok, 1 failed run, 2 no topic).

    PSEUDO-CODE:
      empty topic -> print usage to stderr, return 2
      model = make_model(); start = time.monotonic()
      with open_sandbox() as backend:                # the sandbox is always cleaned up, even on errors
          backend.execute("mkdir -p <WORKDIR>/research/notes <WORKDIR>/report")
          upload(backend, {VALIDATOR_PATH: VALIDATOR_SOURCE.read_bytes(), FINALIZER_PATH: FINALIZER_SOURCE.read_bytes()})
          agent = build_lead_agent(backend, model)
          result = agent.invoke({"messages": [{"role": "user", "content": build_prompt(topic)}]},
                                config={"recursion_limit": 1000})
          save_outputs(...); on RuntimeError print "FAILED: ..." to stderr and return 1
      print where the report was saved; return 0
    """
    topic = (topic or "").strip()
    if not topic:
        print('usage: python research.py "<topic>"', file=sys.stderr)
        return 2
    try:
        print(f"running: {topic}", flush=True)
        model = make_model()
        model_name = os.getenv("LAB_MODEL") or getattr(model, "model_name", None) or getattr(model, "model", "unknown")
        start = time.monotonic()
        with open_sandbox() as backend:
            backend.execute(f"mkdir -p {WORKDIR}/research/notes {WORKDIR}/report")
            upload(backend, {
                VALIDATOR_PATH: VALIDATOR_SOURCE.read_bytes(),
                FINALIZER_PATH: FINALIZER_SOURCE.read_bytes(),
            })
            agent = build_lead_agent(backend, model)
            result = agent.invoke(
                {"messages": [{"role": "user", "content": build_prompt(topic)}]},
                config={"recursion_limit": 1000},
            )
            elapsed = time.monotonic() - start
            report_path = save_outputs(
                backend,
                topic,
                result.get("messages", []) if isinstance(result, dict) else getattr(result, "messages", []),
                elapsed,
                model_name,
            )
        print(f"saved report: {report_path}")
        return 0
    except RuntimeError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - a failed run must exit non-zero and leave no report
        print(f"FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main(" ".join(sys.argv[1:])))
