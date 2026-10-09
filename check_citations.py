"""check_citations.py - STUDENT IMPLEMENTS `check`.   Runs INSIDE the sandbox (standard library only).

research.py uploads this file to the sandbox and the lead agent runs it with the `execute` tool:
    python3 /tmp/work/research/check_citations.py [report.md] [sources.json]
It must exit 0 and print "OK: ..." when the report is consistent, else print each problem and exit 1.
"""
import json
import re
import sys

REPORT = "/tmp/work/report/report.md"
SOURCES = "/tmp/work/research/sources.json"
REF_HEADING = re.compile(r"(?m)^##[ \t]+References[ \t]*$")
CODE = re.compile(r"(```.*?```|`[^`\n]*`)", re.DOTALL)
CITE = re.compile(r"\[(\d+(?:\s*[,–-]\s*\d+)*)\](?!\()")
REF_LINE = re.compile(r"^\[(\d+)\]\s+(.+)$")
URL = re.compile(r"https?://[^\s)]+")


def check(report_text, sources):
    """Return a list of problem strings (empty list = OK).

    PSEUDO-CODE:
      problems = []
      if sources is empty: return ["no sources in sources.json"]
      for each source entry:
          n must be an int                       -> problem if not
          url must start with http:// or https://-> problem if not
          the same url must not appear twice     -> problem if duplicated
      split report_text at the heading "## References":
          body = text before it; if the heading is missing -> problem
      cited = set of numbers found as [n] in the BODY only (not in the reference list; use a regex)
      every number in `cited` must exist in sources -> problem "[n] cited but missing from sources.json"
      every source number must be in `cited`        -> problem "source [n] never cited"
      the lines of the References section that start with "[n]" (regex) are the reference lines:
          every source needs exactly ONE reference line (none missing, no number twice, no number that is not a source)
          each reference line holds exactly ONE http(s) URL and it must equal that source's url
          (a line bundling several sources under one number is a problem)
      return problems
    """
    problems = []
    if not isinstance(sources, list) or not sources:
        return ["no sources in sources.json"]

    source_by_n = {}
    seen_urls = {}
    for index, source in enumerate(sources, 1):
        if not isinstance(source, dict):
            problems.append(f"source #{index} is not an object")
            continue
        n = source.get("n")
        url = source.get("url")
        if not isinstance(n, int):
            problems.append(f"source #{index} n is not an integer")
        elif n in source_by_n:
            problems.append(f"duplicate source number [{n}]")
        else:
            source_by_n[n] = source
        if not isinstance(url, str) or not url.startswith(("http://", "https://")):
            problems.append(f"source [{n}] url is not http(s)")
        elif url in seen_urls:
            problems.append(f"duplicate url in sources.json: {url}")
        else:
            seen_urls[url] = n

    matches = list(REF_HEADING.finditer(report_text or ""))
    if not matches:
        problems.append("missing ## References")
        body, refs = report_text or "", ""
    else:
        match = matches[-1]
        body, refs = report_text[:match.start()], report_text[match.end():]

    cited = _citations_in_body(body)
    for n in sorted(cited - set(source_by_n)):
        problems.append(f"[{n}] cited but missing from sources.json")
    for n in sorted(set(source_by_n) - cited):
        problems.append(f"source [{n}] never cited")

    ref_numbers = {}
    for line in refs.splitlines():
        line = line.strip()
        if not line:
            continue
        ref_match = REF_LINE.match(line)
        if not ref_match:
            continue
        n = int(ref_match.group(1))
        ref_numbers.setdefault(n, []).append(line)

    for n in sorted(source_by_n):
        lines = ref_numbers.get(n, [])
        if not lines:
            problems.append(f"missing reference line for [{n}]")
        elif len(lines) > 1:
            problems.append(f"multiple reference lines for [{n}]")
    for n in sorted(set(ref_numbers) - set(source_by_n)):
        problems.append(f"reference line [{n}] has no source in sources.json")

    for n, lines in sorted(ref_numbers.items()):
        if n not in source_by_n:
            continue
        for line in lines:
            urls = URL.findall(line)
            if len(urls) != 1:
                problems.append(f"reference [{n}] must contain exactly one URL")
            elif urls[0] != source_by_n[n].get("url"):
                problems.append(f"reference [{n}] URL does not match sources.json")
    return problems


def _expand_citation(group):
    numbers = []
    for part in re.split(r"\s*,\s*", group):
        span = re.fullmatch(r"(\d+)\s*[–-]\s*(\d+)", part)
        if span:
            start, end = int(span.group(1)), int(span.group(2))
            numbers.extend(range(start, end + 1) if 0 <= end - start <= 200 else [start, end])
        else:
            numbers.append(int(part))
    return numbers


def _citations_in_body(body):
    cited = set()
    parts = CODE.split(body or "")
    for index, part in enumerate(parts):
        if index % 2:
            continue
        for match in CITE.finditer(part):
            cited.update(_expand_citation(match.group(1)))
    return cited


def main(argv):
    report_path = argv[1] if len(argv) > 1 else REPORT
    sources_path = argv[2] if len(argv) > 2 else SOURCES
    try:
        with open(report_path, encoding="utf-8") as f:
            report = f.read()
        with open(sources_path, encoding="utf-8") as f:
            sources = json.load(f)
    except (OSError, ValueError) as exc:
        print(f"cannot read inputs: {exc}")
        return 1
    problems = check(report, sources)
    if problems:
        print("\n".join(problems))
        return 1
    print(f"OK: {len(sources)} sources, all citations resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
