#!/usr/bin/env python3
"""
Evaluate EPUB summaries against a NotebookLM reference using a configurable local or cloud LLM evaluator.

Usage:
    python3 evaluate_summaries.py BOOK.epub NOTEBOOKLM_SUMMARY.md RESULTS_DIR

The evaluator:
- extracts the EPUB chapters as the primary factual source;
- maps NotebookLM chapter sections to EPUB chapters;
- creates importance-tagged atomic reference facts;
- evaluates every discovered results/<model>/run-* summary against those facts;
- checks coverage, omissions, contradictions and unsupported claims;
- evaluates the complete-book summaries;
- records evaluator model/configuration and source-model runtime metadata;
- produces JSON + Markdown + HTML reports.

Book text is sent only to the evaluator provider selected on the command line.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Optional

OPF_NS = "http://www.idpf.org/2007/opf"
DC_NS = "http://purl.org/dc/elements/1.1/"


# ---------------------------------------------------------------------------
# EPUB parsing
# ---------------------------------------------------------------------------

class TextExtractor(HTMLParser):
    BLOCK_TAGS = {
        "p", "div", "section", "article", "blockquote",
        "h1", "h2", "h3", "h4", "h5", "h6", "li", "br", "hr", "tr"
    }
    SKIP_TAGS = {"script", "style", "svg", "head"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skip_depth = 0
        self.headings: list[str] = []
        self.current_heading: Optional[str] = None

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag in self.SKIP_TAGS:
            self.skip_depth += 1
            return
        if self.skip_depth:
            return
        if tag in self.BLOCK_TAGS:
            self.parts.append("\n")
        if re.fullmatch(r"h[1-6]", tag):
            self.current_heading = ""

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag in self.SKIP_TAGS:
            if self.skip_depth:
                self.skip_depth -= 1
            return
        if self.skip_depth:
            return
        if self.current_heading is not None and re.fullmatch(r"h[1-6]", tag):
            title = clean_text(self.current_heading)
            if title:
                self.headings.append(title)
            self.current_heading = None
        if tag in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data):
        if self.skip_depth:
            return
        if self.current_heading is not None:
            self.current_heading += " " + data
        else:
            self.parts.append(data)


def clean_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


@dataclass
class Chapter:
    number: str
    title: str
    href: str
    text: str


def find_opf_path(z: zipfile.ZipFile) -> str:
    container = ET.fromstring(z.read("META-INF/container.xml"))
    rootfile = container.find(
        ".//{urn:oasis:names:tc:opendocument:xmlns:container}rootfile"
    )
    if rootfile is None:
        raise RuntimeError("META-INF/container.xml does not contain a rootfile.")
    return rootfile.attrib["full-path"]


def read_epub(epub_path: Path) -> tuple[str, str, list[Chapter]]:
    with zipfile.ZipFile(epub_path, "r") as z:
        opf_path = find_opf_path(z)
        root = ET.fromstring(z.read(opf_path))
        manifest = {}
        for item in root.findall(f".//{{{OPF_NS}}}manifest/{{{OPF_NS}}}item"):
            manifest[item.attrib["id"]] = item.attrib["href"]
        spine = [
            x.attrib["idref"]
            for x in root.findall(f".//{{{OPF_NS}}}spine/{{{OPF_NS}}}itemref")
        ]
        title_el = root.find(f".//{{{DC_NS}}}title")
        book_title = title_el.text.strip() if title_el is not None and title_el.text else epub_path.stem
        language_el = root.find(f".//{{{DC_NS}}}language")
        book_language = language_el.text.strip() if language_el is not None and language_el.text else ""
        chapters: list[Chapter] = []
        opf_dir = Path(opf_path).parent

        for idref in spine:
            href = manifest.get(idref)
            if not href or not href.lower().endswith((".html", ".xhtml", ".htm")):
                continue
            full_path = (opf_dir / href).as_posix()
            try:
                raw = z.read(full_path).decode("utf-8", errors="replace")
            except KeyError:
                continue
            parser = TextExtractor()
            parser.feed(raw)
            text = clean_text("".join(parser.parts))

            # EPUBs use different ways of marking chapter titles. Some use
            # h1/h2 elements, while others (e.g. Atlantis-generated EPUBs)
            # store the title in <title> and/or a paragraph such as
            # "1. YIS VENTER RAJONGÁSA". Support both forms.
            heading = parser.headings[0] if parser.headings else ""
            title_match = re.search(r"<title[^>]*>(.*?)</title>", raw, re.I | re.S)
            document_title = clean_text(html.unescape(title_match.group(1))) if title_match else ""

            candidates = [heading, document_title]
            number = None
            chapter_title = ""
            for candidate in candidates:
                candidate = clean_text(candidate)
                m = re.match(r"^(\d+)\.\s*(.+)$", candidate)
                if m:
                    number = m.group(1)
                    chapter_title = m.group(2).strip()
                    break
                m = re.fullmatch(r"(\d+)", candidate)
                if m:
                    number = m.group(1)
                    chapter_title = f"Chapter {number}"
                    break
                if candidate.upper().startswith("EPILOGUE"):
                    number = "EPILOGUE"
                    chapter_title = candidate
                    break

            if number is not None:
                chapters.append(Chapter(number, chapter_title, full_path, text))

        if not chapters:
            raise RuntimeError("No numbered chapters or EPILOGUE detected in EPUB.")
        return book_title, book_language, chapters


# ---------------------------------------------------------------------------
# LM Studio
# ---------------------------------------------------------------------------

def api_request(url: str, payload: dict, headers: Optional[dict] = None, timeout: int = 900) -> dict:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req_headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if headers:
        req_headers.update(headers)
    req = urllib.request.Request(url, data=data, headers=req_headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code} from {url}: {body}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Could not connect to {url}.") from e


def get_loaded_model(base_url: str) -> tuple[str, dict, int]:
    url = base_url.rstrip("/") + "/api/v1/models"
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))
    except Exception as e:
        raise RuntimeError(f"Could not query {url}. Is LM Studio running?") from e

    loaded = []
    for info in data.get("models", []):
        if info.get("type") != "llm":
            continue
        for instance in info.get("loaded_instances") or []:
            loaded.append((info, instance))

    if not loaded:
        raise RuntimeError("No loaded LLM found in LM Studio /api/v1/models.")
    if len(loaded) > 1:
        names = [x[0].get("key", "?") for x in loaded]
        raise RuntimeError(
            "More than one LLM is loaded. Refusing to guess evaluator model: "
            + ", ".join(names)
        )

    info, instance = loaded[0]
    config = instance.get("config") or {}
    context = config.get("context_length") or info.get("max_context_length")
    if not context:
        raise RuntimeError("Loaded evaluator model has no usable context length.")
    return info["key"], info, int(context)


def reasoning_value(value: str) -> str:
    return {"off": "none", "on": "high"}.get(value, value)


def parse_external_usage(usage: dict) -> dict:
    usage = usage or {}
    # Normalize common provider field names to the fields used by the report.
    prompt = usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0
    completion = usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0
    return {"prompt_tokens": int(prompt), "completion_tokens": int(completion), **usage}


def cloud_chat_openai(
    model: str,
    system: str,
    user: str,
    max_tokens: int,
    temperature: float,
    reasoning: str,
    timeout: int,
) -> tuple[str, dict, float, dict]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "reasoning_effort": reasoning_value(reasoning),
    }
    started = time.perf_counter()
    response = api_request(
        "https://api.openai.com/v1/chat/completions",
        payload,
        {"Authorization": f"Bearer {api_key}"},
        timeout,
    )
    elapsed = time.perf_counter() - started
    try:
        choice = response["choices"][0]
        message = choice["message"]
        content = (message.get("content") or "").strip()
        finish_reason = choice.get("finish_reason")
    except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError(f"Unexpected OpenAI response: {response}") from e
    usage = parse_external_usage(response.get("usage"))
    if not content:
        raise RuntimeError(
            f"OpenAI evaluator returned empty content; finish_reason={finish_reason}, "
            f"completion_tokens={usage.get('completion_tokens', 0)}"
        )
    info = {"display_name": model, "model": model, "provider": "openai", "context": None}
    return content, usage, elapsed, info


def cloud_chat_claude_code(
    model: str,
    system: str,
    user: str,
    max_tokens: int,
    temperature: float,
    reasoning: str,
    timeout: int,
) -> tuple[str, dict, float, dict]:
    """Call the locally authenticated Claude Code CLI without an API key.

    Claude Code print mode supports JSON output and can read the prompt from
    stdin. We deliberately use one non-agentic turn because this evaluator
    should only judge the supplied text and must not modify/read project files.
    """
    claude = shutil.which("claude")
    if not claude:
        raise RuntimeError(
            "Claude Code CLI was not found in PATH. Run 'claude' once and make sure it is installed and authenticated."
        )

    if reasoning not in ("off",):
        raise RuntimeError("Claude Code evaluator currently supports --reasoning off only in this script.")

    # max_tokens is communicated in the prompt because Claude Code's print-mode
    # CLI does not expose the same max_tokens parameter as the Messages API.
    user_with_budget = (
        user
        + "\n\nIMPORTANT OUTPUT LIMIT: Keep your JSON response within approximately "
        + str(max_tokens)
        + " output tokens. Return JSON only."
    )

    cmd = [
        claude,
        "-p",
        "--output-format", "json",
        "--model", model,
        "--system-prompt", system,
        "--max-turns", "1",
    ]

    started = time.perf_counter()
    try:
        completed = subprocess.run(
            cmd,
            input=user_with_budget,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"Claude Code evaluator timed out after {timeout} seconds.") from e

    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        stdout = completed.stdout.strip()
        detail = stderr or stdout or f"exit code {completed.returncode}"
        raise RuntimeError(f"Claude Code evaluator failed: {detail}")

    try:
        response = json.loads(completed.stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            "Claude Code did not return valid JSON output. "
            f"Output: {completed.stdout[:1000]}"
        ) from e

    if response.get("is_error"):
        raise RuntimeError(f"Claude Code evaluator returned an error: {response}")

    content = str(response.get("result") or "").strip()
    if not content:
        raise RuntimeError(f"Claude Code evaluator returned empty content: {response}")

    usage = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "duration_ms": response.get("duration_ms"),
        "duration_api_ms": response.get("duration_api_ms"),
        "num_turns": response.get("num_turns"),
        "total_cost_usd": response.get("total_cost_usd"),
        "session_id": response.get("session_id"),
    }
    info = {
        "display_name": model,
        "model": model,
        "provider": "claude-code",
        "context": None,
        "cost_usd": response.get("total_cost_usd"),
    }
    return content, usage, elapsed, info


def cloud_chat_anthropic(
    model: str,
    system: str,
    user: str,
    max_tokens: int,
    temperature: float,
    reasoning: str,
    timeout: int,
) -> tuple[str, dict, float, dict]:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set.")
    if reasoning not in ("off",):
        raise RuntimeError("Anthropic evaluator currently supports --reasoning off only in this script.")
    # Claude's current API does not require a temperature parameter for the
    # benchmark; omitting it also avoids incompatibilities with newer models.
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }
    started = time.perf_counter()
    response = api_request(
        "https://api.anthropic.com/v1/messages",
        payload,
        {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        timeout,
    )
    elapsed = time.perf_counter() - started
    try:
        blocks = response.get("content") or []
        content = "".join(b.get("text", "") for b in blocks if b.get("type") == "text").strip()
        stop_reason = response.get("stop_reason")
    except (AttributeError, TypeError) as e:
        raise RuntimeError(f"Unexpected Anthropic response: {response}") from e
    usage = parse_external_usage(response.get("usage"))
    if not content:
        raise RuntimeError(
            f"Anthropic evaluator returned empty content; stop_reason={stop_reason}, "
            f"output_tokens={usage.get('completion_tokens', 0)}"
        )
    info = {"display_name": model, "model": model, "provider": "anthropic", "context": None}
    return content, usage, elapsed, info


def chat(
    provider: str,
    base_url: str,
    model: str,
    system: str,
    user: str,
    max_tokens: int,
    temperature: float,
    reasoning: str,
    timeout: int,
) -> tuple[str, dict, float, dict]:
    if provider == "local":
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "reasoning_effort": reasoning_value(reasoning),
        }
        started = time.perf_counter()
        response = api_request(base_url.rstrip("/") + "/v1/chat/completions", payload, timeout=timeout)
        elapsed = time.perf_counter() - started
        try:
            choice = response["choices"][0]
            message = choice["message"]
            content = (message.get("content") or "").strip()
            finish_reason = choice.get("finish_reason")
        except (KeyError, IndexError, TypeError) as e:
            raise RuntimeError(f"Unexpected LM Studio response: {response}") from e
        usage = parse_external_usage(response.get("usage"))
        if not content:
            details = usage.get("completion_tokens_details") or {}
            raise RuntimeError(
                f"Evaluator returned empty content; finish_reason={finish_reason}, "
                f"completion_tokens={usage.get('completion_tokens', 0)}, "
                f"reasoning_tokens={details.get('reasoning_tokens', 0)}"
            )
        return content, usage, elapsed, {"provider": "local"}
    if provider == "openai":
        return cloud_chat_openai(model, system, user, max_tokens, temperature, reasoning, timeout)
    if provider == "anthropic":
        return cloud_chat_anthropic(model, system, user, max_tokens, temperature, reasoning, timeout)
    if provider == "claude-code":
        return cloud_chat_claude_code(model, system, user, max_tokens, temperature, reasoning, timeout)
    raise RuntimeError(f"Unsupported evaluator provider: {provider}")

def extract_json(text: str) -> Any:
    """Parse JSON from a response, tolerating markdown fences or surrounding prose."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    candidates = []
    for opener, closer in [("{", "}"), ("[", "]")]:
        start = cleaned.find(opener)
        end = cleaned.rfind(closer)
        if start >= 0 and end > start:
            candidates.append(cleaned[start:end + 1])
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise RuntimeError("Evaluator response did not contain valid JSON.")


def chat_json_with_retry(
    provider: str,
    base_url: str,
    model: str,
    system: str,
    user: str,
    budget: int,
    max_budget: int,
    temperature: float,
    reasoning: str,
    timeout: int,
) -> tuple[Any, dict, float, dict]:
    """Run an evaluator request and retry once with the full JSON budget if parsing fails.

    This protects long fact/claim evaluations from being cut off at a dynamically
    calculated output limit. The first request keeps the adaptive budget; the retry
    uses the configured maximum.
    """
    raw, usage, elapsed, meta = chat(
        provider, base_url, model, system, user, budget, temperature, reasoning, timeout
    )
    try:
        return extract_json(raw), usage, elapsed, meta
    except RuntimeError:
        if max_budget <= budget:
            raise
        print(f"      JSON response incomplete/invalid; retrying with max output {max_budget} tokens")
        raw2, usage2, elapsed2, meta2 = chat(
            provider, base_url, model, system, user, max_budget, temperature, reasoning, timeout
        )
        return extract_json(raw2), usage2, elapsed + elapsed2, meta2


def estimate_tokens(text: str) -> int:
    return max(1, int(len(text) / 3.5))


def dynamic_budget(text: str, cap: int, minimum: int) -> int:
    value = max(minimum, int(estimate_tokens(text) * 0.35))
    value = ((value + 511) // 512) * 512
    return min(cap, value)


# ---------------------------------------------------------------------------
# Reference parsing
# ---------------------------------------------------------------------------

CHAPTER_HEADING_RE = re.compile(
    r"^\s{0,3}#{1,6}\s*Chapter\s+(\d+)\b.*$|^\s*Chapter\s+(\d+)\b.*$",
    re.I | re.M,
)
EPILOGUE_HEADING_RE = re.compile(
    r"^\s{0,3}#{1,6}\s*Epilogue\b.*$|^\s*Epilogue\b.*$",
    re.I | re.M,
)


def parse_reference_sections(text: str) -> dict[str, str]:
    """Parse NotebookLM chapter sections by explicit Chapter N headings.

    NotebookLM may emit headings such as ``### Chapter 1 — The Hunt``.
    We deliberately do not use the prose inside the heading as the key;
    chapter number is the stable identifier shared with the EPUB.
    """
    text = text.replace("\ufeff", "")
    matches: list[tuple[int, int, str]] = []

    for m in CHAPTER_HEADING_RE.finditer(text):
        key = m.group(1) or m.group(2)
        if key:
            matches.append((m.start(), m.end(), key))

    for m in EPILOGUE_HEADING_RE.finditer(text):
        matches.append((m.start(), m.end(), "EPILOGUE"))

    matches.sort(key=lambda x: x[0])
    sections: dict[str, str] = {}
    for i, (start, end, key) in enumerate(matches):
        next_start = matches[i + 1][0] if i + 1 < len(matches) else len(text)
        sections[key] = text[end:next_start].strip()
    return sections


def render_prompt(template: str, **values: object) -> str:
    """Substitute only named prompt placeholders, leaving JSON braces intact."""
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace("{" + key + "}", str(value))
    return rendered


def load_model_runs(results_dir: Path) -> list[Path]:
    runs = []
    for model_dir in sorted(results_dir.iterdir() if results_dir.exists() else []):
        if not model_dir.is_dir() or model_dir.name == "evaluation":
            continue
        for run_dir in sorted(model_dir.glob("run-*")):
            if run_dir.is_dir() and (run_dir / "book_info.json").exists():
                runs.append(run_dir)
    # Backward compatibility: allow a single legacy results/model directory.
    legacy = results_dir / "model"
    if legacy.is_dir() and (legacy / "book_info.json").exists() and not any(r == legacy for r in runs):
        runs.append(legacy)
    return runs


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_model_chapter_summaries(run_dir: Path, chapters: list[Chapter]) -> dict[str, str]:
    result = {}
    for chapter in chapters:
        d = run_dir / (f"{int(chapter.number):03d}" if chapter.number.isdigit() else "epilogue")
        path = d / "chapter_summary.json"
        if path.exists():
            data = read_json(path)
            result[chapter.number] = str(data.get("summary", ""))
    return result


def read_book_summary(run_dir: Path) -> str:
    path = run_dir / "book_summary.md"
    if path.exists():
        return path.read_text(encoding="utf-8")
    path = run_dir / "book_summary.json"
    if path.exists():
        return str(read_json(path).get("summary", ""))
    return ""


# ---------------------------------------------------------------------------
# Evaluation prompts
# ---------------------------------------------------------------------------

SYSTEM = (
    "You are a rigorous factual benchmark evaluator. "
    "Use the supplied source text as the final authority. "
    "The NotebookLM reference is useful for identifying important events, "
    "but it can be imperfect. Do not invent facts. Return valid JSON only."
)

FACT_PROMPT = """Create an importance-tagged reference fact set for this chapter.

The EPUB text is the primary source of truth. The NotebookLM section is a secondary reference used to help identify important plot information.

Return JSON with this exact shape:
{
  "facts": [
    {
      "id": "F001",
      "fact": "concise independently verifiable factual statement",
      "importance": "critical|major|moderate|minor",
      "type": "event|character|discovery|decision|consequence|relationship|setting|other"
    }
  ]
}

Rules:
- Include actual plot facts that matter for understanding the story.
- Assign importance based on significance to the chapter/full plot, not how much text the fact occupies.
- critical: major turning point, major revelation, decisive action, major consequence, or fact essential to the main plot.
- major: important event, character action, discovery, conflict, decision, or consequence.
- moderate: useful plot information but not central.
- minor: correct detail that is safe to omit from a good summary.
- Prefer atomic facts: one independently checkable claim per fact.
- Do not treat character speculation as fact.
- Do not create facts that are only present in NotebookLM but unsupported by the EPUB.
- Do not quote the novel.
- When describing facts or claims, preserve the source terminology and do not translate factual names or terms unnecessarily.

BOOK: {book_title}
SOURCE LANGUAGE: {book_language}
CHAPTER: {chapter_title}

NOTEBOOKLM REFERENCE:
{reference}

EPUB CHAPTER TEXT:
{source}
"""

EVAL_PROMPT = """Evaluate the local model's chapter summary against the reference facts and the original EPUB chapter.

Return JSON with this exact shape:
{
  "fact_results": [
    {
      "fact_id": "F001",
      "status": "supported|partial|missing|contradicted",
      "importance": "critical|major|moderate|minor",
      "note": "brief evidence-based explanation"
    }
  ],
  "model_claims": [
    {
      "claim": "claim made by the model summary",
      "status": "supported|partial|unsupported|contradicted",
      "importance": "critical|major|moderate|minor",
      "note": "brief explanation"
    }
  ]
}

Rules:
- A fact is supported only if the EPUB supports it.
- partial means the summary captures only part of the fact or materially loses an important qualifier.
- missing means the model does not communicate the fact.
- contradicted means the model says something inconsistent with the EPUB.
- Evaluate the importance of model claims by their role in the plot.
- Only list meaningful factual claims in model_claims; do not list every trivial sentence.
- Do not penalize the model for omitting minor facts.
- Do not reward verbosity by itself.
- Pay special attention to causal relationships, character attribution, chronology, and consequences.

BOOK: {book_title}
SOURCE LANGUAGE: {book_language}
CHAPTER: {chapter_title}

REFERENCE FACTS:
{facts}

LOCAL MODEL SUMMARY:
{summary}

ORIGINAL EPUB CHAPTER:
{source}
"""

BOOK_EVAL_PROMPT = """Evaluate the complete-book summary using the chapter-level reference facts.

Return JSON:
{
  "critical_coverage": 0.0,
  "major_coverage": 0.0,
  "moderate_coverage": 0.0,
  "minor_coverage": 0.0,
  "weighted_coverage": 0.0,
  "story_arc_coverage": 0.0,
  "major_omissions": ["..."],
  "critical_omissions": ["..."],
  "unsupported_claims": ["..."],
  "contradictions": ["..."]
}

Coverage is the proportion of reference facts at each importance level that are correctly represented.
The reference facts have already been checked against the original EPUB chapter texts. Use those facts as the factual basis. Do not infer unsupported information.

REFERENCE FACTS:
{facts}

LOCAL COMPLETE-BOOK SUMMARY:
{summary}
"""


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def safe_name(name: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip(".-")
    return value or "model"


def aggregate_chapter_evaluation(results: list[dict]) -> dict:
    counts = {k: {s: 0 for s in ("supported", "partial", "missing", "contradicted")} for k in ("critical", "major", "moderate", "minor")}
    claims = {s: 0 for s in ("supported", "partial", "unsupported", "contradicted")}
    omissions = []
    contradictions = []
    unsupported = []

    for chapter_result in results:
        for item in chapter_result.get("fact_results", []):
            imp = item.get("importance", "moderate")
            if imp not in counts:
                imp = "moderate"
            status = item.get("status", "missing")
            if status not in counts[imp]:
                status = "missing"
            counts[imp][status] += 1
            if status == "missing" and imp in ("critical", "major"):
                omissions.append({"chapter": chapter_result.get("chapter"), "fact_id": item.get("fact_id"), "importance": imp, "note": item.get("note", "")})
            if status == "contradicted":
                contradictions.append({"chapter": chapter_result.get("chapter"), "fact_id": item.get("fact_id"), "importance": imp, "note": item.get("note", "")})
        for item in chapter_result.get("model_claims", []):
            status = item.get("status", "unsupported")
            if status not in claims:
                status = "unsupported"
            claims[status] += 1
            if status == "unsupported":
                unsupported.append({"chapter": chapter_result.get("chapter"), **item})

    weights = {"critical": 4, "major": 3, "moderate": 2, "minor": 1}
    weighted_total = weighted_hit = 0
    coverages = {}
    for imp, weight in weights.items():
        total = sum(counts[imp].values())
        hit = counts[imp]["supported"] + 0.5 * counts[imp]["partial"]
        coverages[imp] = hit / total if total else 1.0
        weighted_total += total * weight
        weighted_hit += hit * weight

    total_facts = sum(sum(x.values()) for x in counts.values())
    all_hit = sum(counts[i]["supported"] + 0.5 * counts[i]["partial"] for i in counts)
    return {
        "fact_counts": counts,
        "coverage": coverages,
        "overall_fact_coverage": all_hit / total_facts if total_facts else 1.0,
        "weighted_coverage": weighted_hit / weighted_total if weighted_total else 1.0,
        "claims": claims,
        "critical_omissions": [x for x in omissions if x["importance"] == "critical"],
        "major_omissions": [x for x in omissions if x["importance"] == "major"],
        "contradictions": contradictions,
        "unsupported_claims": unsupported,
    }


def _safe_dict(value):
    return value if isinstance(value, dict) else {}

def _safe_get(obj, key, default=None):
    d = _safe_dict(obj)
    value = d.get(key)
    return default if value is None else value


DEFAULT_OVERALL_FACT_WEIGHT = 0.70
DEFAULT_OVERALL_STORY_WEIGHT = 0.30

def default_adjusted_score(item: dict) -> float:
    a = _safe_dict(item.get("aggregate"))
    base = float(a.get("weighted_coverage", 0) or 0)
    fc = _safe_dict(a.get("fact_counts"))
    all_facts = missing = contradicted = 0
    for imp in ("critical", "major", "moderate", "minor"):
        c = _safe_dict(fc.get(imp))
        all_facts += sum(int(c.get(k, 0) or 0) for k in ("supported", "partial", "missing", "contradicted"))
        missing += int(c.get("missing", 0) or 0)
        contradicted += int(c.get("contradicted", 0) or 0)
    miss_rate = missing / all_facts if all_facts else 0.0
    contra_rate = contradicted / all_facts if all_facts else 0.0
    claims = _safe_dict(a.get("claims"))
    claim_total = sum(int(claims.get(k, 0) or 0) for k in ("supported", "partial", "unsupported", "contradicted"))
    unsupported_rate = int(claims.get("unsupported", 0) or 0) / max(1, claim_total)
    return max(0.0, base - 0.10 * miss_rate - 0.25 * contra_rate - 0.05 * unsupported_rate)

def default_overall_score(item: dict) -> float:
    adjusted = default_adjusted_score(item)
    story = float(_safe_dict(item.get("book_evaluation")).get("story_arc_coverage", 0) or 0)
    return DEFAULT_OVERALL_FACT_WEIGHT * adjusted + DEFAULT_OVERALL_STORY_WEIGHT * story


def markdown_report(path: Path, report: dict) -> None:
    lines = [
        f"# EPUB LLM Summary Evaluation",
        "",
        f"Generated: {report['generated_at']}",
        f"Book: **{report['book_title']}**",
        "",
        "## Evaluator",
        "",
        f"- Model: `{report['evaluator']['display_name']}`",
        f"- Model key: `{report['evaluator']['model']}`",
        f"- Variant: `{_safe_get(report.get('evaluator'), 'selected_variant', '?')}`",
        f"- Quantization: `{_safe_get(_safe_get(report['evaluator'], 'quantization', {}), 'name', '?')}`",
        f"- Context: `{report['evaluator']['context']:,}` tokens",
        f"- Temperature: `{report['evaluator']['temperature']}`",
        f"- Reasoning: `{report['evaluator']['reasoning']}`",
        "",
        "## Model comparison",
        "",
        "| Model / run | Overall | Adjusted factual | Story arc | Weighted coverage | Runtime | Critical | Major | Unsupported claims | Critical omissions | Major omissions |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    lines += [f"Default overall score: **{DEFAULT_OVERALL_FACT_WEIGHT*100:.0f}% adjusted factual + {DEFAULT_OVERALL_STORY_WEIGHT*100:.0f}% story arc**.", ""]
    for item in report["models"]:
        a = item["aggregate"]
        lines.append(
            f"| `{item['label']}` | {pct(default_overall_score(item))} | {pct(default_adjusted_score(item))} | "
            f"{pct((_safe_dict(item.get('book_evaluation')).get('story_arc_coverage', 0)))} | "
            f"{pct(a['weighted_coverage'])} | {item['runtime_minutes']:.1f} min | {pct(a['coverage']['critical'])} | "
            f"{pct(a['coverage']['major'])} | {a['claims']['unsupported']} | "
            f"{len(a['critical_omissions'])} | {len(a['major_omissions'])} |"
        )
    lines += ["", "## Detailed results", ""]

    for item in report["models"]:
        a = item["aggregate"]
        lines += [
            f"### {item['label']}",
            "",
            f"- Model: `{_safe_get(item.get('model_info'), 'display_name', item['model'])}`",
            f"- Model key: `{item['model']}`",
            f"- Variant: `{_safe_get(item.get('model_info'), 'selected_variant', '?')}`",
            f"- Quantization: `{_safe_get(_safe_get(item.get('model_info'), 'quantization', {}), 'name', '?')}`",
            f"- Runtime: `{item['runtime_minutes']:.1f} min`",
            f"- Prompt tokens: `{item['prompt_tokens']:,}`",
            f"- Completion tokens: `{item['completion_tokens']:,}`",
            "",
            "| Importance | Coverage | Total facts | Supported | Partial | Missing | Contradicted |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for imp in ("critical", "major", "moderate", "minor"):
            c = a["fact_counts"][imp]
            total = sum(c.values())
            lines.append(f"| {imp} | {pct(a['coverage'][imp])} | {total} | {c['supported']} | {c['partial']} | {c['missing']} | {c['contradicted']} |")
        lines += ["", "#### Critical omissions", ""]
        if a["critical_omissions"]:
            for x in a["critical_omissions"]:
                lines.append(f"- Chapter {x['chapter']} / {x['fact_id']}: {x['note']}")
        else:
            lines.append("None detected.")
        lines += ["", "#### Major omissions", ""]
        if a["major_omissions"]:
            for x in a["major_omissions"]:
                lines.append(f"- Chapter {x['chapter']} / {x['fact_id']}: {x['note']}")
        else:
            lines.append("None detected.")
        lines += ["", "#### Unsupported claims", ""]
        if a["unsupported_claims"]:
            for x in a["unsupported_claims"][:50]:
                lines.append(f"- Chapter {x['chapter']}: {x.get('claim', '')} — {x.get('note', '')}")
        else:
            lines.append("None detected.")
        lines += ["", "#### Contradictions", ""]
        if a["contradictions"]:
            for x in a["contradictions"][:50]:
                lines.append(f"- Chapter {x['chapter']} / {x['fact_id']}: {x['note']}")
        else:
            lines.append("None detected.")
        lines += ["", "#### Complete-book summary", ""]
        book_summary = item.get("book_summary") or ""
        if book_summary:
            lines += ["```markdown", book_summary, "```"]
        else:
            lines.append("No complete-book summary found for this run.")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def html_report(path: Path, report: dict) -> None:
    """Generate a self-contained interactive HTML evaluation dashboard."""
    import json as _json
    payload = _json.dumps(report, ensure_ascii=False).replace("</", "<\\/")

    template = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>EPUB LLM Evaluation Dashboard</title>
<style>
:root{--bg:#f6f7f9;--panel:#fff;--text:#17202a;--muted:#68737d;--line:#dfe3e7;--accent:#2563eb;--good:#16803c;--warn:#b7791f;--bad:#c53030;--partial:#8a6d1d}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
main{max-width:1500px;margin:0 auto;padding:24px}.top{position:sticky;top:0;z-index:10;background:rgba(246,247,249,.95);backdrop-filter:blur(8px);padding:12px 0;border-bottom:1px solid var(--line)}
h1{margin:0 0 4px;font-size:26px}h2{margin:28px 0 12px}h3{margin:18px 0 8px}.muted{color:var(--muted)}
.grid{display:grid;gap:12px}.cards{grid-template-columns:repeat(auto-fit,minmax(150px,1fr));margin:16px 0}.card,.panel{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px;box-shadow:0 1px 2px #00000008}.card b{display:block;font-size:23px}.card span{color:var(--muted);font-size:12px}
.controls{display:grid;grid-template-columns:repeat(auto-fit,minmax(145px,1fr));gap:10px;align-items:end}.control label{display:block;font-size:12px;color:var(--muted);margin-bottom:4px}.control input{width:100%;padding:6px 8px;border:1px solid #cbd2d8;border-radius:6px;background:white}.control button{padding:7px 10px;border:1px solid #cbd2d8;border-radius:6px;background:white;cursor:pointer}.control button:hover{background:#f0f2f4}
.charts{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:12px;align-items:start}.charts-left{display:flex;flex-direction:column;gap:12px;min-width:0}.charts>.charts-coverage{min-width:0;min-height:0}.charts .panel{min-width:0;min-height:0}.chart{min-height:0;overflow:visible}.charts-coverage #importanceChart{max-height:none}@media(max-width:800px){.charts{grid-template-columns:1fr}.charts-left{gap:12px}.charts>.charts-coverage{width:100%}}.scatter{width:100%;height:300px;display:block}.legend{display:flex;flex-wrap:wrap;gap:12px 18px;margin:8px 0 14px}.legend-item{display:inline-flex;align-items:center;gap:6px;font-size:12px;color:var(--muted)}.swatch{width:12px;height:12px;border-radius:3px;display:inline-block;border:1px solid #00000018}.barrow{display:grid;grid-template-columns:minmax(140px,1fr) 3fr 55px;gap:8px;align-items:center;margin:9px 0}.bar{height:18px;background:#edf0f2;border-radius:4px;overflow:hidden}.fill{height:100%;background:var(--accent)}.fill.good{background:var(--good)}
.stack{display:flex;height:24px;border-radius:4px;overflow:hidden;background:#eee}.stack span{height:100%;min-width:1px}.supported{background:#48a868}.partial{background:#d4a72c}.missing{background:#cbd1d6}.contradicted{background:#d9534f}
.sort{cursor:pointer;user-select:none;white-space:nowrap}.sort:after{content:" ↕";color:#9aa1a8}.sort.asc:after{content:" ↑"}.sort.desc:after{content:" ↓"}
table{border-collapse:separate;border-spacing:0;width:max-content;min-width:100%;background:white;table-layout:auto}th,td{border-bottom:1px solid var(--line);padding:8px;text-align:left;vertical-align:top}th{background:#f0f3f6;z-index:2;white-space:nowrap}th:first-child,td:first-child{min-width:220px}td{white-space:nowrap}.tablewrap{overflow-x:auto;overflow-y:visible;max-width:100%;-webkit-overflow-scrolling:touch}td.num,th.num{text-align:right}tr:hover td{background:#fafbfc}th.tip{position:relative;cursor:help}th.tip::after{content:"ⓘ";display:inline-block;margin-left:5px;font-size:11px;font-weight:600;color:#68737d;vertical-align:1px}.table-tooltip{position:fixed;z-index:9999;max-width:330px;padding:9px 11px;border-radius:7px;background:#17202a;color:#fff;font-size:12px;line-height:1.4;box-shadow:0 4px 14px rgba(0,0,0,.2);pointer-events:none;white-space:normal}
.pill{display:inline-block;border-radius:999px;padding:2px 7px;font-size:12px;background:#edf0f2}.score{font-weight:700}.details{margin-top:12px}.details>summary{cursor:pointer;font-weight:700;padding:10px;background:#fff;border:1px solid var(--line);border-radius:8px}.details[open]>summary{border-radius:8px 8px 0 0}
.detailbody{background:#fff;border:1px solid var(--line);border-top:0;padding:12px}.booksummary{margin-top:10px;padding:12px;background:#f8fafc;border:1px solid var(--line);border-radius:8px;white-space:pre-wrap;overflow:auto;max-height:900px;font:13px/1.55 ui-monospace,SFMono-Regular,Menlo,monospace}.subdetails{margin:8px 0}.subdetails summary{cursor:pointer;font-weight:600}.issue{padding:7px 0;border-bottom:1px solid #eef0f2}.issue small{color:var(--muted)}
.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:12px;color:var(--muted)}.dot{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px}
.search{width:100%;padding:9px;border:1px solid #cbd2d8;border-radius:7px;margin:8px 0 12px}.note{background:#fff8e6;border:1px solid #f1d58a;padding:10px;border-radius:8px}
@media(max-width:800px){.charts{grid-template-columns:1fr}main{padding:12px}th{top:115px}}
</style>
</head>
<body><main>
<div class="top">
<h1>EPUB LLM Evaluation Dashboard</h1>
<div class="muted" id="subtitle"></div>
</div>

<section class="panel" style="margin-top:14px">
<h2 style="margin-top:0">Scoring controls</h2>
<div class="controls">
<div class="control"><label>Critical weight</label><input id="wCritical" type="number" min="0" step=".1" value="4"></div>
<div class="control"><label>Major weight</label><input id="wMajor" type="number" min="0" step=".1" value="3"></div>
<div class="control"><label>Moderate weight</label><input id="wModerate" type="number" min="0" step=".1" value="2"></div>
<div class="control"><label>Minor weight</label><input id="wMinor" type="number" min="0" step=".1" value="1"></div>
<div class="control"><label>Partial credit</label><input id="partial" type="number" min="0" max="1" step=".05" value=".5"></div>
<div class="control"><label>Omission penalty (pp / 100% weighted missing)</label><input id="pMissing" type="number" min="0" step=".1" value="10"></div>
<div class="control"><label>Contradiction penalty (pp / 100% weighted contradicted)</label><input id="pContradicted" type="number" min="0" step=".1" value="25"></div>
<div class="control"><label>Unsupported-claim penalty (pp / 100 claims)</label><input id="pUnsupported" type="number" min="0" step=".1" value="5"></div>
<div class="control"><label>Overall factual weight</label><input id="overallFact" type="number" min="0" max="100" step="5" value="70"></div>
<div class="control"><label>Overall story-arc weight</label><input id="overallStory" type="number" min="0" max="100" step="5" value="30"></div>
<div class="control"><button onclick="resetWeights()">Reset defaults</button></div>
</div>
<p class="muted">Adjusted score = weighted factual coverage − omission penalty − contradiction penalty − unsupported-claim penalty. Overall score = adjusted factual score × factual weight + story-arc coverage × story-arc weight. Overall is a reporting/analysis metric, not a new evaluator judgment. Default: 70% factual + 30% story arc.</p>
</section>

<div class="cards grid" id="cards"></div>

<section class="charts">
<div class="charts-left">
<div class="panel chart"><h2>Overall score</h2><div class="muted" style="font-size:12px;margin:-4px 0 8px">70% factual + 30% story arc by default; adjustable above.</div><div id="scoreChart"></div></div>
<div class="panel chart"><h2>Runtime vs. weighted coverage</h2><div id="runtimeCoverageChart"></div></div>
</div>
<div class="panel chart charts-coverage"><h2>Coverage by importance</h2><div id="importanceLegend" class="legend"></div><div id="importanceChart"></div></div>
</section>

<section class="panel">
<h2>Model comparison</h2>
<input class="search" id="modelSearch" placeholder="Filter models…">
<div class="tablewrap"><table><thead><tr id="head"></tr></thead><tbody id="tbody"></tbody></table></div>
</section>

<section><h2>Detailed results</h2><div id="details"></div></section>
</main>

<script>
const REPORT=__REPORT_JSON__;
const excluded=/deepseek-r1-distill-qwen-32b/i;
let models=REPORT.models.filter(m=>!excluded.test(m.label));
const defaults={critical:4,major:3,moderate:2,minor:1,partial:.5,pMissing:10,pContradicted:25,pUnsupported:5,overallFact:70,overallStory:30};
const esc=s=>String(s??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const pct=x=>(100*(x||0)).toFixed(1)+"%";
const mins=x=>(x||0).toFixed(1)+" min";
function val(id){return parseFloat(document.getElementById(id).value)||0}
function weights(){return {critical:val("wCritical"),major:val("wMajor"),moderate:val("wModerate"),minor:val("wMinor")}}
function metrics(m){
  const a=m.aggregate||{}, c=a.claims||{}, fc=a.fact_counts||{};
  const weights={critical:val("wCritical"),major:val("wMajor"),moderate:val("wModerate"),minor:val("wMinor")};
  let total=0,hit=0;
  for(const k of Object.keys(weights)){const x=fc[k]||{};const n=(x.supported||0)+(x.partial||0)+(x.missing||0)+(x.contradicted||0);const h=(x.supported||0)+val("partial")*(x.partial||0);total+=n*weights[k];hit+=h*weights[k];}
  const base=total?hit/total:1;
  const allFacts=Object.values(fc).reduce((s,x)=>s+(x.supported||0)+(x.partial||0)+(x.missing||0)+(x.contradicted||0),0);
  const missing=Object.values(fc).reduce((s,x)=>s+(x.missing||0),0);
  const contradicted=Object.values(fc).reduce((s,x)=>s+(x.contradicted||0),0);
  const missRate=allFacts?missing/allFacts:0, contraRate=allFacts?contradicted/allFacts:0;
  const unsupportedRate=(c.unsupported||0)/Math.max(1,(c.supported||0)+(c.partial||0)+(c.unsupported||0)+(c.contradicted||0));
  const adjusted=Math.max(0,base-val("pMissing")/100*missRate-val("pContradicted")/100*contraRate-val("pUnsupported")/100*unsupportedRate);
  const fw=val("overallFact")/100, sw=val("overallStory")/100, sum=fw+sw;
  const factWeight=sum?fw/sum:.7, storyWeight=sum?sw/sum:.3;
  const story=Number(m.book_evaluation?.story_arc_coverage||0);
  const overall=adjusted*factWeight+story*storyWeight;
  return {base,adjusted,story,overall,factWeight,storyWeight,missRate,contraRate,unsupportedRate};
}
function resetWeights(){for(const [k,v] of Object.entries(defaults)){const id={critical:"wCritical",major:"wMajor",moderate:"wModerate",minor:"wMinor",partial:"partial",pMissing:"pMissing",pContradicted:"pContradicted",pUnsupported:"pUnsupported",overallFact:"overallFact",overallStory:"overallStory"}[k];document.getElementById(id).value=v}render()}
function label(m){return m.model_info?.display_name||m.model||m.label}
function renderCards(){
 const avg=models.reduce((s,m)=>s+metrics(m).adjusted,0)/Math.max(1,models.length);
 const avgOverall=models.reduce((s,m)=>s+metrics(m).overall,0)/Math.max(1,models.length);
 const avgStoryArc=models.reduce((s,m)=>s+(m.book_evaluation?.story_arc_coverage||0),0)/Math.max(1,models.length);
 const best=models.reduce((b,m)=>!b||metrics(m).overall>metrics(b).overall?m:b,null);
 const totalFacts=models.length?Object.values(models[0].aggregate.fact_counts).reduce((s,x)=>s+Object.values(x).reduce((a,b)=>a+b,0),0):0;
 document.getElementById("cards").innerHTML=[
 `<div class="card"><b>${models.length}</b><span>models included</span></div>`,
 `<div class="card"><b>${pct(avgOverall)}</b><span>average overall score</span></div>`,
 `<div class="card"><b>${pct(avgStoryArc)}</b><span>average story arc coverage</span></div>`,
 `<div class="card"><b>${pct(avg)}</b><span>average adjusted factual score</span></div>`,
 `<div class="card"><b>${pct(best?metrics(best).overall:0)}</b><span>highest overall score</span></div>`,
 `<div class="card"><b>${totalFacts}</b><span>reference facts / model</span></div>`,
 `<div class="card"><b>${esc(REPORT.evaluator?.display_name||REPORT.evaluator?.model||"?")}</b><span>evaluator</span></div>`
 ].join("");
}
function renderScoreChart(){
 const arr=[...models].sort((a,b)=>metrics(b).overall-metrics(a).overall);
 document.getElementById("scoreChart").innerHTML=arr.map(m=>{const x=metrics(m);return `<div class="barrow"><span>${esc(label(m))}</span><div class="bar"><div class="fill good" style="width:${x.overall*100}%"></div></div><b>${pct(x.overall)}</b></div>`}).join("");
}
function renderImportance(){
 const colors={
   supported:["supported","Fully supported / covered"],
   partial:["partial","Partially supported"],
   missing:["missing","Missing / omitted"],
   contradicted:["contradicted","Contradicted"]
 };
 document.getElementById("importanceLegend").innerHTML=Object.entries(colors).map(([k,[cls,title]])=>
   `<span class="legend-item" title="${title}"><span class="swatch ${cls}"></span>${title}</span>`).join("");
 document.getElementById("importanceChart").innerHTML=models.map(m=>{
   const a=m.aggregate,c=a.fact_counts;
   const parts=["critical","major","moderate","minor"].map(imp=>{
     const x=c[imp], total=x.supported+x.partial+x.missing+x.contradicted||1;
     const seg=(key)=>`<span class="${colors[key][0]}" title="${colors[key][1]} — ${imp}: ${x[key]} / ${total} (${(100*x[key]/total).toFixed(1)}%)" style="width:${100*x[key]/total}%"></span>`;
     return `<div style="margin:10px 0"><div><b>${imp}</b> <span class="muted">${pct(a.coverage[imp])}</span></div><div class="stack">${seg("supported")}${seg("partial")}${seg("missing")}${seg("contradicted")}</div></div>`;
   }).join("");
   return `<div><b>${esc(label(m))}</b>${parts}</div>`;
 }).join("");
}
function renderRuntimeCoverage(){
 const el=document.getElementById("runtimeCoverageChart");
 if(!models.length){el.innerHTML="";return}
 const W=900,H=320,PL=64,PR=28,PT=24,PB=62;
 const rawMaxX=Math.max(...models.map(m=>Number(m.runtime_minutes)||0),1);
 // Round the X-axis ceiling up so the last point has room and the axis gets useful ticks.
 const step=rawMaxX<=30?5:rawMaxX<=60?10:20;
 const maxX=Math.max(step,Math.ceil(rawMaxX/step)*step);
 const maxY=1;
 const x=v=>PL+(v/maxX)*(W-PL-PR);
 const y=v=>H-PB-v/maxY*(H-PT-PB);

 const yGrid=[0,.25,.5,.75,1].map(v=>
   `<line x1="${PL}" y1="${y(v)}" x2="${W-PR}" y2="${y(v)}" stroke="#dfe3e7"/>`+
   `<text x="${PL-8}" y="${y(v)+4}" text-anchor="end" font-size="11" fill="#68737d">${(v*100).toFixed(0)}%</text>`
 ).join("");

 const xTicks=[];
 for(let v=0; v<=maxX; v+=step){
   const xx=x(v);
   xTicks.push(
     `<line x1="${xx}" y1="${H-PB}" x2="${xx}" y2="${H-PB+5}" stroke="#9aa1a8"/>`+
     `<text x="${xx}" y="${H-PB+20}" text-anchor="middle" font-size="11" fill="#68737d">${v}</text>`
   );
 }
 const xTickMarkup=xTicks.join("");

 const dots=models.map(m=>{
   const x0=x(Number(m.runtime_minutes)||0), y0=y(m.aggregate.weighted_coverage||0);
   const rightSide=x0>W-PR-150;
   const labelX=rightSide ? W-PR-5 : Math.min(x0+9,W-PR-5);
   const anchor=rightSide ? "end" : "start";
   return `<g>`+
     `<circle cx="${x0}" cy="${y0}" r="6" fill="var(--accent)" stroke="white" stroke-width="2">`+
     `<title>${esc(label(m))} — ${mins(m.runtime_minutes)} — ${pct(m.aggregate.weighted_coverage)}</title>`+
     `</circle>`+
     `<text x="${labelX}" y="${y0+4}" text-anchor="${anchor}" font-size="10" fill="#17202a">${esc(label(m))}</text>`+
     `</g>`;
 }).join("");

 el.innerHTML=`<svg class="scatter" viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet" style="overflow:visible" role="img" aria-label="Runtime versus weighted coverage">`+
   `${yGrid}${xTickMarkup}`+
   `<line x1="${PL}" y1="${H-PB}" x2="${W-PR}" y2="${H-PB}" stroke="#9aa1a8"/>`+
   `<line x1="${PL}" y1="${PT}" x2="${PL}" y2="${H-PB}" stroke="#9aa1a8"/>`+
   `<text x="${W/2}" y="${H-12}" text-anchor="middle" font-size="12" fill="#68737d">Runtime (minutes)</text>`+
   `<text x="15" y="${H/2}" transform="rotate(-90 15 ${H/2})" text-anchor="middle" font-size="12" fill="#68737d">Weighted coverage</text>`+
   `${dots}</svg>`;
}
const columns=[
 ["model","Model","Model/run being compared."],["overall","Overall","Combined reporting score: adjusted factual score × factual weight + story-arc coverage × story-arc weight. Default 70% factual + 30% story arc; weights are adjustable."],["adjusted","Adjusted","Adjusted factual score: weighted factual coverage after subtracting the omission, contradiction, and unsupported-claim penalties. The penalties use the current controls above."],["storyArc","Story arc","Evaluator's book-level assessment of how well the summary represents the broader story arcs, major turning points, consequences, and overall story."],["base","Base weighted","Weighted factual coverage before penalties. Fact importance weights and the partial-credit setting are applied, but omission/contradiction/unsupported-claim penalties are not."],["runtime","Runtime","Time required to generate the model's summary, in minutes."],["critical","Critical","Coverage of reference facts classified as critical. Supported facts count fully; partial facts receive the configured partial-credit value."],["major","Major","Coverage of reference facts classified as major."],["moderate","Moderate","Coverage of reference facts classified as moderate."],["minor","Minor","Coverage of reference facts classified as minor."],["contradicted","Contradictions","Number of model claims judged contradicted by the reference/source."],["unsupported","Unsupported","Number of model claims judged unsupported by the reference/source."],["criticalO","Critical omissions","Number of critical reference facts omitted or not adequately covered."],["majorO","Major omissions","Number of major reference facts omitted or not adequately covered."]
];
let sortKey="overall",sortDir=-1;
function renderHead(){
 document.getElementById("head").innerHTML=columns.map(([k,n,tip])=>`<th class="${k!=="model"?"num ":""}tip sort ${sortKey===k?(sortDir>0?"asc":"desc"):""}" data-tip="${esc(tip)}" onclick="sortBy('${k}')">${n}</th>`).join("");
 document.querySelectorAll("#head th.tip").forEach(th=>{
   th.addEventListener("mouseenter",()=>showTableTip(th));
   th.addEventListener("mouseleave",hideTableTip);
   th.addEventListener("focus",()=>showTableTip(th));
   th.addEventListener("blur",hideTableTip);
 });
}
let tableTipEl=null;
function showTableTip(th){
 if(tableTipEl)tableTipEl.remove();
 tableTipEl=document.createElement("div"); tableTipEl.className="table-tooltip"; tableTipEl.textContent=th.dataset.tip||"";
 document.body.appendChild(tableTipEl);
 const r=th.getBoundingClientRect(), pad=8;
 let left=Math.max(pad,Math.min(r.left,r.right-330));
 let top=r.bottom+8;
 const h=tableTipEl.offsetHeight;
 if(top+h>window.innerHeight-pad)top=Math.max(pad,r.top-h-8);
 tableTipEl.style.left=left+"px"; tableTipEl.style.top=top+"px";
}
function hideTableTip(){if(tableTipEl){tableTipEl.remove();tableTipEl=null;}}
function rowValue(m,k){
 const a=m.aggregate,x=metrics(m);
 return {model:label(m),overall:x.overall,adjusted:x.adjusted,storyArc:x.story,base:x.base,runtime:m.runtime_minutes||0,critical:a.coverage.critical||0,major:a.coverage.major||0,moderate:a.coverage.moderate||0,minor:a.coverage.minor||0,contradicted:a.claims?.contradicted||0,unsupported:a.claims?.unsupported||0,criticalO:a.critical_omissions?.length||0,majorO:a.major_omissions?.length||0}[k];
}
function renderTable(){
 const q=(document.getElementById("modelSearch").value||"").toLowerCase();
 let arr=models.filter(m=>label(m).toLowerCase().includes(q)||m.label.toLowerCase().includes(q));
 arr.sort((a,b)=>{let x=rowValue(a,sortKey),y=rowValue(b,sortKey);if(typeof x==="string")return sortDir*x.localeCompare(y);return sortDir*(x-y)});
 document.getElementById("tbody").innerHTML=arr.map(m=>{const a=m.aggregate,x=metrics(m);return `<tr>
 <td><b>${esc(label(m))}</b><br><span class="muted">${esc(m.label)}</span></td>
 <td class="num score">${pct(x.overall)}</td><td class="num score">${pct(x.adjusted)}</td><td class="num score">${pct(x.story)}</td><td class="num">${pct(x.base)}</td><td class="num">${mins(m.runtime_minutes)}</td>
 <td class="num">${pct(a.coverage.critical)}</td><td class="num">${pct(a.coverage.major)}</td><td class="num">${pct(a.coverage.moderate)}</td><td class="num">${pct(a.coverage.minor)}</td>
 <td class="num">${a.claims?.contradicted||0}</td><td class="num">${a.claims?.unsupported||0}</td><td class="num">${a.critical_omissions?.length||0}</td><td class="num">${a.major_omissions?.length||0}</td>
 </tr>`}).join("");
}
function sortBy(k){if(sortKey===k)sortDir*=-1;else{sortKey=k;sortDir=-1}render()}
function issueList(arr,limit=100){
 if(!arr?.length)return `<div class="muted">None detected.</div>`;
 return arr.slice(0,limit).map(x=>`<div class="issue"><b>Chapter ${esc(x.chapter??"?")}${x.fact_id?` / ${esc(x.fact_id)}`:""}</b><br>${esc(x.note||x.claim||"")}</div>`).join("")+(arr.length>limit?`<div class="muted">Showing first ${limit} of ${arr.length}.</div>`:"");
}
function renderDetails(){
 document.getElementById("details").innerHTML=models.map((m,i)=>{const a=m.aggregate,x=metrics(m),mi=m.model_info||{},b=m.book_evaluation||{};
 const counts=a.fact_counts;
 return `<details class="details"><summary>${esc(label(m))} — <span class="score">${pct(x.overall)}</span> overall / <span class="score">${pct(x.adjusted)}</span> adjusted / <span class="score">${pct(b.story_arc_coverage)}</span> story arc / <span class="score">${pct(x.base)}</span> weighted · ${mins(m.runtime_minutes)}</summary>
 <div class="detailbody">
 <div class="grid cards" style="margin:0 0 10px;grid-template-columns:repeat(auto-fit,minmax(120px,1fr))">
 <div class="card"><b>${pct(x.overall)}</b><span>overall</span></div><div class="card"><b>${pct(x.adjusted)}</b><span>adjusted factual</span></div><div class="card"><b>${pct(b.story_arc_coverage)}</b><span>story arc coverage</span></div><div class="card"><b>${pct(x.base)}</b><span>base weighted</span></div>
 <div class="card"><b>${a.claims?.contradicted||0}</b><span>contradictions</span></div><div class="card"><b>${a.claims?.unsupported||0}</b><span>unsupported claims</span></div>
 </div>
 <p><b>Model:</b> ${esc(mi.display_name||m.model)} · <b>Key:</b> ${esc(m.model)} · <b>Variant:</b> ${esc(mi.selected_variant||"?")} · <b>Quantization:</b> ${esc(mi.quantization?.name||"?")} · <b>Runtime:</b> ${mins(m.runtime_minutes)} · <b>Prompt:</b> ${(m.prompt_tokens||0).toLocaleString()} · <b>Completion:</b> ${(m.completion_tokens||0).toLocaleString()}</p>
 <h3>Fact coverage</h3><div class="tablewrap"><table><tr><th>Importance</th><th>Coverage</th><th>Total</th><th>Supported</th><th>Partial</th><th>Missing</th><th>Contradicted</th></tr>
 ${["critical","major","moderate","minor"].map(imp=>{const c=counts[imp],t=Object.values(c).reduce((s,v)=>s+v,0);return `<tr><td>${imp}</td><td>${pct(a.coverage[imp])}</td><td>${t}</td><td>${c.supported}</td><td>${c.partial}</td><td>${c.missing}</td><td>${c.contradicted}</td></tr>`}).join("")}</table></div>
 <details class="subdetails"><summary>Critical omissions (${a.critical_omissions?.length||0})</summary>${issueList(a.critical_omissions)}</details>
 <details class="subdetails"><summary>Major omissions (${a.major_omissions?.length||0})</summary>${issueList(a.major_omissions)}</details>
 <details class="subdetails"><summary>Unsupported claims (${a.unsupported_claims?.length||0})</summary>${issueList(a.unsupported_claims)}</details>
 <details class="subdetails"><summary>Contradictions (${a.contradictions?.length||0})</summary>${issueList(a.contradictions)}</details>
 <details class="subdetails"><summary>Book-level evaluation</summary>
 <div><b>Overall:</b> ${pct(x.overall)} · <b>Adjusted factual:</b> ${pct(x.adjusted)} · <b>Story arc coverage:</b> ${pct(b.story_arc_coverage)} · <b>Weighted:</b> ${pct(b.weighted_coverage)} · <b>Critical:</b> ${pct(b.critical_coverage)} · <b>Major:</b> ${pct(b.major_coverage)} · <b>Moderate:</b> ${pct(b.moderate_coverage)} · <b>Minor:</b> ${pct(b.minor_coverage)}</div>
 <details class="subdetails"><summary>Book critical omissions</summary>${issueList((b.critical_omissions||[]).map(x=>({note:x})))} </details>
 <details class="subdetails"><summary>Book major omissions</summary>${issueList((b.major_omissions||[]).map(x=>({note:x})))} </details>
 <details class="subdetails"><summary>Book unsupported claims</summary>${issueList((b.unsupported_claims||[]).map(x=>({note:x})))} </details>
 <details class="subdetails"><summary>Book contradictions</summary>${issueList((b.contradictions||[]).map(x=>({note:x})))} </details>
 </details>
 <details class="subdetails"><summary>Complete-book summary</summary><div class="booksummary">${esc(m.book_summary||"No complete-book summary found for this run.")}</div></details>
 </div></details>`}).join("");
}
function render(){renderCards();renderScoreChart();renderImportance();renderRuntimeCoverage();renderHead();renderTable();renderDetails()}
document.querySelectorAll(".control input").forEach(x=>x.addEventListener("input",render));
document.getElementById("modelSearch").addEventListener("input",renderTable);
document.getElementById("subtitle").textContent=`${REPORT.book_title} · generated ${REPORT.generated_at} · evaluator ${REPORT.evaluator?.display_name||REPORT.evaluator?.model||"?"}`;
render();
</script>
</body></html>"""
    path.write_text(template.replace("__REPORT_JSON__", payload), encoding="utf-8")
def run(args):
    epub_path = Path(args.epub).expanduser().resolve()
    reference_path = Path(args.reference).expanduser().resolve()
    results_dir = Path(args.results).expanduser().resolve()

    book_title, book_language, chapters = read_epub(epub_path)
    reference_text = reference_path.read_text(encoding="utf-8")
    reference_sections = parse_reference_sections(reference_text)
    if not reference_sections:
        raise RuntimeError(
            "Could not detect chapter sections in the NotebookLM summary. "
            "Expected headings such as '### Chapter 1' and '### Chapter 2'."
        )

    runs = load_model_runs(results_dir)
    if not runs:
        raise RuntimeError(f"No benchmark runs found below {results_dir}")

    if args.evaluator == "local":
        evaluator_model, evaluator_info, evaluator_context = get_loaded_model(args.base_url)
    elif args.evaluator == "claude-code":
        evaluator_model = args.evaluator_model or "opus"
        evaluator_info = {
            "display_name": evaluator_model,
            "selected_variant": None,
            "quantization": None,
            "provider": "claude-code",
        }
        evaluator_context = 0
    else:
        if not args.evaluator_model:
            raise RuntimeError(f"--evaluator-model is required when --evaluator={args.evaluator}")
        evaluator_model = args.evaluator_model
        evaluator_info = {
            "display_name": evaluator_model,
            "selected_variant": None,
            "quantization": None,
            "provider": args.evaluator,
        }
        evaluator_context = 0

    # Evaluation runs are explicit: create a new run by default, or resume
    # exactly the run selected with --resume. Never guess between multiple
    # incomplete evaluations.
    evaluation_root = results_dir / "evaluation"
    evaluation_root.mkdir(parents=True, exist_ok=True)
    eval_root: Path

    if args.resume:
        eval_root = Path(args.resume).expanduser().resolve()
        if not eval_root.is_dir():
            raise RuntimeError(f"Resume path does not exist or is not a directory: {eval_root}")
        if not eval_root.is_relative_to(evaluation_root):
            raise RuntimeError(
                f"Resume path must be inside the results evaluation directory {evaluation_root}: {eval_root}"
            )
        resuming = True
    else:
        eval_root = evaluation_root / datetime.now().astimezone().strftime("run-%Y%m%d-%H%M%S")
        eval_root.mkdir(parents=True, exist_ok=False)
        resuming = False

    if resuming:
        print(f"Resuming evaluation: {eval_root}")

    print("=" * 70)
    print(" EPUB LLM SUMMARY EVALUATION")
    print("=" * 70)
    print(f"Book:       {book_title}")
    print(f"Language:   {book_language or 'metadata unavailable'}")
    print(f"Chapters:   {len(chapters)}")
    print(f"Reference:  {reference_path}")
    print(f"Runs:       {len(runs)}")
    print(f"Evaluator:  {args.evaluator}")
    print(f"Eval model: {evaluator_info.get('display_name', evaluator_model)}")
    if evaluator_context:
        print(f"Context:    {evaluator_context:,} tokens")
    print(f"Output:     {eval_root}")
    print()

    save_json(eval_root / "evaluator_info.json", {
        "provider": args.evaluator,
        "model": evaluator_model,
        "source_language": book_language,
        "display_name": evaluator_info.get("display_name", evaluator_model),
        "selected_variant": evaluator_info.get("selected_variant"),
        "quantization": evaluator_info.get("quantization"),
        "context": evaluator_context,
        "temperature": args.temperature,
        "reasoning": args.reasoning,
        "base_url": args.base_url,
        "resumed": resuming,
    })

    all_reference_facts: dict[str, list[dict]] = {}
    reference_elapsed = 0.0
    ref_prompt_tokens = ref_completion_tokens = 0

    # Phase 1: reference fact extraction.
    print("Phase 1/3: building importance-tagged reference facts...")
    ref_dir = eval_root / "reference"
    ref_dir.mkdir(parents=True, exist_ok=True)
    for pos, chapter in enumerate(chapters, 1):
        ref_path = ref_dir / (f"{int(chapter.number):03d}.json" if chapter.number.isdigit() else "epilogue.json")
        if resuming and ref_path.exists():
            try:
                cached = read_json(ref_path)
                cached_facts = cached.get("facts")
                if isinstance(cached_facts, list):
                    all_reference_facts[chapter.number] = cached_facts
                    cached_usage = cached.get("usage") or {}
                    reference_elapsed += float(cached.get("elapsed_seconds", 0) or 0)
                    ref_prompt_tokens += int(cached_usage.get("prompt_tokens", 0) or 0)
                    ref_completion_tokens += int(cached_usage.get("completion_tokens", 0) or 0)
                    print(f"  [{pos:02d}/{len(chapters)}] {chapter.title}: {len(cached_facts)} facts [CACHED]")
                    continue
            except Exception:
                pass

        reference_section = reference_sections.get(chapter.number, "")
        if not reference_section:
            print(f"  [{pos:02d}/{len(chapters)}] {chapter.title}: no matching NotebookLM section")
            all_reference_facts[chapter.number] = []
            continue
        prompt = render_prompt(
            FACT_PROMPT,
            book_title=book_title,
            book_language=book_language or "same as source text",
            chapter_title=chapter.title,
            reference=reference_section,
            source=chapter.text,
        )
        budget = min(args.max_tokens_facts, max(args.min_output_tokens, dynamic_budget(prompt, args.max_tokens_facts, args.min_output_tokens)))
        raw, usage, elapsed, _ = chat(args.evaluator, args.base_url, evaluator_model, SYSTEM, prompt, budget, args.temperature, args.reasoning, args.timeout)
        parsed = extract_json(raw)
        facts = parsed.get("facts", []) if isinstance(parsed, dict) else []
        # Ensure stable IDs even if evaluator omitted/duplicated IDs.
        normalized = []
        for i, fact in enumerate(facts, 1):
            if not isinstance(fact, dict) or not fact.get("fact"):
                continue
            normalized.append({
                "id": fact.get("id") or f"{chapter.number}-F{i:03d}",
                "fact": fact["fact"],
                "importance": fact.get("importance", "moderate"),
                "type": fact.get("type", "other"),
            })
        all_reference_facts[chapter.number] = normalized
        save_json(ref_dir / (f"{int(chapter.number):03d}.json" if chapter.number.isdigit() else "epilogue.json"), {
            "chapter": chapter.number,
            "title": chapter.title,
            "facts": normalized,
            "elapsed_seconds": elapsed,
            "usage": usage,
        })
        reference_elapsed += elapsed
        ref_prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
        ref_completion_tokens += int(usage.get("completion_tokens", 0) or 0)
        print(f"  [{pos:02d}/{len(chapters)}] {chapter.title}: {len(normalized)} facts")

    # Flatten reference facts for book-level evaluation.
    flat_facts = []
    for chapter in chapters:
        for fact in all_reference_facts.get(chapter.number, []):
            flat_facts.append({"chapter": chapter.number, **fact})
    save_json(eval_root / "reference_facts.json", {"book_title": book_title, "source_language": book_language, "facts": flat_facts})

    # Phase 2: evaluate each run chapter by chapter.
    model_reports = []
    print("\nPhase 2/3: evaluating local model summaries...")
    for run_index, run_dir in enumerate(runs, 1):
        book_info = read_json(run_dir / "book_info.json")
        run_stats = read_json(run_dir / "run_stats.json") if (run_dir / "run_stats.json").exists() else {}
        model_key = book_info.get("model", run_dir.parent.name)
        label = f"{run_dir.parent.name}/{run_dir.name}"
        model_out = eval_root / safe_name(run_dir.parent.name) / run_dir.name
        model_out.mkdir(parents=True, exist_ok=True)
        chapter_summaries = read_model_chapter_summaries(run_dir, chapters)
        chapter_results = []
        total_eval_elapsed = 0.0
        total_eval_prompt = total_eval_completion = 0

        print(f"\n  [{run_index}/{len(runs)}] {label}")
        for pos, chapter in enumerate(chapters, 1):
            result_path = model_out / (f"{int(chapter.number):03d}.json" if chapter.number.isdigit() else "epilogue.json")
            if resuming and result_path.exists():
                try:
                    cached_result = read_json(result_path)
                    if isinstance(cached_result, dict) and "fact_results" in cached_result and "model_claims" in cached_result:
                        chapter_results.append(cached_result)
                        cached_usage = cached_result.get("usage") or {}
                        total_eval_elapsed += float(cached_result.get("elapsed_seconds", 0) or 0)
                        total_eval_prompt += int(cached_usage.get("prompt_tokens", 0) or 0)
                        total_eval_completion += int(cached_usage.get("completion_tokens", 0) or 0)
                        print(f"    [{pos:02d}/{len(chapters)}] {chapter.title} [CACHED]")
                        continue
                except Exception:
                    pass

            facts = all_reference_facts.get(chapter.number, [])
            summary = chapter_summaries.get(chapter.number, "")
            if not summary:
                result = {
                    "chapter": chapter.number,
                    "title": chapter.title,
                    "fact_results": [
                        {"fact_id": f["id"], "status": "missing", "importance": f["importance"], "note": "No local-model chapter summary found."}
                        for f in facts
                    ],
                    "model_claims": [],
                }
                chapter_results.append(result)
                continue

            facts_text = json.dumps(facts, ensure_ascii=False, indent=2)
            prompt = render_prompt(
                EVAL_PROMPT,
                book_title=book_title,
                book_language=book_language or "same as source text",
                chapter_title=chapter.title,
                facts=facts_text,
                summary=summary,
                source=chapter.text,
            )
            budget = min(args.max_tokens_eval, max(args.min_output_tokens, dynamic_budget(prompt, args.max_tokens_eval, args.min_output_tokens)))
            parsed, usage, elapsed, _ = chat_json_with_retry(
                args.evaluator, args.base_url, evaluator_model, SYSTEM, prompt,
                budget, args.max_tokens_eval, args.temperature, args.reasoning, args.timeout
            )
            result = {
                "chapter": chapter.number,
                "title": chapter.title,
                "fact_results": parsed.get("fact_results", []) if isinstance(parsed, dict) else [],
                "model_claims": parsed.get("model_claims", []) if isinstance(parsed, dict) else [],
                "elapsed_seconds": elapsed,
                "usage": usage,
            }
            chapter_results.append(result)
            save_json(model_out / (f"{int(chapter.number):03d}.json" if chapter.number.isdigit() else "epilogue.json"), result)
            total_eval_elapsed += elapsed
            total_eval_prompt += int(usage.get("prompt_tokens", 0) or 0)
            total_eval_completion += int(usage.get("completion_tokens", 0) or 0)
            print(f"    [{pos:02d}/{len(chapters)}] {chapter.title}")

        aggregate = aggregate_chapter_evaluation(chapter_results)
        save_json(model_out / "evaluation.json", aggregate)

        model_report = {
            "label": label,
            "run_dir": str(run_dir),
            "model": model_key,
            "model_info": book_info.get("model_info") or {},
            "runtime_seconds": float(run_stats.get("total_elapsed_seconds", 0) or 0),
            "runtime_minutes": float(run_stats.get("total_elapsed_seconds", 0) or 0) / 60,
            "prompt_tokens": int(run_stats.get("total_prompt_tokens", 0) or 0),
            "completion_tokens": int(run_stats.get("total_completion_tokens", 0) or 0),
            "book_summary": read_book_summary(run_dir),
            "evaluation_runtime_seconds": total_eval_elapsed,
            "evaluation_prompt_tokens": total_eval_prompt,
            "evaluation_completion_tokens": total_eval_completion,
            "aggregate": aggregate,
        }
        model_reports.append(model_report)

    # Phase 3: book-level summary evaluation.
    print("\nPhase 3/3: evaluating complete-book summaries...")
    book_level = []
    # Chapter-level reference facts have already been checked against the EPUB,
    # so the full book text is not sent again at book level.
    facts_text = json.dumps(flat_facts, ensure_ascii=False, indent=2)
    for item in model_reports:
        run_dir = Path(item["run_dir"])
        book_eval_path = eval_root / safe_name(Path(item["run_dir"]).parent.name) / Path(item["run_dir"]).name / "book_evaluation.json"
        if resuming and book_eval_path.exists():
            try:
                cached_book = read_json(book_eval_path)
                if isinstance(cached_book, dict) and "result" in cached_book:
                    book_level.append(cached_book)
                    print(f"  {item['label']} [CACHED]")
                    continue
            except Exception:
                pass

        summary = read_book_summary(run_dir)
        if not summary:
            continue
        prompt = render_prompt(
            BOOK_EVAL_PROMPT,
            facts=facts_text,
            summary=summary,
        )
        budget = min(args.max_tokens_book_eval, max(args.min_output_tokens, dynamic_budget(prompt, args.max_tokens_book_eval, args.min_output_tokens)))
        parsed, usage, elapsed, _ = chat_json_with_retry(
            args.evaluator, args.base_url, evaluator_model, SYSTEM, prompt,
            budget, args.max_tokens_book_eval, args.temperature, args.reasoning, args.timeout
        )
        result = {
            "model": item["model"],
            "run_dir": item["run_dir"],
            "result": parsed,
            "elapsed_seconds": elapsed,
            "usage": usage,
        }
        book_level.append(result)
        save_json(eval_root / safe_name(Path(item["run_dir"]).parent.name) / Path(item["run_dir"]).name / "book_evaluation.json", result)

    for item in model_reports:
        match = next((x for x in book_level if isinstance(x, dict) and x.get("run_dir") == item["run_dir"]), None)
        item["book_evaluation"] = _safe_dict(match.get("result")) if isinstance(match, dict) else {}
        item["model_info"] = _safe_dict(item.get("model_info"))
        item["aggregate"] = _safe_dict(item.get("aggregate"))
        # Defensive normalization for partially written/legacy aggregate caches.
        item["aggregate"].setdefault("coverage", {})
        item["aggregate"].setdefault("claims", {})
        item["aggregate"].setdefault("fact_counts", {})
        item["aggregate"].setdefault("critical_omissions", [])
        item["aggregate"].setdefault("major_omissions", [])
        for imp in ("critical", "major", "moderate", "minor"):
            item["aggregate"]["coverage"][imp] = item["aggregate"]["coverage"].get(imp, 0) or 0

    report = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "book_title": book_title,
        "epub": str(epub_path),
        "reference": str(reference_path),
        "results_dir": str(results_dir),
        "evaluator": {
            "provider": args.evaluator,
            "model": evaluator_model,
            "display_name": evaluator_info.get("display_name", evaluator_model),
            "selected_variant": evaluator_info.get("selected_variant"),
            "quantization": _safe_dict(evaluator_info.get("quantization")),
            "context": evaluator_context,
            "temperature": args.temperature,
            "reasoning": args.reasoning,
            "reference_fact_extraction_seconds": reference_elapsed,
            "reference_prompt_tokens": ref_prompt_tokens,
            "reference_completion_tokens": ref_completion_tokens,
        },
        "models": model_reports,
    }
    save_json(eval_root / "evaluation.json", report)
    markdown_report(eval_root / "report.md", report)
    html_report(eval_root / "report.html", report)

    print("\nDone.")
    print(f"Report: {eval_root / 'report.html'}")
    print(f"JSON:   {eval_root / 'evaluation.json'}")
    print(f"MD:     {eval_root / 'report.md'}")


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate EPUB summaries against a NotebookLM reference using a local or cloud LLM evaluator."
    )
    parser.add_argument("epub", help="Original EPUB file")
    parser.add_argument("reference", help="NotebookLM-generated reference summary file")
    parser.add_argument("results", help="Results directory produced by epub_llm_benchmark.py")
    parser.add_argument("--evaluator", choices=["local", "claude-code", "openai", "anthropic"], default="local", help="Evaluator backend (default: local)")
    parser.add_argument("--evaluator-model", help="Evaluator model ID; for claude-code defaults to 'opus'")
    parser.add_argument("--base-url", default="http://localhost:1234", help="LM Studio base URL for --evaluator local")
    parser.add_argument("--temperature", type=float, default=0.0, help="Evaluator temperature (default: 0.0)")
    parser.add_argument("--reasoning", choices=["off", "low", "medium", "high", "xhigh", "on"], default="off")
    parser.add_argument("--max-tokens-facts", type=int, default=4096, help="Max tokens per chapter reference-fact extraction")
    parser.add_argument("--max-tokens-eval", type=int, default=4096, help="Max tokens per chapter evaluation")
    parser.add_argument("--max-tokens-book-eval", type=int, default=4096, help="Max tokens for complete-book evaluation")
    parser.add_argument("--min-output-tokens", type=int, default=2048)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument(
        "--resume",
        metavar="PATH",
        help="Resume exactly the evaluation run at PATH. Already completed phases/chapters are reused.",
    )
    args = parser.parse_args()
    try:
        run(args)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
