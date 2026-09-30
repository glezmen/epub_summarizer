#!/usr/bin/env python3
"""
Evaluate EPUB summaries against a NotebookLM reference using a local LLM in LM Studio.

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

No book text is sent anywhere except the LM Studio server configured with --base-url.
"""

from __future__ import annotations

import argparse
import html
import json
import re
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


def read_epub(epub_path: Path) -> tuple[str, list[Chapter]]:
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
        return book_title, chapters


# ---------------------------------------------------------------------------
# LM Studio
# ---------------------------------------------------------------------------

def api_request(url: str, payload: dict, timeout: int = 900) -> dict:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LM Studio HTTP {e.code}: {body}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Could not connect to LM Studio at {url}.") from e


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


def chat(
    base_url: str,
    model: str,
    system: str,
    user: str,
    max_tokens: int,
    temperature: float,
    reasoning: str,
    timeout: int,
) -> tuple[str, dict, float]:
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
    response = api_request(base_url.rstrip("/") + "/v1/chat/completions", payload, timeout)
    elapsed = time.perf_counter() - started
    try:
        choice = response["choices"][0]
        message = choice["message"]
        content = (message.get("content") or "").strip()
        finish_reason = choice.get("finish_reason")
    except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError(f"Unexpected LM Studio response: {response}") from e
    usage = response.get("usage") or {}
    if not content:
        details = usage.get("completion_tokens_details") or {}
        raise RuntimeError(
            f"Evaluator returned empty content; finish_reason={finish_reason}, "
            f"completion_tokens={usage.get('completion_tokens', 0)}, "
            f"reasoning_tokens={details.get('reasoning_tokens', 0)}"
        )
    return content, usage, elapsed


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
    r"^\s{0,3}(?:#{1,6}\s*)?(?:chapter\s+(\d+)\b.*|(\d+)\.\s+.+)$",
    re.I | re.M,
)
EPILOGUE_HEADING_RE = re.compile(r"^\s{0,3}(?:#{1,6}\s*)?epilogue\b.*$", re.I | re.M)


def parse_reference_sections(text: str) -> dict[str, str]:
    """Parse Markdown-like NotebookLM chapter headings into sections."""
    matches = []
    for m in CHAPTER_HEADING_RE.finditer(text):
        key = m.group(1) or m.group(2)
        if key:
            matches.append((m.start(), m.end(), key))
    for m in EPILOGUE_HEADING_RE.finditer(text):
        matches.append((m.start(), m.end(), "EPILOGUE"))
    matches.sort()

    sections: dict[str, str] = {}
    for i, (start, end, key) in enumerate(matches):
        next_start = matches[i + 1][0] if i + 1 < len(matches) else len(text)
        section = text[end:next_start].strip()
        sections[key] = section
    return sections


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

BOOK: {book_title}
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
        f"- Variant: `{report['evaluator'].get('selected_variant', '?')}`",
        f"- Quantization: `{report['evaluator'].get('quantization', {}).get('name', '?')}`",
        f"- Context: `{report['evaluator']['context']:,}` tokens",
        f"- Temperature: `{report['evaluator']['temperature']}`",
        f"- Reasoning: `{report['evaluator']['reasoning']}`",
        "",
        "## Model comparison",
        "",
        "| Model / run | Runtime | Weighted coverage | Critical | Major | Unsupported claims | Critical omissions | Major omissions |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in report["models"]:
        a = item["aggregate"]
        lines.append(
            f"| `{item['label']}` | {item['runtime_minutes']:.1f} min | "
            f"{pct(a['weighted_coverage'])} | {pct(a['coverage']['critical'])} | "
            f"{pct(a['coverage']['major'])} | {a['claims']['unsupported']} | "
            f"{len(a['critical_omissions'])} | {len(a['major_omissions'])} |"
        )
    lines += ["", "## Detailed results", ""]

    for item in report["models"]:
        a = item["aggregate"]
        lines += [
            f"### {item['label']}",
            "",
            f"- Model: `{item['model_info'].get('display_name', item['model'])}`",
            f"- Model key: `{item['model']}`",
            f"- Variant: `{item['model_info'].get('selected_variant', '?')}`",
            f"- Quantization: `{(item['model_info'].get('quantization') or {}).get('name', '?')}`",
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
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def html_report(path: Path, report: dict) -> None:
    rows = []
    for item in report["models"]:
        a = item["aggregate"]
        rows.append(
            "<tr>"
            f"<td>{html.escape(item['label'])}</td>"
            f"<td>{item['runtime_minutes']:.1f} min</td>"
            f"<td>{pct(a['weighted_coverage'])}</td>"
            f"<td>{pct(a['coverage']['critical'])}</td>"
            f"<td>{pct(a['coverage']['major'])}</td>"
            f"<td>{a['claims']['unsupported']}</td>"
            f"<td>{len(a['critical_omissions'])}</td>"
            f"<td>{len(a['major_omissions'])}</td>"
            "</tr>"
        )
    detail = []
    for item in report["models"]:
        a = item["aggregate"]
        detail.append(f"<h2>{html.escape(item['label'])}</h2>")
        detail.append("<h3>Model</h3><ul>")
        detail.append(f"<li>Name: {html.escape(item['model_info'].get('display_name', item['model']))}</li>")
        detail.append(f"<li>Key: {html.escape(item['model'])}</li>")
        detail.append(f"<li>Variant: {html.escape(item['model_info'].get('selected_variant', '?'))}</li>")
        detail.append(f"<li>Quantization: {html.escape((item['model_info'].get('quantization') or {}).get('name', '?'))}</li>")
        detail.append(f"<li>Runtime: {item['runtime_minutes']:.1f} min</li>")
        detail.append(f"<li>Prompt tokens: {item['prompt_tokens']:,}</li>")
        detail.append(f"<li>Completion tokens: {item['completion_tokens']:,}</li>")
        detail.append("</ul>")
        detail.append("<table><tr><th>Importance</th><th>Coverage</th><th>Total</th><th>Supported</th><th>Partial</th><th>Missing</th><th>Contradicted</th></tr>")
        for imp in ("critical", "major", "moderate", "minor"):
            c = a["fact_counts"][imp]
            detail.append(f"<tr><td>{imp}</td><td>{pct(a['coverage'][imp])}</td><td>{sum(c.values())}</td><td>{c['supported']}</td><td>{c['partial']}</td><td>{c['missing']}</td><td>{c['contradicted']}</td></tr>")
        detail.append("</table>")
        detail.append("<h3>Critical omissions</h3><ul>")
        detail.extend(f"<li>Chapter {html.escape(str(x['chapter']))}: {html.escape(x['note'])}</li>" for x in a["critical_omissions"])
        if not a["critical_omissions"]: detail.append("<li>None detected.</li>")
        detail.append("</ul><h3>Major omissions</h3><ul>")
        detail.extend(f"<li>Chapter {html.escape(str(x['chapter']))}: {html.escape(x['note'])}</li>" for x in a["major_omissions"])
        if not a["major_omissions"]: detail.append("<li>None detected.</li>")
        detail.append("</ul><h3>Unsupported claims</h3><ul>")
        detail.extend(f"<li>Chapter {html.escape(str(x['chapter']))}: {html.escape(x.get('claim', ''))} — {html.escape(x.get('note', ''))}</li>" for x in a["unsupported_claims"][:50])
        if not a["unsupported_claims"]: detail.append("<li>None detected.</li>")
        detail.append("</ul>")

    body = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>EPUB LLM Summary Evaluation</title>
<style>body{{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;max-width:1200px;margin:40px auto;padding:0 20px;line-height:1.45}}table{{border-collapse:collapse;width:100%;margin:15px 0 30px}}th,td{{border:1px solid #ccc;padding:7px;text-align:left}}th{{background:#f3f3f3}}code{{background:#f3f3f3;padding:2px 4px}}li{{margin:4px 0}}</style></head>
<body><h1>EPUB LLM Summary Evaluation</h1>
<p>Generated: {html.escape(report['generated_at'])}</p><p>Book: <b>{html.escape(report['book_title'])}</b></p>
<h2>Evaluator</h2><ul><li>{html.escape(report['evaluator']['display_name'])}</li><li>{html.escape(report['evaluator']['model'])}</li><li>Variant: {html.escape(report['evaluator'].get('selected_variant','?'))}</li><li>Context: {report['evaluator']['context']:,}</li><li>Reasoning: {html.escape(report['evaluator']['reasoning'])}</li></ul>
<h2>Model comparison</h2><table><tr><th>Model / run</th><th>Runtime</th><th>Weighted coverage</th><th>Critical</th><th>Major</th><th>Unsupported claims</th><th>Critical omissions</th><th>Major omissions</th></tr>{''.join(rows)}</table>
{''.join(detail)}
</body></html>"""
    path.write_text(body, encoding="utf-8")


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def run(args):
    epub_path = Path(args.epub).expanduser().resolve()
    reference_path = Path(args.reference).expanduser().resolve()
    results_dir = Path(args.results).expanduser().resolve()

    book_title, chapters = read_epub(epub_path)
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

    evaluator_model, evaluator_info, evaluator_context = get_loaded_model(args.base_url)
    eval_root = results_dir / "evaluation" / datetime.now().astimezone().strftime("run-%Y%m%d-%H%M%S")
    eval_root.mkdir(parents=True, exist_ok=False)

    print("=" * 70)
    print(" EPUB LLM SUMMARY EVALUATION")
    print("=" * 70)
    print(f"Book:       {book_title}")
    print(f"Chapters:   {len(chapters)}")
    print(f"Reference:  {reference_path}")
    print(f"Runs:       {len(runs)}")
    print(f"Evaluator:  {evaluator_info.get('display_name', evaluator_model)}")
    print(f"Eval model: {evaluator_model}")
    print(f"Context:    {evaluator_context:,} tokens")
    print(f"Output:     {eval_root}")
    print()

    save_json(eval_root / "evaluator_info.json", {
        "model": evaluator_model,
        "display_name": evaluator_info.get("display_name", evaluator_model),
        "selected_variant": evaluator_info.get("selected_variant"),
        "quantization": evaluator_info.get("quantization"),
        "context": evaluator_context,
        "temperature": args.temperature,
        "reasoning": args.reasoning,
        "base_url": args.base_url,
    })

    all_reference_facts: dict[str, list[dict]] = {}
    reference_elapsed = 0.0
    ref_prompt_tokens = ref_completion_tokens = 0

    # Phase 1: reference fact extraction.
    print("Phase 1/3: building importance-tagged reference facts...")
    ref_dir = eval_root / "reference"
    ref_dir.mkdir()
    for pos, chapter in enumerate(chapters, 1):
        reference_section = reference_sections.get(chapter.number, "")
        if not reference_section:
            print(f"  [{pos:02d}/{len(chapters)}] {chapter.title}: no matching NotebookLM section")
            all_reference_facts[chapter.number] = []
            continue
        prompt = FACT_PROMPT.format(
            book_title=book_title,
            chapter_title=chapter.title,
            reference=reference_section,
            source=chapter.text,
        )
        budget = min(args.max_tokens_facts, max(args.min_output_tokens, dynamic_budget(prompt, args.max_tokens_facts, args.min_output_tokens)))
        raw, usage, elapsed = chat(args.base_url, evaluator_model, SYSTEM, prompt, budget, args.temperature, args.reasoning, args.timeout)
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
    save_json(eval_root / "reference_facts.json", {"book_title": book_title, "facts": flat_facts})

    # Phase 2: evaluate each run chapter by chapter.
    model_reports = []
    print("\nPhase 2/3: evaluating local model summaries...")
    for run_index, run_dir in enumerate(runs, 1):
        book_info = read_json(run_dir / "book_info.json")
        run_stats = read_json(run_dir / "run_stats.json") if (run_dir / "run_stats.json").exists() else {}
        model_key = book_info.get("model", run_dir.parent.name)
        label = f"{run_dir.parent.name}/{run_dir.name}"
        model_out = eval_root / safe_name(run_dir.parent.name) / run_dir.name
        model_out.mkdir(parents=True)
        chapter_summaries = read_model_chapter_summaries(run_dir, chapters)
        chapter_results = []
        total_eval_elapsed = 0.0
        total_eval_prompt = total_eval_completion = 0

        print(f"\n  [{run_index}/{len(runs)}] {label}")
        for pos, chapter in enumerate(chapters, 1):
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
            prompt = EVAL_PROMPT.format(
                book_title=book_title,
                chapter_title=chapter.title,
                facts=facts_text,
                summary=summary,
                source=chapter.text,
            )
            budget = min(args.max_tokens_eval, max(args.min_output_tokens, dynamic_budget(prompt, args.max_tokens_eval, args.min_output_tokens)))
            raw, usage, elapsed = chat(args.base_url, evaluator_model, SYSTEM, prompt, budget, args.temperature, args.reasoning, args.timeout)
            parsed = extract_json(raw)
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
            "model_info": book_info.get("model_info", {}),
            "runtime_seconds": float(run_stats.get("total_elapsed_seconds", 0) or 0),
            "runtime_minutes": float(run_stats.get("total_elapsed_seconds", 0) or 0) / 60,
            "prompt_tokens": int(run_stats.get("total_prompt_tokens", 0) or 0),
            "completion_tokens": int(run_stats.get("total_completion_tokens", 0) or 0),
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
        summary = read_book_summary(run_dir)
        if not summary:
            continue
        prompt = BOOK_EVAL_PROMPT.format(
            book_title=book_title,
            facts=facts_text,
            summary=summary,
        )
        budget = min(args.max_tokens_book_eval, max(args.min_output_tokens, dynamic_budget(prompt, args.max_tokens_book_eval, args.min_output_tokens)))
        raw, usage, elapsed = chat(args.base_url, evaluator_model, SYSTEM, prompt, budget, args.temperature, args.reasoning, args.timeout)
        parsed = extract_json(raw)
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
        match = next((x for x in book_level if x["run_dir"] == item["run_dir"]), None)
        item["book_evaluation"] = match["result"] if match else {}

    report = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "book_title": book_title,
        "epub": str(epub_path),
        "reference": str(reference_path),
        "results_dir": str(results_dir),
        "evaluator": {
            "model": evaluator_model,
            "display_name": evaluator_info.get("display_name", evaluator_model),
            "selected_variant": evaluator_info.get("selected_variant"),
            "quantization": evaluator_info.get("quantization"),
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
        description="Evaluate local EPUB summaries against a NotebookLM reference using a local LM Studio evaluator."
    )
    parser.add_argument("epub", help="Original EPUB file")
    parser.add_argument("reference", help="NotebookLM-generated reference summary file")
    parser.add_argument("results", help="Results directory produced by epub_llm_benchmark.py")
    parser.add_argument("--base-url", default="http://localhost:1234", help="LM Studio base URL")
    parser.add_argument("--temperature", type=float, default=0.0, help="Evaluator temperature (default: 0.0)")
    parser.add_argument("--reasoning", choices=["off", "low", "medium", "high", "xhigh", "on"], default="off")
    parser.add_argument("--max-tokens-facts", type=int, default=4096, help="Max tokens per chapter reference-fact extraction")
    parser.add_argument("--max-tokens-eval", type=int, default=4096, help="Max tokens per chapter evaluation")
    parser.add_argument("--max-tokens-book-eval", type=int, default=4096, help="Max tokens for complete-book evaluation")
    parser.add_argument("--min-output-tokens", type=int, default=2048)
    parser.add_argument("--timeout", type=int, default=900)
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
