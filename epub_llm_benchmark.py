#!/usr/bin/env python3
"""
EPUB -> chapter summaries benchmark for LM Studio.

Designed for reproducible local-LLM comparisons:
- extracts the EPUB reading order from content.opf
- detects numbered chapters and EPILOGUE
- ignores front/back matter
- splits long chapters into deterministic chunks
- sends the same prompts to the LM Studio OpenAI-compatible API
- saves per-chunk, per-chapter and full-book summaries
- records timing/token usage when the server provides it

Tested conceptually with:
  The Butcher's Masquerade: Dungeon Crawler Carl Book 5

LM Studio default API:
  http://localhost:1234

No book text is sent anywhere except the LM Studio server you configure.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import html
import re
import sys
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass, asdict
from html.parser import HTMLParser
from pathlib import Path
from typing import Optional
import xml.etree.ElementTree as ET


OPF_NS = "http://www.idpf.org/2007/opf"
DC_NS = "http://purl.org/dc/elements/1.1/"


# -----------------------------
# EPUB parsing
# -----------------------------

class TextExtractor(HTMLParser):
    BLOCK_TAGS = {
        "p", "div", "section", "article", "blockquote",
        "h1", "h2", "h3", "h4", "h5", "h6",
        "li", "br", "hr", "tr"
    }
    SKIP_TAGS = {"script", "style", "svg", "head"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skip_depth = 0
        self.headings: list[str] = []
        self.current_heading: Optional[str] = None
        self.heading_depth = 0

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
            self.heading_depth = 1

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


def is_toc_document(href: str, headings: list[str], text: str) -> bool:
    """Return True for common EPUB table-of-contents documents.

    TOC pages can contain numbered chapter-looking lines, so they must be
    filtered before chapter detection rather than treated as chapters.
    """
    path_name = Path(href).name.lower()
    stem = Path(href).stem.lower()
    if stem in {"toc", "contents", "content", "tableofcontents", "table-of-contents", "nav"}:
        return True
    if any(token in path_name for token in ("toc", "contents", "tableofcontents", "table-of-contents")):
        return True

    heading_text = " ".join(headings[:5]).lower()
    first_text = " ".join(clean_text(x) for x in text.splitlines()[:30]).lower()
    toc_markers = (
        "table of contents", "contents", "content",
        "tartalomjegyzék", "tartalom",
    )
    if any(marker in heading_text for marker in toc_markers):
        return True
    if any(marker in first_text for marker in toc_markers):
        # Avoid classifying an ordinary chapter merely because the word
        # "content" occurs later; only inspect the beginning of the document.
        return True

    # A TOC typically contains many short numbered entries and very little
    # prose.  This catches TOCs whose file is named generically (e.g. 001.html).
    lines = [clean_text(x) for x in text.splitlines() if clean_text(x)]
    numbered = sum(bool(re.fullmatch(r"(?:Chapter\s+)?\d+(?:\.\s*.*)?", x, re.I)) for x in lines[:80])
    short_lines = sum(len(x) <= 120 for x in lines[:80])
    if numbered >= 5 and short_lines >= max(5, int(0.7 * min(len(lines), 80))):
        return True

    return False


def detect_chapter_marker(headings: list[str], text: str) -> tuple[Optional[str], Optional[str]]:
    """Detect chapter number/title across common EPUB layouts.

    Supported examples:
      - <h1>1</h1>
      - <h1>EPILOGUE</h1>
      - plain text: "1. YIS VENTER RAJONGÁSA"
      - plain text: "Chapter 1" / "Chapter 1: Title"

    Some EPUBs store the chapter heading in a <p> rather than a heading tag,
    so the extracted body text is also scanned line-by-line.
    """
    candidates = list(headings)

    # Prefer explicit HTML headings when available.
    for candidate in candidates:
        value = clean_text(candidate)
        if not value:
            continue
        m = re.fullmatch(r"(?:chapter\s+)?(\d+)(?:\s*[:.-]\s*(.*))?", value, re.IGNORECASE)
        if m:
            number = m.group(1)
            title_suffix = clean_text(m.group(2) or "")
            title = f"Chapter {number}" if not title_suffix else title_suffix
            return number, title
        if value.upper() in {"EPILOGUE", "EPILÓGUS"}:
            return "EPILOGUE", "Epilogue"

    # Fallback for EPUBs where the chapter heading is a normal paragraph.
    # Scan only the beginning of the document to avoid mistaking numbered
    # lists/references later in the chapter for the chapter title.
    lines = [clean_text(line) for line in text.splitlines()]
    lines = [line for line in lines if line]
    for line in lines[:80]:
        m = re.fullmatch(r"(\d+)\.\s+(.+)", line)
        if m:
            return m.group(1), clean_text(m.group(2))

        m = re.fullmatch(r"Chapter\s+(\d+)(?:\s*[:.-]\s*(.*))?", line, re.IGNORECASE)
        if m:
            number = m.group(1)
            suffix = clean_text(m.group(2) or "")
            return number, (f"Chapter {number}" if not suffix else suffix)

        if line.upper() in {"EPILOGUE", "EPILÓGUS"}:
            return "EPILOGUE", "Epilogue"

    return None, None


def read_epub(epub_path: Path) -> tuple[str, str, list[Chapter]]:
    with zipfile.ZipFile(epub_path, "r") as z:
        opf_path = find_opf_path(z)
        opf_data = z.read(opf_path)
        root = ET.fromstring(opf_data)

        manifest = {}
        for item in root.findall(f".//{{{OPF_NS}}}manifest/{{{OPF_NS}}}item"):
            manifest[item.attrib["id"]] = {
                "href": item.attrib.get("href", ""),
                "properties": item.attrib.get("properties", ""),
                "media_type": item.attrib.get("media-type", ""),
            }

        spine = []
        for itemref in root.findall(f".//{{{OPF_NS}}}spine/{{{OPF_NS}}}itemref"):
            spine.append(itemref.attrib["idref"])

        book_title = ""
        book_language = ""
        title_el = root.find(f".//{{{DC_NS}}}title")
        if title_el is not None and title_el.text:
            book_title = title_el.text.strip()
        language_el = root.find(f".//{{{DC_NS}}}language")
        if language_el is not None and language_el.text:
            book_language = language_el.text.strip()

        chapters: list[Chapter] = []

        for idref in spine:
            item = manifest.get(idref)
            if not item:
                continue
            href = item["href"]
            properties = item["properties"].split()

            # EPUB3 navigation documents are metadata/navigation, not plot
            # chapters, even when they appear in the spine.
            if "nav" in properties or "toc" in properties:
                continue

            # Resolve relative to the OPF directory.
            opf_dir = Path(opf_path).parent
            full_path = (opf_dir / href).as_posix()

            if not full_path.lower().endswith((".html", ".xhtml", ".htm")):
                continue

            try:
                html = z.read(full_path).decode("utf-8", errors="replace")
            except KeyError:
                continue

            parser = TextExtractor()
            parser.feed(html)
            text = clean_text("".join(parser.parts))
            headings = parser.headings

            # Some EPUBs put the table of contents in the spine and use a
            # generic filename.  Filter it before chapter-number detection.
            if is_toc_document(full_path, headings, text):
                continue

            number, title = detect_chapter_marker(headings, text)
            if number and title:
                chapters.append(Chapter(number, title, full_path, text))

        if not chapters:
            raise RuntimeError(
                "No numbered chapters or EPILOGUE were detected. "
                "Inspect the EPUB structure or add a custom chapter detector."
            )

        return book_title, book_language, chapters


def find_opf_path(z: zipfile.ZipFile) -> str:
    container = ET.fromstring(z.read("META-INF/container.xml"))
    rootfile = container.find(
        ".//{urn:oasis:names:tc:opendocument:xmlns:container}rootfile"
    )
    if rootfile is None:
        raise RuntimeError("META-INF/container.xml does not contain a rootfile.")
    return rootfile.attrib["full-path"]


# -----------------------------
# Adaptive chapter handling
# -----------------------------

def split_text(text: str, max_chars: int, overlap: int) -> list[str]:
    """Fallback chunking used only when a chapter does not fit the context."""
    if len(text) <= max_chars:
        return [text]
    paragraphs = re.split(r"\n\s*\n", text)
    chunks, current = [], ""
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
        if len(paragraph) > max_chars:
            pos = 0
            while pos < len(paragraph):
                end_pos = min(pos + max_chars, len(paragraph))
                chunks.append(paragraph[pos:end_pos])
                if end_pos >= len(paragraph):
                    break
                pos = max(0, end_pos - overlap)
            current = ""
        else:
            current = paragraph
    if current:
        chunks.append(current)
    if overlap > 0 and len(chunks) > 1:
        return [chunks[0]] + [(chunks[i-1][-overlap:] + "\n\n" + chunks[i]).strip() for i in range(1,len(chunks))]
    return chunks

def estimate_tokens(text: str) -> int:
    # Conservative estimate used only for deciding whether a chapter fits.
    return max(1, int(len(text) / 3.5))

def choose_output_tokens(text: str, cap: int, minimum: int = 2048, ratio: float = 0.5) -> int:
    """Choose an output budget from the approximate input size.

    The result is rounded up to a 512-token boundary. This is deliberately
    capped: a long source text does not imply that its summary needs to be
    equally long.
    """
    input_tokens = estimate_tokens(text)
    recommended = max(minimum, int(input_tokens * ratio))
    recommended = ((recommended + 511) // 512) * 512
    return max(minimum, min(cap, recommended))

def fit_output_budget_to_context(
    input_tokens: int,
    requested_output_tokens: int,
    context_tokens: int,
    safety_margin: int = 256,
) -> int:
    """Reduce an output budget so input + output stays within model context.

    The dynamic budget minimum is intentionally allowed to shrink here: for a
    near-context-limit prompt, preserving a valid request is more important
    than enforcing the normal minimum output budget.
    """
    available = context_tokens - input_tokens - safety_margin
    if available < 256:
        raise ValueError(
            f"Prompt is too large for the model context: input={input_tokens:,}, "
            f"context={context_tokens:,}, safety_margin={safety_margin:,}."
        )
    return min(requested_output_tokens, available)


def get_context_limit(model_info: dict, default: int = 32768) -> int:
    for key in ("context_length", "contextLength", "context"):
        value = model_info.get(key)
        if isinstance(value, (int, float)) and value > 0:
            return int(value)
    return default

def chapter_chunks(text: str, context_limit: int, reserved_output_tokens: int, prompt_overhead_tokens: int, overlap_chars: int, chunk_size_tokens: int | None = None) -> list[str]:
    """Split a chapter using an optional explicit input-token chunk size.

    When ``chunk_size_tokens`` is provided, it is the maximum approximate
    input size of each chunk. Otherwise the existing adaptive context-based
    behavior is preserved.
    """
    if chunk_size_tokens is not None:
        if chunk_size_tokens <= 0:
            raise ValueError("chunk_size_tokens must be greater than zero")
        max_chars = max(4000, int(chunk_size_tokens * 3.5))
        return split_text(text, max_chars=max_chars, overlap=overlap_chars)

    available = context_limit - reserved_output_tokens - prompt_overhead_tokens
    if estimate_tokens(text) <= available:
        return [text]
    max_chars = max(4000, int(available * 3.5))
    return split_text(text, max_chars=max_chars, overlap=overlap_chars)

# -----------------------------
# LM Studio API
# -----------------------------

def api_request(url: str, payload: dict, timeout: int = 600) -> dict:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LM Studio HTTP {e.code}: {body}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Could not connect to LM Studio at {url}. "
            f"Make sure the local server is running."
        ) from e


def get_loaded_model(base_url: str) -> tuple[str, dict, int, list[str]]:
    """Find the actually loaded LLM via LM Studio's native API."""
    url = base_url.rstrip("/") + "/api/v1/models"
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))
    except Exception as e:
        raise RuntimeError(f"Could not query {url}. Is LM Studio's local server running?") from e

    loaded = []
    for info in data.get("models", []):
        if info.get("type") != "llm":
            continue
        for instance in (info.get("loaded_instances") or []):
            loaded.append((info, instance))

    if not loaded:
        raise RuntimeError("LM Studio reports no loaded LLM. Load a model in LM Studio.")

    if len(loaded) > 1:
        lines = []
        for info, instance in loaded:
            cfg = instance.get("config") or {}
            lines.append(f"  - {info.get('key', '<unknown>')} (instance={instance.get('id', '<unknown>')}, context={cfg.get('context_length', '?')})")
        raise RuntimeError("LM Studio has multiple loaded LLM instances; refusing to guess.\n" + "\n".join(lines))

    info, instance = loaded[0]
    config = instance.get("config") or {}
    context_length = config.get("context_length")
    if not isinstance(context_length, int) or context_length <= 0:
        context_length = info.get("max_context_length")
    if not isinstance(context_length, int) or context_length <= 0:
        raise RuntimeError(f"No valid context_length found for loaded model {info.get('key')}")

    # Native LM Studio model metadata exposes which reasoning settings are
    # actually supported by the loaded model. Different model families can
    # have different restrictions (e.g. DeepSeek R1 may support only "on").
    reasoning_caps = info.get("capabilities", {}).get("reasoning", {})
    allowed_reasoning = reasoning_caps.get("allowed_options") or []
    if not isinstance(allowed_reasoning, list):
        allowed_reasoning = []
    allowed_reasoning = [str(x) for x in allowed_reasoning]

    return info["key"], info, context_length, allowed_reasoning


def resolve_reasoning(requested: str, allowed: list[str], model_info: dict) -> str:
    """Resolve the requested reasoning mode against LM Studio capabilities.

    The benchmark should not fail merely because a model requires reasoning.
    If the requested mode is unsupported, prefer the model's declared default;
    otherwise fall back to the first supported option. If metadata is missing,
    keep the requested value and let LM Studio validate it.
    """
    if not allowed:
        return requested
    if requested in allowed:
        return requested

    default = (
        model_info.get("capabilities", {})
        .get("reasoning", {})
        .get("default")
    )
    if default in allowed:
        return default
    if requested == "off" and "on" in allowed:
        return "on"
    return allowed[0]


def chat(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    timeout: int,
    reasoning: str,
    send_reasoning: bool = True,
    reasoning_max_tokens: Optional[int] = None,
    final_answer_tokens: int = 512,
) -> tuple[str, dict, float]:
    """Call LM Studio and return the final message plus authoritative stats.

    LM Studio's native v1 API exposes one combined `max_output_tokens` budget;
    it does not expose a separate reasoning-token ceiling. Therefore the
    benchmark can reserve a final-answer budget by capping the total request
    and, when a reasoning-only response is returned, retrying through the
    OpenAI-compatible endpoint. The latter is also useful for model families
    whose native reasoning parser returns reasoning but no final message.
    """
    # For reasoning models, split the configured total output budget into two
    # independent passes. The first pass gets the reasoning allowance; if it
    # produces no final message, the fallback gets the remaining final-answer
    # allowance. This keeps the combined benchmark budget bounded.
    total_budget = max(1, int(max_tokens))
    if reasoning_max_tokens is not None and reasoning != "off":
        requested_final = max(1, int(final_answer_tokens))
        # Never let the reasoning allowance consume the entire combined budget.
        # If the dynamic total is smaller than the requested final-answer
        # reserve, split the available budget roughly evenly.
        final_answer_budget = min(requested_final, max(1, total_budget // 2))
        reasoning_budget = min(
            max(1, int(reasoning_max_tokens)),
            max(1, total_budget - final_answer_budget),
        )
        first_budget = reasoning_budget
    else:
        reasoning_budget = 0
        final_answer_budget = total_budget
        first_budget = total_budget

    system_prompt = (
        "You are a precise literary text summarizer. "
        "You must only use information present in the supplied text. "
        "Do not invent events, characters, motivations, or facts."
    )

    def parse_native(response: dict) -> tuple[str, dict]:
        output = response.get("output") or []
        message_parts = [
            item.get("content", "")
            for item in output
            if isinstance(item, dict) and item.get("type") == "message"
        ]
        content = "\n".join(part for part in message_parts if part).strip()
        reasoning_parts = [
            item.get("content", "")
            for item in output
            if isinstance(item, dict) and item.get("type") == "reasoning"
        ]
        reasoning_content = "\n".join(part for part in reasoning_parts if part).strip()
        stats = response.get("stats") or {}
        input_tokens = int(stats.get("input_tokens", 0) or 0)
        completion_tokens = int(stats.get("total_output_tokens", 0) or 0)
        reasoning_tokens = int(stats.get("reasoning_output_tokens", 0) or 0)
        usage = {
            "prompt_tokens": input_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": input_tokens + completion_tokens,
            "completion_tokens_details": {
                "reasoning_tokens": reasoning_tokens,
            },
            "reasoning_tokens": reasoning_tokens,
            "lmstudio_stats": stats,
            "reasoning_content": reasoning_content,
        }
        return content, usage

    def native_call(budget: int) -> tuple[str, dict, float]:
        url = base_url.rstrip("/") + "/api/v1/chat"
        payload = {
            "model": model,
            "input": prompt,
            "system_prompt": system_prompt,
            "temperature": temperature,
            "max_output_tokens": budget,
            "stream": False,
        }
        if send_reasoning:
            payload["reasoning"] = reasoning

        started = time.perf_counter()
        response = api_request(url, payload, timeout)
        elapsed = time.perf_counter() - started
        content, usage = parse_native(response)
        return content, usage, elapsed

    # For reasoning-enabled models, prefer the OpenAI-compatible endpoint.
    # LM Studio's native endpoint can return a long reasoning block without a
    # final message for DeepSeek R1, while /v1/chat/completions returns both
    # reasoning and final content when given the full output budget.
    #
    # There is no separate reasoning-token ceiling exposed by this endpoint,
    # so max_tokens must remain the complete budget for the single inference.
    def openai_call(budget: int):
        url = base_url.rstrip("/") + "/v1/chat/completions"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": budget,
            "stream": False,
        }
        started = time.perf_counter()
        response = api_request(url, payload, timeout)
        elapsed = time.perf_counter() - started
        choices = response.get("choices") or []
        choice = choices[0] if choices else {}
        message = choice.get("message") or {}
        content = (message.get("content") or "").strip()
        reasoning_content = (
            message.get("reasoning_content")
            or message.get("reasoning")
            or ""
        )
        openai_usage = response.get("usage") or {}
        input_tokens = int(openai_usage.get("prompt_tokens", 0) or openai_usage.get("input_tokens", 0) or 0)
        completion_tokens = int(openai_usage.get("completion_tokens", 0) or openai_usage.get("output_tokens", 0) or 0)
        details = openai_usage.get("completion_tokens_details") or {}
        reasoning_tokens = int(details.get("reasoning_tokens", 0) or openai_usage.get("reasoning_tokens", 0) or 0)
        usage = {
            "prompt_tokens": input_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": int(openai_usage.get("total_tokens", 0) or (input_tokens + completion_tokens)),
            "completion_tokens_details": {"reasoning_tokens": reasoning_tokens},
            "reasoning_tokens": reasoning_tokens,
            "lmstudio_stats": {**openai_usage, "endpoint": "/v1/chat/completions", "finish_reason": choice.get("finish_reason")},
            "reasoning_content": reasoning_content,
        }
        return content, usage, elapsed

    if reasoning != "off":
        try:
            content, usage, elapsed = openai_call(total_budget)
            if content:
                return content, usage, elapsed
            openai_error = "OpenAI-compatible endpoint returned an empty final answer."
        except RuntimeError as e:
            openai_error = str(e)

        # Only if the compatible endpoint itself fails do we try the native
        # endpoint. This is a genuine fallback, not a second full inference.
        try:
            content, usage, elapsed = native_call(first_budget)
            if content:
                return content, usage, elapsed
            native_error = "Native endpoint returned an empty final answer."
        except RuntimeError as e:
            native_error = str(e)
        raise RuntimeError(
            "LM Studio returned no final answer from either endpoint. "
            f"OpenAI-compatible: {openai_error}; Native: {native_error}"
        )

    # Explicit non-reasoning runs retain the native endpoint first.
    try:
        content, usage, elapsed = native_call(total_budget)
        if content:
            return content, usage, elapsed
        native_error = "Native endpoint returned an empty final answer."
    except RuntimeError as e:
        native_error = str(e)

    try:
        content, usage, elapsed = openai_call(total_budget)
        if content:
            usage["fallback_from_native"] = True
            return content, usage, elapsed
        openai_error = "OpenAI-compatible endpoint returned an empty final answer."
    except RuntimeError as e:
        openai_error = str(e)
    raise RuntimeError(
        "LM Studio returned no final answer from either endpoint. "
        f"Native: {native_error}; OpenAI-compatible: {openai_error}"
    )


# -----------------------------
# Prompts
# -----------------------------

CHUNK_PROMPT = """We are running a reproducible benchmark of local LLMs on a novel.

Book: {book_title}
Chapter: {chapter_title}
Summary language: {summary_language}
Source language: {book_language}
This is chunk {chunk_index} of {chunk_count}.

Summarize ONLY the plot events contained in this text.

The summary must:
- focus on what actually happens;
- identify important characters involved;
- include important decisions, discoveries, conflicts and consequences;
- preserve the causal relationship between events where it is stated or clear;
- include details that may matter later in the story;
- avoid literary criticism and interpretation;
- avoid guessing or filling gaps;
- avoid information not present in this chunk.

Do not quote the novel.
Write the summary in {summary_language}. Do not translate it to another language unless that is the selected summary language.

TEXT:
{chunk}
"""

MERGE_PROMPT = """We are running a reproducible benchmark of local LLMs on a novel.

Book: {book_title}
Source language: {book_language}
Summary language: {summary_language}
Chapter: {chapter_title}

Below are summaries of consecutive chunks from the SAME chapter.

Create ONE coherent factual plot summary of the entire chapter.

Requirements:
- combine the events into chronological order;
- remove duplicated information caused by chunk overlap;
- retain important events, characters, decisions, discoveries, conflicts and consequences;
- retain information that can be important for later plot developments;
- do not invent anything;
- do not add information from outside these summaries;
- do not discuss the quality of the writing;
- do not quote the novel.
Write the summary in {summary_language}. Do not translate it to another language unless that is the selected summary language.

CHUNK SUMMARIES:
{summaries}
"""

BOOK_DIGEST_PROMPT = """We are preparing a compact intermediate digest of a novel for a later whole-book summary.

Book: {book_title}
Language: {summary_language}

Below are chapter summaries. Compress them into a concise, factually faithful digest.
Preserve important plot events, characters, relationships, causes/effects, discoveries, conflicts, and the ending information present in the summaries.
Do not invent information and do not add commentary about the summarization process.

CHAPTER SUMMARIES:
{summaries}
"""

BOOK_PROMPT = """We are running a reproducible benchmark of local LLMs on a novel.

Book: {book_title}
Source language: {book_language}
Summary language: {summary_language}

Below are factual summaries of every chapter, in chronological order.

Create a coherent, detailed summary of the COMPLETE PLOT.

Requirements:
- preserve chronological order;
- explain the major causal chains of the story;
- identify the main characters and their important actions;
- include major conflicts, discoveries, turning points and consequences;
- include important developments that set up later events;
- do not invent anything;
- use ONLY the chapter summaries supplied below;
- do not discuss the quality of the writing;
- do not quote the novel.
Write the summary in {summary_language}. Do not translate it to another language unless that is the selected summary language.

CHAPTER SUMMARIES:
{summaries}
"""


# -----------------------------
# Benchmark runner
# -----------------------------

LANGUAGE_NAMES = {
    "en": "English",
    "hu": "Hungarian",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "it": "Italian",
    "pt": "Portuguese",
    "nl": "Dutch",
    "pl": "Polish",
    "cs": "Czech",
    "sk": "Slovak",
    "ro": "Romanian",
    "ru": "Russian",
    "uk": "Ukrainian",
    "ja": "Japanese",
    "ko": "Korean",
    "zh": "Chinese",
    "tr": "Turkish",
    "sv": "Swedish",
    "no": "Norwegian",
    "da": "Danish",
    "fi": "Finnish",
}

def resolve_summary_language(requested: str, source_language: Optional[str]) -> str:
    value = (requested or "source").strip()
    if value.lower() == "source":
        code = (source_language or "").strip().lower()
        if code:
            base = code.split("-")[0].split("_")[0]
            return LANGUAGE_NAMES.get(base, base.upper())
        return "the same language as the source text"
    if value.lower() == "english":
        return "English"
    code = value.lower().replace("_", "-")
    base = code.split("-")[0]
    return LANGUAGE_NAMES.get(base, code)


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def safe_model_dir_name(model_key: str) -> str:
    """Convert an LM Studio model key into a filesystem-safe directory name."""
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", model_key).strip(".-")
    return name or "unknown-model"


def create_run_output_dir(base_output: Path, model_key: str) -> tuple[Path, str, str]:
    """Create results/<model>/<timestamp>/ and return its path plus identifiers."""
    model_dir_name = safe_model_dir_name(model_key)
    timestamp = datetime.now().astimezone().strftime("run-%Y%m%d-%H%M%S")
    run_dir = base_output / model_dir_name / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir, model_dir_name, timestamp



def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def usage_totals(usage: dict) -> tuple[int, int, int]:
    return (
        int(usage.get("prompt_tokens", 0) or 0),
        int(usage.get("completion_tokens", 0) or 0),
        int(usage.get("reasoning_tokens", 0) or 0),
    )


def cached_chunk_summary(
    chunk_path: Path,
    expected_chapter: str,
    expected_chunk: int,
    expected_chunk_count: int,
) -> tuple[Optional[str], dict, float]:
    """Load a completed chunk artifact if it matches the current deterministic plan."""
    if not chunk_path.exists():
        return None, {}, 0.0
    try:
        record = load_json(chunk_path)
        if (
            str(record.get("chapter")) != str(expected_chapter)
            or int(record.get("chunk", -1)) != expected_chunk
            or int(record.get("chunk_count", -1)) != expected_chunk_count
            or not str(record.get("summary", "")).strip()
        ):
            return None, {}, 0.0
        return (
            str(record["summary"]).strip(),
            record.get("usage") or {},
            float(record.get("elapsed_seconds", 0.0) or 0.0),
        )
    except (OSError, ValueError, TypeError, KeyError):
        return None, {}, 0.0


def cached_chapter_summary(
    chapter_path: Path,
    expected_chapter: str,
    expected_title: str,
    expected_source_characters: int,
    expected_chunk_count: int,
) -> tuple[Optional[str], dict, float]:
    """Load a completed chapter artifact if it matches the current EPUB parse."""
    if not chapter_path.exists():
        return None, {}, 0.0
    try:
        record = load_json(chapter_path)
        if (
            str(record.get("chapter")) != str(expected_chapter)
            or record.get("title") != expected_title
            or int(record.get("source_characters", -1)) != expected_source_characters
            or int(record.get("chunk_count", -1)) != expected_chunk_count
            or not str(record.get("summary", "")).strip()
        ):
            return None, {}, 0.0
        return (
            str(record["summary"]).strip(),
            record.get("merge_usage") or {},
            float(record.get("merge_elapsed_seconds", 0.0) or 0.0),
        )
    except (OSError, ValueError, TypeError, KeyError):
        return None, {}, 0.0


def validate_resume_metadata(
    output_dir: Path,
    epub_path: Path,
    book_title: str,
    model: str,
    effective_context: int,
) -> None:
    """Reject an obviously incompatible --resume target instead of mixing runs."""
    info_path = output_dir / "book_info.json"
    if not info_path.exists():
        raise RuntimeError(
            f"Cannot resume '{output_dir}': book_info.json is missing. "
            "Choose a benchmark run directory created by this script."
        )

    info = load_json(info_path)
    checks = [
        ("source_file", epub_path.name, info.get("source_file")),
        ("book_title", book_title, info.get("book_title")),
        ("model", model, info.get("model")),
    ]
    for field, expected, actual in checks:
        if actual and actual != expected:
            raise RuntimeError(
                f"Cannot resume '{output_dir}': {field} mismatch "
                f"(existing={actual!r}, current={expected!r})."
            )

    existing_context = info.get("context")
    if existing_context and int(existing_context) != int(effective_context):
        raise RuntimeError(
            f"Cannot resume '{output_dir}': context mismatch "
            f"(existing={existing_context}, current={effective_context}). "
            "Use the same --context value as the original run."
        )


def run(args):
    epub_path = Path(args.epub).expanduser().resolve()
    base_output_dir = Path(args.output).expanduser().resolve()

    book_title, book_language, chapters = read_epub(epub_path)
    summary_language = resolve_summary_language(args.summary_language, book_language)

    model, model_info, detected_context, allowed_reasoning = get_loaded_model(args.base_url)
    effective_reasoning = resolve_reasoning(args.reasoning, allowed_reasoning, model_info)

    # If reasoning capability metadata is unavailable, an explicit
    # reasoning="off" is unsafe for some models: LM Studio may reject the
    # field because the model does not expose reasoning configuration.
    # If metadata explicitly lists supported options, keep sending the field
    # so a requested "off" remains authoritative.
    send_reasoning = bool(allowed_reasoning) or args.reasoning != "off"

    effective_context = args.context if args.context is not None else detected_context

    if args.resume:
        output_dir = Path(args.resume).expanduser().resolve()
        if not output_dir.is_dir():
            raise RuntimeError(f"Resume directory does not exist: {output_dir}")

        # The run directory itself determines the model/run identity.
        model_dir_name = output_dir.parent.name
        run_id = output_dir.name
        validate_resume_metadata(
            output_dir,
            epub_path,
            book_title,
            model,
            effective_context,
        )
        print(f"Resuming benchmark: {output_dir}")
    else:
        base_output_dir.mkdir(parents=True, exist_ok=True)
        output_dir, model_dir_name, run_id = create_run_output_dir(
            base_output_dir, model
        )

    print("=" * 58)
    print(" EPUB LLM BENCHMARK")
    print("=" * 58)
    print()
    print(f"Book:       {book_title}")
    print(f"Language:   {book_language or 'metadata unavailable'}")
    print(f"Summary:    {summary_language}")
    print(f"Chapters:   {len(chapters)}")
    effective_context = args.context if args.context is not None else detected_context
    print(f"Model:      {model_info.get('display_name', model)}")
    print(f"Model key:  {model}")
    print(f"Variant:    {model_info.get('selected_variant', '?')}")
    print(f"Quant:      {(model_info.get('quantization') or {}).get('name', '?')}")
    print(f"API:        {args.base_url}")
    print(f"Temperature:{args.temperature}")
    print(f"Reasoning:  {effective_reasoning} (requested: {args.reasoning})")
    print(f"Reasoning supported: {', '.join(allowed_reasoning) if allowed_reasoning else 'metadata unavailable'}")
    print(f"Context:    {effective_context:,} tokens")
    print(f"Chunking:   explicit {args.chunk_size:,} input tokens/chunk" if args.chunk_size else "Chunking:   whole chapter when it fits; adaptive fallback otherwise")
    print(f"Output:     {output_dir}")
    print(f"Run ID:     {run_id}")
    print()

    # On a new run this creates the immutable run metadata. On resume, keep
    # the original metadata and only add/update the current execution fields.
    book_info_path = output_dir / "book_info.json"
    if not book_info_path.exists():
        save_json(
            book_info_path,
            {
                "book_title": book_title,
                "source_language": book_language,
                "summary_language": summary_language,
                "source_file": epub_path.name,
                "model": model,
                "model_dir": model_dir_name,
                "run_id": run_id,
                "model_info": model_info,
                "base_url": args.base_url,
                "temperature": args.temperature,
                "reasoning_requested": args.reasoning,
                "reasoning_effective": effective_reasoning,
                "reasoning_supported": allowed_reasoning,
                "min_output_tokens": args.min_output_tokens,
                "output_ratio": args.output_ratio,
                "reasoning_output_ratio": args.reasoning_output_ratio,
                "reasoning_min_output_tokens": args.reasoning_min_output_tokens,
                "reasoning_max_tokens": args.reasoning_max_tokens,
                "final_answer_tokens": args.final_answer_tokens,
                "max_tokens_chunk": args.max_tokens_chunk,
                "max_tokens_merge": args.max_tokens_merge,
                "max_tokens_book": args.max_tokens_book,
                "context": effective_context,
                "loaded_instance": (model_info.get("loaded_instances") or [{}])[0],
                "chunking": (
                    f"explicit {args.chunk_size} input tokens/chunk"
                    if args.chunk_size
                    else "whole chapter when it fits; adaptive fallback otherwise"
                ),
                "chunk_size_tokens": args.chunk_size,
                "chunk_overlap": args.chunk_overlap,
                "chapters": [
                    {
                        "number": c.number,
                        "title": c.title,
                        "href": c.href,
                        "characters": len(c.text),
                    }
                    for c in chapters
                ],
            },
        )

    all_chapter_summaries = []
    total_elapsed = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_reasoning_tokens = 0
    completed_chapters_elapsed = 0.0

    for chapter_pos, chapter in enumerate(chapters, 1):
        chapter_dir = (
            output_dir / f"{int(chapter.number):03d}"
            if chapter.number.isdigit()
            else output_dir / "epilogue"
        )
        chapter_dir.mkdir(parents=True, exist_ok=True)

        context_limit = effective_context
        chapter_output_budget = choose_output_tokens(
            chapter.text,
            cap=args.max_tokens_chunk,
            minimum=(
                args.reasoning_min_output_tokens
                if effective_reasoning != "off"
                else args.min_output_tokens
            ),
            ratio=(
                args.reasoning_output_ratio
                if effective_reasoning != "off"
                else args.output_ratio
            ),
        )
        chunks = chapter_chunks(
            chapter.text,
            context_limit=context_limit,
            reserved_output_tokens=chapter_output_budget,
            prompt_overhead_tokens=1000,
            overlap_chars=args.chunk_overlap,
            chunk_size_tokens=args.chunk_size,
        )

        chapter_summary_path = chapter_dir / "chapter_summary.json"
        cached_summary, cached_merge_usage, cached_merge_elapsed = cached_chapter_summary(
            chapter_summary_path,
            chapter.number,
            chapter.title,
            len(chapter.text),
            len(chunks),
        )

        if cached_summary is not None:
            print(
                f"[{chapter_pos:02d}/{len(chapters)}] "
                f"{chapter.title}: [CACHED] "
                f"{len(chunks)} chunk(s)"
            )
            all_chapter_summaries.append(
                f"--- {chapter.title} ---\n{cached_summary}"
            )

            # Include cached inference statistics in the resumed run totals.
            total_prompt, total_completion, total_reasoning = usage_totals(
                cached_merge_usage
            )
            total_prompt_tokens += total_prompt
            total_completion_tokens += total_completion
            total_reasoning_tokens += total_reasoning
            total_elapsed += cached_merge_elapsed

            # For a multi-chunk chapter, chunk statistics are stored separately.
            for i in range(1, len(chunks) + 1):
                _, chunk_usage, chunk_elapsed = cached_chunk_summary(
                    chapter_dir / f"chunk_{i:02d}.json",
                    chapter.number,
                    i,
                    len(chunks),
                )
                p, c, r = usage_totals(chunk_usage)
                total_prompt_tokens += p
                total_completion_tokens += c
                total_reasoning_tokens += r
                total_elapsed += chunk_elapsed
            continue

        if completed_chapters_elapsed > 0 and chapter_pos > 1:
            avg_chapter_seconds = completed_chapters_elapsed / (chapter_pos - 1)
            remaining_chapters = len(chapters) - chapter_pos + 1
            eta_seconds = avg_chapter_seconds * (remaining_chapters + 1)
            eta_text = format_duration(eta_seconds)
        else:
            eta_text = "calculating..."

        print(
            f"[{chapter_pos:02d}/{len(chapters)}] "
            f"{chapter.title}: {len(chapter.text):,} chars, "
            f"~{estimate_tokens(chapter.text):,} input tokens, "
            f"output budget {chapter_output_budget:,}, "
            f"{len(chunks)} chunk(s), "
            f"ETA ~{eta_text}"
        )

        chapter_started = time.perf_counter()
        chunk_summaries = []

        for i, chunk in enumerate(chunks, 1):
            chunk_path = chapter_dir / f"chunk_{i:02d}.json"
            cached_chunk, cached_usage, cached_elapsed = cached_chunk_summary(
                chunk_path,
                chapter.number,
                i,
                len(chunks),
            )

            if cached_chunk is not None:
                print(f"    Chunk {i}/{len(chunks)}: [CACHED]")
                summary = cached_chunk
                usage = cached_usage
                elapsed = cached_elapsed
            else:
                prompt = CHUNK_PROMPT.format(
                    book_title=book_title,
                    book_language=book_language or "metadata unavailable",
                    summary_language=summary_language,
                    chapter_title=chapter.title,
                    chunk_index=i,
                    chunk_count=len(chunks),
                    chunk=chunk,
                )

                summary, usage, elapsed = chat(
                    args.base_url,
                    model,
                    prompt,
                    chapter_output_budget,
                    args.temperature,
                    args.timeout,
                    effective_reasoning,
                    send_reasoning,
                    args.reasoning_max_tokens,
                    args.final_answer_tokens,
                )

                save_json(
                    chunk_path,
                    {
                        "chapter": chapter.number,
                        "chunk": i,
                        "chunk_count": len(chunks),
                        "summary": summary,
                        "elapsed_seconds": elapsed,
                        "usage": usage,
                    },
                )

            total_elapsed += elapsed
            p, c, r = usage_totals(usage)
            total_prompt_tokens += p
            total_completion_tokens += c
            total_reasoning_tokens += r
            chunk_summaries.append(summary)

        # If there is only one chunk, its summary is already the chapter summary.
        if len(chunk_summaries) == 1:
            chapter_summary = chunk_summaries[0]
            merge_elapsed = 0.0
            merge_usage = {}
        else:
            merge_prompt = MERGE_PROMPT.format(
                book_title=book_title,
                book_language=book_language or "metadata unavailable",
                summary_language=summary_language,
                chapter_title=chapter.title,
                summaries="\n\n".join(
                    f"--- Chunk {i} ---\n{s}"
                    for i, s in enumerate(chunk_summaries, 1)
                ),
            )

            merge_output_budget = choose_output_tokens(
                merge_prompt,
                cap=args.max_tokens_merge,
                minimum=(
                    args.reasoning_min_output_tokens
                    if effective_reasoning != "off"
                    else args.min_output_tokens
                ),
                ratio=(
                    args.reasoning_output_ratio
                    if effective_reasoning != "off"
                    else args.output_ratio
                ),
            )
            chapter_summary, merge_usage, merge_elapsed = chat(
                args.base_url,
                model,
                merge_prompt,
                merge_output_budget,
                args.temperature,
                args.timeout,
                effective_reasoning,
                send_reasoning,
                args.reasoning_max_tokens,
                args.final_answer_tokens,
            )

            total_elapsed += merge_elapsed
            p, c, r = usage_totals(merge_usage)
            total_prompt_tokens += p
            total_completion_tokens += c
            total_reasoning_tokens += r

        chapter_record = {
            "chapter": chapter.number,
            "title": chapter.title,
            "summary": chapter_summary,
            "source_characters": len(chapter.text),
            "chunk_count": len(chunks),
            "merge_elapsed_seconds": merge_elapsed,
            "merge_usage": merge_usage,
        }
        save_json(chapter_summary_path, chapter_record)

        all_chapter_summaries.append(
            f"--- {chapter.title} ---\n{chapter_summary}"
        )
        completed_chapters_elapsed += time.perf_counter() - chapter_started

    # Final whole-book summary.
    book_summary_path = output_dir / "book_summary.json"
    cached_book_summary = None

    if book_summary_path.exists() and (output_dir / "book_summary.md").exists():
        try:
            record = load_json(book_summary_path)
            if (
                record.get("book_title") == book_title
                and record.get("model") == model
                and str(record.get("summary", "")).strip()
            ):
                cached_book_summary = record
        except (OSError, ValueError, TypeError):
            cached_book_summary = None

    if cached_book_summary is not None:
        final_summary = str(cached_book_summary["summary"]).strip()
        final_usage = cached_book_summary.get("usage") or {}
        final_elapsed = float(cached_book_summary.get("elapsed_seconds", 0.0) or 0.0)
        print("\nComplete-book summary: [CACHED]")
        p, c, r = usage_totals(final_usage)
        total_prompt_tokens += p
        total_completion_tokens += c
        total_reasoning_tokens += r
        total_elapsed += final_elapsed
    else:
        print("\nGenerating complete-book summary...")

        raw_book_summaries = "\n\n".join(all_chapter_summaries)
        book_prompt = BOOK_PROMPT.format(
            book_title=book_title,
            book_language=book_language or "metadata unavailable",
            summary_language=summary_language,
            summaries=raw_book_summaries,
        )

        book_output_budget = choose_output_tokens(
            book_prompt,
            cap=args.max_tokens_book,
            minimum=(
                args.reasoning_min_output_tokens
                if effective_reasoning != "off"
                else args.min_output_tokens
            ),
            ratio=(
                args.reasoning_output_ratio
                if effective_reasoning != "off"
                else args.output_ratio
            ),
        )

        # estimate_tokens() is intentionally conservative enough for chapter
        # decisions, but the book prompt contains model-generated summaries
        # whose tokenizer density can be very different (especially in
        # Hungarian). If the estimated prompt is close to the context limit,
        # first compress the chapter summaries in deterministic batches. This
        # avoids LM Studio rejecting the request because the real tokenizer
        # count is higher than our character-based estimate.
        book_input_tokens = estimate_tokens(book_prompt)
        context_for_book = effective_context
        # Character-based token estimates can significantly underestimate
        # LM Studio's actual tokenizer count, especially for generated
        # Hungarian summaries. Keep a generous reserve for tokenizer
        # overhead and API prompt formatting. For the book-level request we
        # therefore compact before the prompt gets anywhere near the raw
        # context limit.
        book_context_safety_margin = 1024
        requested_book_output_budget = book_output_budget
        max_safe_input = max(1000, context_for_book - requested_book_output_budget - book_context_safety_margin)

        if book_input_tokens > max_safe_input:
            print(
                f"Book summary input is too large for the {context_for_book:,}-token context "
                f"(~{book_input_tokens:,} estimated tokens); compacting chapter summaries..."
            )

            # Build deterministic batches by estimated token size. Keep each
            # digest request well below half the context so its output also has
            # ample room.
            batch_limit = max(1200, (context_for_book - 2048) // 2)
            batches: list[str] = []
            current: list[str] = []
            current_tokens = 0
            for chapter_summary in all_chapter_summaries:
                chapter_tokens = estimate_tokens(chapter_summary)
                if current and current_tokens + chapter_tokens > batch_limit:
                    batches.append("\n\n".join(current))
                    current = []
                    current_tokens = 0
                current.append(chapter_summary)
                current_tokens += chapter_tokens
            if current:
                batches.append("\n\n".join(current))

            compacted: list[str] = []
            for i, batch in enumerate(batches, 1):
                digest_prompt = BOOK_DIGEST_PROMPT.format(
                    book_title=book_title,
                    summary_language=summary_language,
                    summaries=batch,
                )
                digest_budget = min(1024, max(512, context_for_book // 8))
                digest_budget = fit_output_budget_to_context(
                    estimate_tokens(digest_prompt),
                    digest_budget,
                    context_for_book,
                    safety_margin=512,
                )
                print(
                    f"  Digest batch {i}/{len(batches)}: "
                    f"~{estimate_tokens(digest_prompt):,} input, "
                    f"{digest_budget:,} output"
                )
                digest, digest_usage, digest_elapsed = chat(
                    args.base_url,
                    model,
                    digest_prompt,
                    digest_budget,
                    args.temperature,
                    args.timeout,
                    effective_reasoning,
                    send_reasoning,
                    args.reasoning_max_tokens,
                    args.final_answer_tokens,
                )
                if not digest.strip():
                    raise RuntimeError(f"Book digest batch {i} returned no final answer.")
                compacted.append(digest.strip())
                total_elapsed += digest_elapsed
                p_tokens, c_tokens, r_tokens = usage_totals(digest_usage)
                total_prompt_tokens += p_tokens
                total_completion_tokens += c_tokens
                total_reasoning_tokens += r_tokens

            book_prompt = BOOK_PROMPT.format(
                book_title=book_title,
                book_language=book_language or "metadata unavailable",
                summary_language=summary_language,
                summaries="\n\n".join(
                    f"--- Intermediate digest {i} ---\n{digest}"
                    for i, digest in enumerate(compacted, 1)
                ),
            )
            book_input_tokens = estimate_tokens(book_prompt)
            print(f"Compacted book summary input: ~{book_input_tokens:,} tokens")

        book_output_budget = fit_output_budget_to_context(
            book_input_tokens,
            book_output_budget,
            context_for_book,
            safety_margin=book_context_safety_margin,
        )
        if book_output_budget != requested_book_output_budget:
            print(
                f"Book summary input: ~{book_input_tokens:,} tokens, "
                f"output budget: {book_output_budget:,} "
                f"(requested {requested_book_output_budget:,}; "
                f"context {context_for_book:,})"
            )
        else:
            print(
                f"Book summary input: ~{book_input_tokens:,} tokens, "
                f"output budget: {book_output_budget:,}"
            )

        final_summary, final_usage, final_elapsed = chat(
            args.base_url,
            model,
            book_prompt,
            book_output_budget,
            args.temperature,
            args.timeout,
            effective_reasoning,
            send_reasoning,
            args.reasoning_max_tokens,
            args.final_answer_tokens,
        )

        total_elapsed += final_elapsed
        p, c, r = usage_totals(final_usage)
        total_prompt_tokens += p
        total_completion_tokens += c
        total_reasoning_tokens += r

        (output_dir / "book_summary.md").write_text(
            f"# {book_title}\n\n"
            f"## Complete plot summary\n\n"
            f"{final_summary}\n",
            encoding="utf-8",
        )

        save_json(
            book_summary_path,
            {
                "book_title": book_title,
                "model": model,
                "model_dir": model_dir_name,
                "run_id": run_id,
                "summary": final_summary,
                "elapsed_seconds": final_elapsed,
                "usage": final_usage,
            },
        )

    save_json(
        output_dir / "run_stats.json",
        {
            "book_title": book_title,
            "model": model,
            "model_dir": model_dir_name,
            "run_id": run_id,
            "model_info": model_info,
            "chapters": len(chapters),
            "completed_chapters": sum(
                1
                for c in chapters
                if (
                    (
                        output_dir
                        / (f"{int(c.number):03d}" if c.number.isdigit() else "epilogue")
                        / "chapter_summary.json"
                    ).exists()
                )
            ),
            "completed_book_summary": bool(cached_book_summary or (output_dir / "book_summary.json").exists()),
            "total_elapsed_seconds": total_elapsed,
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "total_reasoning_tokens": total_reasoning_tokens,
        },
    )

    print("\nDone.")
    print(f"Total time: {total_elapsed / 60:.1f} min")
    if total_prompt_tokens or total_completion_tokens:
        print(f"Prompt tokens:     {total_prompt_tokens:,}")
        print(f"Completion tokens: {total_completion_tokens:,}")
        print(f"Reasoning tokens:   {total_reasoning_tokens:,}")
    print(f"Results: {output_dir}")



def main():
    parser = argparse.ArgumentParser(
        description="Benchmark an LM Studio model by summarizing an EPUB chapter by chapter."
    )
    parser.add_argument("epub", help="Input EPUB file")
    parser.add_argument(
        "--output",
        default="./results",
        help="Benchmark results root directory (default: ./results)",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:1234",
        help="LM Studio server base URL (default: http://localhost:1234)",
    )
    parser.add_argument(
        "--context",
        type=int,
        default=None,
        help="Override context size for chunk decisions. By default use the loaded instance context_length from LM Studio.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=500,
        help="Character overlap used only when a chapter must be split (default: 500)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Maximum approximate input tokens per chunk. If omitted, use the existing adaptive chunking behavior.",
    )
    parser.add_argument(
        "--max-tokens-chunk",
        type=int,
        default=8192,
        help="Maximum generated tokens per chunk summary (default: 8192; reasoning models need extra room)",
    )
    parser.add_argument(
        "--max-tokens-merge",
        type=int,
        default=8192,
        help="Maximum generated tokens for a multi-chunk chapter merge (default: 8192)",
    )
    parser.add_argument(
        "--max-tokens-book",
        type=int,
        default=8192,
        help="Maximum generated tokens for the complete-book summary (dynamic budget is chosen up to this cap)",
    )
    parser.add_argument(
        "--min-output-tokens",
        type=int,
        default=2048,
        help="Minimum dynamic output budget (default: 2048)",
    )
    parser.add_argument(
        "--output-ratio",
        type=float,
        default=0.5,
        help="Dynamic output budget as a fraction of estimated input tokens (default: 0.5)",
    )
    parser.add_argument(
        "--reasoning-output-ratio",
        type=float,
        default=1.0,
        help="Dynamic output budget ratio for reasoning models (default: 1.0)",
    )
    parser.add_argument(
        "--reasoning-min-output-tokens",
        type=int,
        default=4096,
        help="Minimum output budget for reasoning models (default: 4096)",
    )
    parser.add_argument(
        "--reasoning-max-tokens",
        type=int,
        default=8192,
        help=(
            "Maximum reasoning allowance used before reserving final-answer space "
            "(default: 8192). LM Studio exposes one combined output budget, so this "
            "is implemented as a bounded first-pass budget plus a final-answer reserve."
        ),
    )
    parser.add_argument(
        "--final-answer-tokens",
        type=int,
        default=8192,
        help=(
            "Final-answer token budget used for the fallback pass when reasoning "
            "is enabled (default: 8192). The reasoning and final-answer budgets "
            "together must stay within --max-tokens-*."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature. 0.0 is recommended for benchmark runs.",
    )
    parser.add_argument(
        "--summary-language",
        default="source",
        help="Language of generated summaries: source (default), english, or an explicit ISO language code such as hu, de, fr.",
    )
    parser.add_argument(
        "--reasoning",
        choices=["off", "low", "medium", "high", "on"],
        default="off",
        help="Requested LM Studio reasoning mode. If unsupported by the loaded model, the script automatically uses a supported mode.",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Resume exactly the specified benchmark run directory. Without this option a new run is always created.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=900,
        help="HTTP timeout in seconds per request",
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
