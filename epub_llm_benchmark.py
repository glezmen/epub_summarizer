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

from datetime import datetime
import argparse
import json
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


def read_epub(epub_path: Path) -> tuple[str, list[Chapter]]:
    with zipfile.ZipFile(epub_path, "r") as z:
        opf_path = find_opf_path(z)
        opf_data = z.read(opf_path)
        root = ET.fromstring(opf_data)

        manifest = {}
        for item in root.findall(f".//{{{OPF_NS}}}manifest/{{{OPF_NS}}}item"):
            manifest[item.attrib["id"]] = item.attrib["href"]

        spine = []
        for itemref in root.findall(f".//{{{OPF_NS}}}spine/{{{OPF_NS}}}itemref"):
            spine.append(itemref.attrib["idref"])

        book_title = ""
        title_el = root.find(f".//{{{DC_NS}}}title")
        if title_el is not None and title_el.text:
            book_title = title_el.text.strip()

        chapters: list[Chapter] = []

        for idref in spine:
            href = manifest.get(idref)
            if not href:
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

            # This EPUB stores chapters as h1 containing just "1", "2", ...
            # and the epilogue as "EPILOGUE".
            heading = headings[0] if headings else ""

            m = re.fullmatch(r"(\d+)", heading.strip())
            if m:
                number = m.group(1)
                chapters.append(
                    Chapter(number, f"Chapter {number}", full_path, text)
                )
            elif heading.strip().upper() == "EPILOGUE":
                chapters.append(
                    Chapter("EPILOGUE", "Epilogue", full_path, text)
                )

        if not chapters:
            raise RuntimeError(
                "No numbered chapters or EPILOGUE were detected. "
                "Inspect the EPUB structure or add a custom chapter detector."
            )

        return book_title, chapters


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

def get_context_limit(model_info: dict, default: int = 32768) -> int:
    for key in ("context_length", "contextLength", "context"):
        value = model_info.get(key)
        if isinstance(value, (int, float)) and value > 0:
            return int(value)
    return default

def chapter_chunks(text: str, context_limit: int, reserved_output_tokens: int, prompt_overhead_tokens: int, overlap_chars: int) -> list[str]:
    """Send the whole chapter when it fits; otherwise adaptively chunk it."""
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


def get_loaded_model(base_url: str) -> tuple[str, dict, int]:
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

    return info["key"], info, context_length


def chat(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    timeout: int,
    reasoning: str,
) -> tuple[str, dict, float]:
    url = base_url.rstrip("/") + "/v1/chat/completions"

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a precise literary text summarizer. "
                    "You must only use information present in the supplied text. "
                    "Do not invent events, characters, motivations, or facts."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    # LM Studio expects "none" to disable reasoning. Keep the CLI
    # spelling "off" for readability, but translate it for the API.
    reasoning_effort = {
        "off": "none",
        "on": "high",
    }.get(reasoning, reasoning)
    payload["reasoning_effort"] = reasoning_effort

    started = time.perf_counter()
    response = api_request(url, payload, timeout)
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
        details = usage.get("completion_tokens_details", {}) or {}
        reasoning_tokens = details.get("reasoning_tokens", 0)
        raise RuntimeError(
            "LM Studio returned an empty final answer. "
            f"finish_reason={finish_reason}, "
            f"prompt_tokens={usage.get('prompt_tokens', 0)}, "
            f"completion_tokens={usage.get('completion_tokens', 0)}, "
            f"reasoning_tokens={reasoning_tokens}. "
            "The output budget may have been exhausted by reasoning."
        )
    return content, usage, elapsed


# -----------------------------
# Prompts
# -----------------------------

CHUNK_PROMPT = """We are running a reproducible benchmark of local LLMs on a novel.

Book: {book_title}
Chapter: {chapter_title}
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

TEXT:
{chunk}
"""

MERGE_PROMPT = """We are running a reproducible benchmark of local LLMs on a novel.

Book: {book_title}
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

CHUNK SUMMARIES:
{summaries}
"""

BOOK_PROMPT = """We are running a reproducible benchmark of local LLMs on a novel.

Book: {book_title}

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

CHAPTER SUMMARIES:
{summaries}
"""


# -----------------------------
# Benchmark runner
# -----------------------------

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


def run(args):
    epub_path = Path(args.epub).expanduser().resolve()
    base_output_dir = Path(args.output).expanduser().resolve()
    base_output_dir.mkdir(parents=True, exist_ok=True)

    book_title, chapters = read_epub(epub_path)

    model, model_info, detected_context = get_loaded_model(args.base_url)
    output_dir, model_dir_name, run_id = create_run_output_dir(
        base_output_dir, model
    )

    print("=" * 58)
    print(" EPUB LLM BENCHMARK")
    print("=" * 58)
    print()
    print(f"Book:       {book_title}")
    print(f"Chapters:   {len(chapters)} + Epilogue")
    effective_context = args.context if args.context is not None else detected_context
    print(f"Model:      {model_info.get('display_name', model)}")
    print(f"Model key:  {model}")
    print(f"Variant:    {model_info.get('selected_variant', '?')}")
    print(f"Quant:      {(model_info.get('quantization') or {}).get('name', '?')}")
    print(f"API:        {args.base_url}")
    print(f"Temperature:{args.temperature}")
    print(f"Reasoning:  {args.reasoning}")
    print(f"Context:    {effective_context:,} tokens")
    print("Chunking:   whole chapter when it fits")
    print(f"Output:     {output_dir}")
    print(f"Run ID:     {run_id}")
    print()

    save_json(
        output_dir / "book_info.json",
        {
            "book_title": book_title,
            "source_file": epub_path.name,
            "model": model,
            "model_dir": model_dir_name,
            "run_id": run_id,
            "model_info": model_info,
            "base_url": args.base_url,
            "temperature": args.temperature,
            "reasoning": args.reasoning,
            "min_output_tokens": args.min_output_tokens,
            "output_ratio": args.output_ratio,
            "max_tokens_chunk": args.max_tokens_chunk,
            "max_tokens_merge": args.max_tokens_merge,
            "max_tokens_book": args.max_tokens_book,
            "context": effective_context,
            "loaded_instance": (model_info.get("loaded_instances") or [{}])[0],
            "chunking": "whole chapter when it fits; adaptive fallback otherwise",
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
    completed_chapters_elapsed = 0.0

    for chapter_pos, chapter in enumerate(chapters, 1):
        chapter_dir = output_dir / f"{int(chapter.number):03d}" if chapter.number.isdigit() else output_dir / "epilogue"
        chapter_dir.mkdir(parents=True, exist_ok=True)

        context_limit = effective_context
        chapter_output_budget = choose_output_tokens(
            chapter.text,
            cap=args.max_tokens_chunk,
            minimum=args.min_output_tokens,
            ratio=args.output_ratio,
        )
        chunks = chapter_chunks(
            chapter.text,
            context_limit=context_limit,
            reserved_output_tokens=chapter_output_budget,
            prompt_overhead_tokens=1000,
            overlap_chars=args.chunk_overlap,
        )

        if completed_chapters_elapsed > 0 and chapter_pos > 1:
            avg_chapter_seconds = completed_chapters_elapsed / (chapter_pos - 1)
            remaining_chapters = len(chapters) - chapter_pos + 1
            # The final whole-book summary is one additional request. Use the
            # observed average chapter time as a conservative first estimate.
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
            prompt = CHUNK_PROMPT.format(
                book_title=book_title,
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
                args.reasoning,
            )

            total_elapsed += elapsed
            total_prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
            total_completion_tokens += int(usage.get("completion_tokens", 0) or 0)

            chunk_record = {
                "chapter": chapter.number,
                "chunk": i,
                "chunk_count": len(chunks),
                "summary": summary,
                "elapsed_seconds": elapsed,
                "usage": usage,
            }
            save_json(chapter_dir / f"chunk_{i:02d}.json", chunk_record)
            chunk_summaries.append(summary)

        # If there is only one chunk, its summary is already the chapter summary.
        if len(chunk_summaries) == 1:
            chapter_summary = chunk_summaries[0]
            merge_elapsed = 0.0
            merge_usage = {}
        else:
            merge_prompt = MERGE_PROMPT.format(
                book_title=book_title,
                chapter_title=chapter.title,
                summaries="\n\n".join(
                    f"--- Chunk {i} ---\n{s}"
                    for i, s in enumerate(chunk_summaries, 1)
                ),
            )

            merge_output_budget = choose_output_tokens(
                merge_prompt,
                cap=args.max_tokens_merge,
                minimum=args.min_output_tokens,
                ratio=args.output_ratio,
            )
            chapter_summary, merge_usage, merge_elapsed = chat(
                args.base_url,
                model,
                merge_prompt,
                merge_output_budget,
                args.temperature,
                args.timeout,
                args.reasoning,
            )

            total_elapsed += merge_elapsed
            total_prompt_tokens += int(
                merge_usage.get("prompt_tokens", 0) or 0
            )
            total_completion_tokens += int(
                merge_usage.get("completion_tokens", 0) or 0
            )

        chapter_record = {
            "chapter": chapter.number,
            "title": chapter.title,
            "summary": chapter_summary,
            "source_characters": len(chapter.text),
            "chunk_count": len(chunks),
            "merge_elapsed_seconds": merge_elapsed,
            "merge_usage": merge_usage,
        }
        save_json(chapter_dir / "chapter_summary.json", chapter_record)

        all_chapter_summaries.append(
            f"--- {chapter.title} ---\n{chapter_summary}"
        )
        completed_chapters_elapsed += time.perf_counter() - chapter_started

    # Final whole-book summary.
    print("\nGenerating complete-book summary...")

    book_prompt = BOOK_PROMPT.format(
        book_title=book_title,
        summaries="\n\n".join(all_chapter_summaries),
    )

    book_output_budget = choose_output_tokens(
        book_prompt,
        cap=args.max_tokens_book,
        minimum=args.min_output_tokens,
        ratio=args.output_ratio,
    )
    print(
        f"Book summary input: ~{estimate_tokens(book_prompt):,} tokens, "
        f"output budget: {book_output_budget:,}"
    )

    final_summary, final_usage, final_elapsed = chat(
        args.base_url,
        model,
        book_prompt,
        book_output_budget,
        args.temperature,
        args.timeout,
        args.reasoning,
    )

    total_elapsed += final_elapsed
    total_prompt_tokens += int(final_usage.get("prompt_tokens", 0) or 0)
    total_completion_tokens += int(
        final_usage.get("completion_tokens", 0) or 0
    )

    (output_dir / "book_summary.md").write_text(
        f"# {book_title}\n\n"
        f"## Complete plot summary\n\n"
        f"{final_summary}\n",
        encoding="utf-8",
    )

    save_json(
        output_dir / "book_summary.json",
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
            "total_elapsed_seconds": total_elapsed,
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
        },
    )

    print("\nDone.")
    print(f"Total time: {total_elapsed / 60:.1f} min")
    if total_prompt_tokens or total_completion_tokens:
        print(f"Prompt tokens:     {total_prompt_tokens:,}")
        print(f"Completion tokens: {total_completion_tokens:,}")
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
        "--max-tokens-chunk",
        type=int,
        default=4096,
        help="Maximum generated tokens per chunk summary (dynamic budget is chosen up to this cap)",
    )
    parser.add_argument(
        "--max-tokens-merge",
        type=int,
        default=4096,
        help="Maximum generated tokens for a multi-chunk chapter merge (dynamic budget is chosen up to this cap)",
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
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature. 0.0 is recommended for benchmark runs.",
    )
    parser.add_argument(
        "--reasoning",
        choices=["off", "low", "medium", "high", "xhigh", "on"],
        default="off",
        help="LM Studio reasoning mode. 'off' is recommended for summarization benchmark runs.",
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
