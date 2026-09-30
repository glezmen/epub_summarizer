# EPUB LLM Benchmark

A small, reproducible benchmark for comparing local LLMs on long-form EPUB summarization using [LM Studio](https://lmstudio.ai/) as the local inference server.

The benchmark:

- extracts chapters from an EPUB in reading order;
- sends each chapter to the locally loaded LLM;
- generates a factual summary for every chapter;
- generates a complete-book summary from the chapter summaries;
- automatically detects the currently loaded LM Studio model;
- uses the loaded model's actual context length for chunking decisions;
- dynamically chooses an output-token budget based on the input size;
- records timing and token-usage information when provided by LM Studio;
- stores every run separately so different models or repeated runs do not overwrite previous results.

The intended use is to compare local models on **factual summarization quality**, for example by comparing their output against a manually prepared or NotebookLM-generated reference summary.

## Requirements

- Python 3.10+ recommended
- [LM Studio](https://lmstudio.ai/)
- A locally loaded chat/instruct LLM in LM Studio
- LM Studio's local server enabled
- An EPUB file

The script uses only Python's standard library. No `pip install` is required.

## LM Studio setup

Load exactly one LLM in LM Studio before starting the benchmark.

The script discovers the loaded model through LM Studio's native endpoint:

```text
http://localhost:1234/api/v1/models
```

It selects the LLM that has a non-empty `loaded_instances` list and reads the effective `context_length` from the loaded instance.

If multiple LLM instances are loaded, the script stops instead of guessing which model should be benchmarked.

The actual generation request uses LM Studio's OpenAI-compatible endpoint:

```text
http://localhost:1234/v1/chat/completions
```

## Basic usage

```bash
python3 epub_llm_benchmark.py "My Book.epub"
```

Results are written to `./results` by default.

Example:

```bash
python3 epub_llm_benchmark.py \
  "The Butchers Masquerade Dungeon Crawler Carl (Book 5).epub"
```

## Important options

### Change the results directory

```bash
python3 epub_llm_benchmark.py "My Book.epub" --output ./benchmark-results
```

### Change the LM Studio server

```bash
python3 epub_llm_benchmark.py "My Book.epub" \
  --base-url http://localhost:1234
```

### Reasoning

Reasoning is disabled by default, which is recommended for a summarization benchmark:

```bash
--reasoning off
```

The CLI value `off` is translated to LM Studio's API value `none`.

Other supported modes are:

```text
low
medium
high
xhigh
```

`on` is also accepted and maps to `high`.

For a reproducible factual-summary benchmark, keep reasoning disabled unless reasoning itself is one of the variables being tested.

### Temperature

The default is:

```text
--temperature 0.0
```

This is recommended for benchmark runs because it reduces sampling variability.

## Context and chunking

The script does **not** split every chapter into arbitrary fixed-size chunks.

Instead, it estimates the input token count and checks whether the complete chapter fits into the available context after reserving space for the generated answer and prompt overhead.

If it fits, the complete chapter is sent in one request.

Only chapters that do not fit are split into deterministic chunks. A small overlap is used for those fallback chunks.

The default overlap is 500 characters:

```bash
--chunk-overlap 500
```

The context size is automatically taken from the loaded LM Studio instance. It can be overridden if necessary:

```bash
--context 32768
```

## Dynamic output-token budget

The script does not use one unnecessarily large output limit for every request.

It estimates the input size and selects an output budget based on:

```text
estimated input tokens × output ratio
```

The default ratio is `0.5`, with a minimum of 2048 tokens and the following maximum caps:

```text
Chapter/chunk summary:       4096
Multi-chunk chapter merge:   4096
Complete-book summary:       8192
```

These can be changed from the command line, for example:

```bash
--max-tokens-chunk 4096
--max-tokens-merge 4096
--max-tokens-book 8192
--min-output-tokens 2048
--output-ratio 0.5
```

The output budget is only a generation limit. It does not mean the model will necessarily use that many tokens.

## Output structure

Every execution gets its own timestamped directory.

For example:

```text
results/
└── qwen-qwen3.8-27b/
    ├── run-20260930-134512/
    │   ├── book_info.json
    │   ├── run_stats.json
    │   ├── book_summary.md
    │   ├── book_summary.json
    │   ├── 001/
    │   │   ├── chunk_01.json
    │   │   └── chapter_summary.json
    │   ├── 002/
    │   │   ├── chunk_01.json
    │   │   └── chapter_summary.json
    │   └── ...
    └── run-20260930-150301/
        └── ...
```

The model directory is derived from the LM Studio model key. For example:

```text
qwen/qwen3.8-27b
```

becomes:

```text
qwen-qwen3.8-27b
```

This allows several models and several runs of the same model to coexist without overwriting earlier results.

## Generated files

### `book_info.json`

Contains benchmark configuration and model information, including:

- book title and source filename;
- model key and model metadata;
- selected variant and quantization information when provided by LM Studio;
- context length;
- temperature;
- reasoning mode;
- output-budget settings;
- chapter metadata.

### `chunk_XX.json`

Created for every model request used to summarize a chapter chunk.

Contains:

- chapter/chunk identification;
- generated summary;
- elapsed time;
- LM Studio usage information when available.

### `chapter_summary.json`

Contains the final summary for one chapter. For a chapter that fits into one request, this is the direct model response. For a multi-chunk chapter, it is the result of the additional merge step.

### `book_summary.md`

Human-readable complete-book summary.

### `book_summary.json`

The complete-book summary plus timing and usage information.

### `run_stats.json`

Aggregated timing and token-usage statistics for the run.

## Benchmark methodology

For meaningful model comparisons, keep the benchmark parameters identical between runs.

Recommended baseline:

```text
Temperature: 0.0
Reasoning:   off
Context:     model's loaded context length
Chunking:    automatic
```

The same EPUB and the same prompts should be used for every model.

A useful evaluation workflow is:

```text
EPUB
 │
 ├──> Model A ──> chapter summaries ──> complete summary
 │
 ├──> Model B ──> chapter summaries ──> complete summary
 │
 └──> Model C ──> chapter summaries ──> complete summary
                  
Reference summary (e.g. NotebookLM)
                  │
                  └──> factual comparison / evaluation
```

The goal should be to measure more than textual similarity. Useful evaluation dimensions include:

- important-event coverage;
- factual accuracy;
- character/action accuracy;
- causal-relationship accuracy;
- chronological accuracy;
- important omissions;
- hallucinated events or details.

A reference summary should be treated as a benchmark reference rather than unquestionable ground truth. When an important discrepancy appears, the original EPUB text should be used to verify the fact.

## Privacy

The book text is sent to the LM Studio server configured by `--base-url`. With the default configuration, inference is performed locally on the machine running LM Studio.

The script does not intentionally send the EPUB text to a remote API.

## Limitations

- Chapter detection is based on numbered headings (`1`, `2`, `3`, ...) and an `EPILOGUE` heading.
- EPUBs with a different chapter structure may require changes to the chapter detector.
- Token counts used for chunk decisions are estimates; actual tokenization is model-dependent.
- Timing and token statistics depend on what LM Studio reports through its API.
- The benchmark measures summarization behavior, not general model quality.

## Example console output

```text
==========================================================
 EPUB LLM BENCHMARK
==========================================================

Book:       The Butcher's Masquerade: Dungeon Crawler Carl Book 5
Chapters:   76 + Epilogue
Model:      Qwen3.8 27B
Model key:  qwen/qwen3.8-27b
Variant:    qwen/qwen3.8-27b@4bit
Quant:      4bit
API:        http://localhost:1234
Temperature:0.0
Reasoning:  off
Context:    61,696 tokens
Chunking:   whole chapter when it fits
Output:     /.../results/qwen-qwen3.8-27b/run-20260930-134512
Run ID:     run-20260930-134512

[01/76] Chapter 1: 27,109 chars, ~7,745 input tokens, output budget 4,096, 1 chunk(s), ETA ~calculating...
```

## License

Add the project's license here if/when one is selected.
