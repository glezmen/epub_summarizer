# EPUB LLM Benchmark

A reproducible benchmark for comparing LLMs on long-form EPUB summarization. The benchmark uses [LM Studio](https://lmstudio.ai/) for local inference and can evaluate the generated summaries against a NotebookLM reference using a local LLM, Claude Code, OpenAI, or Anthropic evaluator.

The project has two main scripts:

- `epub_llm_benchmark.py` — generates chapter-by-chapter and complete-book summaries.
- `evaluate_summaries.py` — evaluates one or more completed model runs against a NotebookLM reference and the original EPUB.

## What is measured?

The goal is not textual similarity. The benchmark is intended to measure factual summarization quality, including:

- important-event coverage;
- factual accuracy;
- character/action accuracy;
- cause-and-effect accuracy;
- chronology;
- important omissions;
- unsupported or hallucinated claims;
- importance of covered and omitted facts;
- story-arc coverage;
- generation and evaluation runtime;
- token usage when reported by the inference backend.

The original EPUB is treated as the primary factual source. NotebookLM provides a structured reference summary, but it is not assumed to be infallible. Important discrepancies can therefore be checked against the original book.

## Requirements

- Python 3.10+ recommended
- [LM Studio](https://lmstudio.ai/) for local summarization and/or local evaluation
- A locally loaded chat/instruct LLM in LM Studio
- LM Studio's local server enabled
- An EPUB file

The scripts use only Python's standard library. No `pip install` is required for the local benchmark/evaluator.

## 1. Generate summaries

Load exactly one LLM instance in LM Studio before starting the benchmark.

The script discovers the loaded model through:

```text
http://localhost:1234/api/v1/models
```

It selects the LLM with a non-empty `loaded_instances` list and reads the effective `context_length` from the loaded instance. If multiple LLM instances are loaded, the script stops instead of guessing.

Actual generation uses LM Studio's OpenAI-compatible endpoint:

```text
http://localhost:1234/v1/chat/completions
```

### Basic usage

```bash
python3 epub_llm_benchmark.py "My Book.epub"
```

Results are written to `./results` by default.

### Summary language

By default summaries are generated in the source book's language:

```bash
python3 epub_llm_benchmark.py "My Book.epub"
```

English can be requested explicitly:

```bash
python3 epub_llm_benchmark.py "My Book.epub" --summary-language english
```

An explicit ISO language code can also be supplied:

```bash
python3 epub_llm_benchmark.py "My Book.epub" --summary-language hu
python3 epub_llm_benchmark.py "My Book.epub" --summary-language de
python3 epub_llm_benchmark.py "My Book.epub" --summary-language fr
```

`source` uses the EPUB language metadata when available and instructs the model to follow the source text language.

### Reasoning

Reasoning is disabled by default and is recommended for the baseline summarization benchmark:

```text
--reasoning off
```

The script translates `off` to LM Studio's API value `none`.

Other modes are:

```text
low
medium
high
xhigh
```

`on` is accepted and maps to `high`.

### Temperature

The default is:

```text
--temperature 0.0
```

This is recommended for reproducible benchmark runs.

## Context and chunking

The script does not split every chapter into arbitrary fixed-size chunks.

It estimates the input token count and checks whether the complete chapter fits into the available context after reserving space for the generated answer and prompt overhead. If it fits, the complete chapter is sent in one request.

Only chapters that do not fit are split. Fallback chunks use a small character overlap.

The context size is taken automatically from the loaded LM Studio instance. It can be overridden when necessary:

```bash
--context 32768
```

The parser also filters common front matter and table-of-contents entries so that a TOC is not accidentally counted as a plot chapter.

## Dynamic output-token budget

Output limits are adaptive rather than one fixed value for every request.

The default budget is based on estimated input tokens, with these defaults:

```text
minimum:                  2048
input/output ratio:       0.5
chapter/chunk maximum:    4096
chapter merge maximum:   4096
book summary maximum:     8192
```

They can be changed with:

```bash
--max-tokens-chunk 4096
--max-tokens-merge 4096
--max-tokens-book 8192
--min-output-tokens 2048
--output-ratio 0.5
```

## Output structure

Every execution gets its own timestamped directory, so repeated runs do not overwrite previous results.

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
    │   │   └── ...
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

## 2. Evaluate summaries

Once one or more model runs are available, use the evaluator script with:

1. the original EPUB;
2. the NotebookLM-generated reference summary;
3. the `results` directory produced by `epub_llm_benchmark.py`.

```bash
python3 evaluate_summaries.py \
  "My Book.epub" \
  notebooklm_summary.md \
  results
```

The evaluator discovers all `model/run-*` directories under `results`, so multiple models and repeated runs can be evaluated in one invocation.

### Evaluation model backends

The evaluator supports four backends:

```text
local
claude-code
openai
anthropic
```

#### Local LM Studio evaluator

Default:

```bash
python3 evaluate_summaries.py book.epub notebooklm_summary.md results
```

The evaluator automatically detects the loaded LM Studio LLM in the same way as the benchmark script.

#### Claude Code evaluator

If Claude Code is already installed and authenticated, no Anthropic API key is required:

```bash
python3 evaluate_summaries.py \
  book.epub \
  notebooklm_summary.md \
  results \
  --evaluator claude-code \
  --evaluator-model opus
```

`opus` is the default Claude Code model alias for this backend; `sonnet` can also be selected when supported by the installed Claude Code version.

This uses the local `claude` CLI and therefore uses the Claude Code authentication available on the machine. A Claude web subscription and Anthropic API access are separate mechanisms; this backend specifically uses the authenticated Claude Code CLI.

#### OpenAI API evaluator

```bash
export OPENAI_API_KEY="..."

python3 evaluate_summaries.py \
  book.epub \
  notebooklm_summary.md \
  results \
  --evaluator openai \
  --evaluator-model <model-id>
```

OpenAI API access and ChatGPT subscriptions are separate. The script requires an API key for this backend.

#### Anthropic API evaluator

```bash
export ANTHROPIC_API_KEY="..."

python3 evaluate_summaries.py \
  book.epub \
  notebooklm_summary.md \
  results \
  --evaluator anthropic \
  --evaluator-model <model-id>
```

Anthropic API access and Claude subscriptions are separate. The script requires an API key for this backend.

### Evaluator options

Useful options include:

```bash
--evaluator local|claude-code|openai|anthropic
--evaluator-model MODEL
--temperature 0.0
--reasoning off
--max-tokens-facts 4096
--max-tokens-eval 4096
--max-tokens-book-eval 8192
```

For reproducible evaluation, use temperature `0.0` and keep the evaluator configuration identical across model runs.

## Evaluation methodology

The evaluator does not simply compare summary strings.

### Reference facts

The original EPUB is parsed into chapters and the NotebookLM reference is mapped to those chapters. The evaluator LLM extracts atomic facts and assigns an importance level:

```text
critical
major
moderate
minor
```

### Coverage

Each reference fact is checked against the model summary:

```text
supported
partially_supported
missing
contradicted
```

This makes it possible to distinguish a missing minor detail from a missing critical plot event.

### Model claims

The evaluator also checks claims made by the model summary for unsupported or contradictory information. This captures hallucinations and factual distortions that a simple reference-recall metric would miss.

### Story-level quality

The evaluation can also assess broader plot coverage, including story arcs, major turning points, consequences, and the overall representation of the story.

### Runtime and model metadata

The final report includes, where available:

- source model name/key;
- model variant and quantization;
- parameter information reported by LM Studio;
- context length;
- reasoning mode;
- temperature;
- output-token settings;
- total and per-chapter runtime;
- input/output token usage;
- evaluator provider and model;
- evaluator configuration.

This allows quality to be considered together with generation cost/time rather than using a single quality score in isolation.

## Evaluation output

The evaluator creates a separate timestamped directory, for example:

```text
results/
└── evaluation/
    └── run-20260930-160000/
        ├── evaluator_info.json
        ├── reference_facts.json
        ├── reference/
        ├── qwen-qwen3.8-27b/
        │   └── run-20260930-134512/
        │       ├── 001.json
        │       ├── ...
        │       ├── evaluation.json
        │       └── book_evaluation.json
        ├── evaluation.json
        ├── report.md
        └── report.html
```

`report.html` is intended as the main human-readable result.

The evaluation report should show both aggregate metrics and concrete error examples, especially:

- critical omissions;
- major omissions;
- unsupported claims;
- contradictions;
- chapter-level weaknesses.

## Recommended benchmark workflow

For each model, keep the generation settings identical unless the setting itself is being tested.

```text
                    EPUB
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
       Model A     Model B     Model C
          │           │           │
          ▼           ▼           ▼
      summaries   summaries   summaries
          │           │           │
          └───────────┼───────────┘
                      ▼
              NotebookLM reference
                      │
                      ▼
             evaluate_summaries.py
                      │
             ┌────────┼─────────┐
             ▼        ▼         ▼
           Local   Claude Code  API
             │        │         │
             └────────┼─────────┘
                      ▼
                benchmark report
```

For a serious comparison, it is useful to evaluate the same summaries with more than one independent evaluator. Disagreements between evaluators should be inspected rather than silently averaged away.

## Legacy runs

Older benchmark runs may have been produced by an earlier version of the script using:

```text
results/model/
```

instead of the current:

```text
results/<model-key>/run-<timestamp>/
```

Such a run can be migrated manually after it finishes. It does not need to be regenerated solely because the directory structure changed.

## Privacy and external evaluation

With the default benchmark configuration, EPUB text is sent to the local LM Studio server.

When `evaluate_summaries.py` uses `claude-code`, `openai`, or `anthropic`, the relevant reference/model-summary content is sent to that external evaluator. Do not use an external evaluator for copyrighted, confidential, or otherwise sensitive material unless you are permitted to do so.

API keys must never be committed to the Git repository.

## Limitations

- EPUB chapter structures vary. The parser handles common numbered headings, `Chapter N` headings, epilogues, and several common TOC/front-matter patterns, but unusual EPUBs may still require parser adjustments.
- Estimated token counts are approximate and model-dependent.
- Runtime and token statistics depend on what the inference backend reports.
- NotebookLM is a reference source, not an infallible ground-truth oracle.
- LLM-based evaluation is itself imperfect. Important disagreements should be checked against the original EPUB.
- The benchmark measures summarization behavior, not general model intelligence or overall model quality.

## License

Add the project's license here if/when one is selected.
