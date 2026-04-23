---
name: memory-machine
description: Generate high-quality Anki flashcards from any text, markdown/Obsidian file, pasted highlights, or URL. Uses the research-grade generation + T0–T3 grading rubrics from the memory-machines project. Invoke when the user says "generate cards from ...", "make flashcards from ...", "turn this into Anki cards", "memory machine ...", or asks to convert notes/highlights/articles into spaced-repetition prompts.
---

# Memory Machine — Flashcard Generation Skill

Turns a reader's highlights into Anki-importable memory prompts, using the rubrics from `laddermedia/memory-machines` (Kirkby & Matuschak, 2026).

## Triggers

Invoke this skill when the user asks to:

- Generate / make / create flashcards, memory prompts, or Anki cards from a file, URL, or pasted text
- Turn notes, highlights, or an article into spaced-repetition cards
- "Use memory machine on X" / "run memory machine on X"

## Input modes

The user may specify a source in any of three ways. Detect which and proceed.

| Mode | Example utterance | Action |
| --- | --- | --- |
| File path | `generate cards from notes/book.md` | `Read` the file |
| Pasted text | User pastes text (often blockquotes) directly in chat | Use inline content |
| URL | `generate cards from https://...` | `WebFetch` the URL |

If the source is ambiguous, ask once — do not guess.

## Procedure

Follow `PIPELINE.md` step-by-step. Do not deviate from the ordering (ingest → extract highlights → extract metadata → per-highlight loop → write CSV → summarize). Every step has a reason grounded in the rubrics.

The rubrics live in this directory:

- `rubrics/generate.md` — how to write a good memory prompt (applied per highlight)
- `rubrics/grade.md` — T0–T3 quality scale (applied per generated card; **keep only T2 and T3**)

The output format is defined in `ANKI_EXPORT.md`.

## Output

Anki-importable CSV at `cards/<source-slug>.csv` with columns `front,back,tags,source,tier`. After writing, summarize in chat:

- N highlights processed
- M cards kept (T3: x, T2: y)
- K cards discarded (T0: a, T1: b)
- Path to CSV + one-line Anki import instruction

## Non-goals

- Do not call the Python research code in `memory_machines/` — this skill does the generation itself using Claude.
- Do not output formats other than Anki CSV in this version.
- Do not write cards that you graded T0 or T1. Silent discard with a count is correct.
