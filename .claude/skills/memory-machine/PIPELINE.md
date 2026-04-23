# Pipeline

Execute these steps in order per invocation.

## 1. Resolve input

| Input | Tool |
| --- | --- |
| File path | `Read` |
| URL | `WebFetch` |
| Pasted text in chat | Use as-is |

If user gives a directory or ambiguous reference, ask once for the specific file/URL.

## 2. Extract highlights

Priority order — stop at the first that finds matches:

1. **Obsidian highlights** — regex `==([^=]+)==`
2. **Markdown blockquotes** — lines starting with `> ` (merge consecutive blockquote lines into one highlight)
3. **Explicit highlight markers** — e.g. `<mark>...</mark>`, `**...**` *only if user instructed to use bold*
4. **Fallback**: if no markers found, identify 5–15 salient passages yourself. Before generating, show the user the count and first 2 passages, ask: "Proceed with N passages, or adjust?" Wait for confirmation.

Deduplicate identical highlights. Preserve order.

## 3. Extract source metadata

Needed once per document. Look in this order:

1. **YAML frontmatter** at top of markdown: `title`, `author`, `url` / `source`
2. **HTML `<title>`** / `<meta author>` for WebFetch content
3. **Filename** → title (e.g. `deep-work-cal-newport.md` → title "Deep Work", author "Cal Newport" is a guess — ask)
4. If none of the above work, ask user for `title` and `author`. `url` is optional.

## 4. Per-highlight loop

For each highlight:

### 4a. Interpret

Write 1–2 sentences answering: *why does this highlight matter in the context of the surrounding text?* Use the paragraphs before and after the highlight. This is internal scaffolding — not output — and anchors the generation step.

If the source is a short pasted snippet with no surrounding context, skip interpretation and proceed directly.

### 4b. Generate

Apply `rubrics/generate.md`. Emit one or more Q/A pairs in this exact format:

```
Q. [question]
A. [answer]
```

A single highlight can produce multiple cards when it carries multiple distinct ideas. Err toward fewer, sharper cards.

### 4c. Self-grade

Apply `rubrics/grade.md` to each generated card. Assign T0, T1, T2, or T3.

**Discipline**: do not relax the rubric. Generic questions ("What is X?"), multi-answer questions, and questions that lose the source's specific framing are T1, not T2 — even if the answer is good. Re-read the T1-vs-T2 section of `grade.md` before classifying borderline cards.

### 4d. Filter

Keep cards graded T2 or T3. Discard T0 and T1. Record counts for the final summary.

## 5. Write output

Write all kept cards to `cards/<source-slug>.csv` using the format defined in `ANKI_EXPORT.md`.

Source-slug rules:
- Derive from filename without extension, or URL hostname + path slug, or first 4 words of title
- Lowercase, kebab-case, ASCII-only
- If `cards/<slug>.csv` exists, append timestamp: `<slug>-2026-04-23.csv`

## 6. Summarize

Output a short report in chat:

```
Memory Machine: cards/<slug>.csv

Highlights processed: N
Cards kept: M (T3: x, T2: y)
Discarded: K (T1: a, T0: b)

Import into Anki: File → Import → select CSV, Fields separated by Comma, allow HTML: off.
```

## Failure modes and responses

| Situation | Response |
| --- | --- |
| Zero highlights extracted | Tell user, show first 500 chars of input, ask how they want to mark highlights |
| All cards discarded (M=0) | Tell user — the source may be too abstract/narrative. Offer to regenerate with a lower bar or suggest better highlighting. Do not write an empty CSV. |
| WebFetch fails | Ask user to paste content or provide a mirror |
| File not found | List what's in the directory, ask for correction |
