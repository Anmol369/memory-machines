# memory-machines

Research code for evaluating and training LLM-based memory prompt (flashcards) generators for spaced repetition.

---

## Quickstart — Generate cards via Claude Code

If you just want flashcards from your own notes, you do not need to run any Python or provide any API key. A Claude Code skill ships with this repo.

1. Fork and clone this repo, then open it in [Claude Code](https://claude.com/claude-code).
2. In chat, say any of:
   - `generate cards from notes/my-article.md`
   - `generate cards from https://andymatuschak.org/prompts/`
   - Paste highlighted blockquotes directly and say `run memory machine on this`
3. The skill reads your input, extracts highlights (Obsidian `==highlights==`, markdown `> blockquotes`, or salient passages), generates Q/A pairs using the repo's generation rubric, self-grades each card on the T0–T3 scale, and keeps only T2/T3.
4. Output lands in `cards/<source-slug>.csv`, ready to import into Anki (File → Import → Comma-separated).

Skill definition: [`.claude/skills/memory-machine/SKILL.md`](.claude/skills/memory-machine/SKILL.md).

The skill uses the same generation and grading rubrics as the research code below — it is not a diluted version.

---

## Research code

### Datasets

Both datasets are published under the [`laddermedia`](https://huggingface.co/laddermedia) HuggingFace org.

| Dataset | Granularity | Purpose |
| --- | --- | --- |
| [`laddermedia/srs-prompts`](https://huggingface.co/datasets/laddermedia/srs-prompts) | one row per candidate memory prompt | Card-level: pluckability classification, reward-model preference pairs, raw input for `srs-highlights` |
| [`laddermedia/srs-highlights`](https://huggingface.co/datasets/laddermedia/srs-highlights) | one row per highlight (with all candidate prompts grouped) | Highlight-level: tiering evaluation, SFT tiering, masked task tiering |

`srs-highlights` is built from `srs-prompts` via `memory_machines.highlight.build_dataset`.

### Setup

```bash
uv sync
```

### Citation

If you use this code or the associated datasets, please cite:

```bibtex
@misc{memory-machines,
  title  = {Memory Machines: Can LLMs create lasting flashcards from readers' highlights?},
  author = {Kirkby, Ozzie and Matuschak, Andy},
  year   = {2026},
  address = {San Francisco},
  url    = {https://memory-machines.com/}
}
```

### License

Apache-2.0. See [LICENSE](LICENSE).
