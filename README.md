# memory-machines

Research code for evaluating and training LLM-based memory prompt (flashcards) generators for spaced repetition.

## Datasets

Both datasets are published under the [`laddermedia`](https://huggingface.co/laddermedia) HuggingFace org.

| Dataset | Granularity | Purpose |
| --- | --- | --- |
| [`laddermedia/srs-prompts`](https://huggingface.co/datasets/laddermedia/srs-prompts) | one row per candidate memory prompt | Card-level: pluckability classification, reward-model preference pairs, raw input for `srs-highlights` |
| [`laddermedia/srs-highlights`](https://huggingface.co/datasets/laddermedia/srs-highlights) | one row per highlight (with all candidate prompts grouped) | Highlight-level: tiering evaluation, SFT tiering, masked task tiering |

`srs-highlights` is built from `srs-prompts` via `memory_machines.highlight.build_dataset`.

## Setup

```bash
uv sync
```

## Citation

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

## License

Apache-2.0. See [LICENSE](LICENSE).
