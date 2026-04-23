# Anki CSV Export Format

## Schema

Five columns, comma-separated, UTF-8, no BOM, Unix line endings.

| Column | Content |
| --- | --- |
| `front` | The question (Q text, without the `Q.` prefix) |
| `back` | The answer (A text, without the `A.` prefix) |
| `tags` | Space-separated Anki tags. Default: `memory-machine <source-slug>` |
| `source` | Title — Author (or URL if title/author unknown) |
| `tier` | `T2` or `T3` |

## Header

First line is the header row:

```
front,back,tags,source,tier
```

## Escaping rules

Anki uses standard CSV escaping. Apply in this order to each field:

1. If the field contains a `,`, `"`, or newline → wrap the whole field in double quotes `"..."`.
2. Inside a quoted field, replace every `"` with `""`.
3. Newlines inside a field are allowed (Anki treats them as `<br>` on import if "Allow HTML" is off, or as literal newlines in the card body).

Do **not** HTML-encode. Do not use backslash escapes.

## Example

```
front,back,tags,source,tier
What two criteria must an effective Anki prompt satisfy simultaneously?,"Meaningful (captures what the user found interesting) and Stable (can be reliably retrieved after months without the original context).",memory-machine memory-machines-readme,Memory Machines — Kirkby & Matuschak,T3
"In the T0–T3 scale, what is the threshold that separates ""reviewable"" from ""not reviewable""?","The gap between T1 and T2. T2 cards can be reviewed as-is with minor polish; T1 cards cannot be reviewed effectively without substantial restructuring.",memory-machine memory-machines-readme,Memory Machines — Kirkby & Matuschak,T3
```

Note the doubled quotes around "reviewable" inside the second row's question.

## Anki import settings

Tell the user:

- **File → Import → select the CSV**
- Fields separated by: **Comma**
- Allow HTML in fields: **off** (unless they want `<br>` rendering)
- Field mapping: front → Front, back → Back, tags → Tags, source → (skip or map to a custom field), tier → (skip or map to a custom field)
- Duplicates: "Update existing notes with same first field" is a safe default
