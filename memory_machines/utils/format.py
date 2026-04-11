from memory_machines.utils.types import Highlight, MemoryPrompt

user_prompt_template = """I need your help with the following document: 

## [{title}]({url}) by {author}

**Highlight:**

I highlighted the following text:
> {highlight}


**Interpretation of the highlight within the context of the document:**
{highlight_interpretation}

## Memory Prompt

{content}
"""


def format_highlight_text(row: MemoryPrompt | Highlight) -> str:
    # replace newlines with another markdown quote marker
    formatted_highlight = [line.strip() for line in row["highlight"].split("\n")]
    formatted_highlight = " ".join(formatted_highlight)
    return formatted_highlight


def format_user_message(row: MemoryPrompt) -> str:
    source_meta = row["source_meta"]
    assert isinstance(source_meta, dict), "Source meta is not a dictionary"

    assert "author" in source_meta, "Author is not in source meta"
    assert "title" in source_meta, "Title is not in source meta"

    url = row["source_url"]
    assert isinstance(url, str), "URL is not a string"

    return user_prompt_template.format(
        title=source_meta["title"],
        url=url,
        author=source_meta["author"],
        highlight=format_highlight_text(row),
        highlight_interpretation=row["highlight_interpretation"],
        content=row["content"],
    )
