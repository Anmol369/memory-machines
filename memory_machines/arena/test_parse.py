import pytest
from memory_machines.arena.parse import parse_memory_prompts


def test_parse_standard_qa_pairs():
    """Test parsing standard Q/A format"""
    result_text = "Q. What is Python?\nA. A programming language"
    prompts = parse_memory_prompts(result_text)

    assert len(prompts) == 1
    assert prompts[0] == "Q. What is Python?\nA. A programming language"


def test_parse_qa_with_trailing_newlines():
    """Test parsing Q/A pairs with trailing newlines"""
    result_text = "Q. What is Python?\nA. A programming language\n\n\n"
    prompts = parse_memory_prompts(result_text)

    assert len(prompts) == 1
    # Should not include trailing newlines in capture
    assert prompts[0] == "Q. What is Python?\nA. A programming language"
    assert not prompts[0].endswith("\n")


def test_parse_qa_with_markdown_delimiter():
    """Test parsing Q/A pairs followed by ---"""
    result_text = "Q. What is Python?\nA. A programming language\n---\nSome other text"
    prompts = parse_memory_prompts(result_text)

    assert len(prompts) == 1
    # Should stop at --- delimiter
    assert prompts[0] == "Q. What is Python?\nA. A programming language"
    assert "---" not in prompts[0]
    assert "Some other text" not in prompts[0]


def test_parse_multiple_qa_pairs():
    """Test parsing multiple Q/A pairs"""
    result_text = """Q. What is Python?
A. A programming language

Q. What is JavaScript?
A. A scripting language"""
    prompts = parse_memory_prompts(result_text)

    assert len(prompts) == 2
    assert "Python" in prompts[0]
    assert "JavaScript" in prompts[1]


def test_parse_qa_with_code_fences():
    """Test parsing stops at code fences"""
    result_text = "Q. What is Python?\nA. A programming language\n```python\ncode here\n```"
    prompts = parse_memory_prompts(result_text)

    assert len(prompts) == 1
    # Should stop at ```
    assert "```" not in prompts[0]
    assert "code here" not in prompts[0]


def test_parse_empty_response():
    """Test handling of empty response"""
    result_text = ""
    prompts = parse_memory_prompts(result_text)

    # Should return empty list
    assert prompts == []


def test_parse_malformed_qa_missing_answer():
    """Test handling of malformed Q/A (missing A.)"""
    result_text = "Q. What is Python?\nSome text without A."
    prompts = parse_memory_prompts(result_text)

    # Should return empty list or handle gracefully
    assert prompts == []


def test_parse_malformed_qa_missing_question():
    """Test handling of malformed Q/A (missing Q.)"""
    result_text = "A. A programming language"
    prompts = parse_memory_prompts(result_text)

    # Should return empty list
    assert prompts == []


def test_parse_qa_with_multiline_answer():
    """Test parsing Q/A with multi-line answers"""
    result_text = """Q. What is Python?
A. Python is a high-level programming language.
It is widely used for web development, data analysis, and automation."""
    prompts = parse_memory_prompts(result_text)

    assert len(prompts) == 1
    assert "high-level programming language" in prompts[0]
    assert "web development" in prompts[0]


def test_parse_qa_with_markdown_delimiter_after_multiple():
    """Test parsing multiple Q/A pairs followed by ---"""
    result_text = """Q. What is Python?
A. A programming language

Q. What is JavaScript?
A. A scripting language

---
Additional content"""
    prompts = parse_memory_prompts(result_text)

    assert len(prompts) == 2
    assert "Python" in prompts[0]
    assert "JavaScript" in prompts[1]
    assert "Additional content" not in prompts[0]
    assert "Additional content" not in prompts[1]


def test_parse_qa_with_special_characters():
    """Test parsing Q/A with special characters in content"""
    result_text = """Q. What's the difference between & and &&?
A. & is bitwise AND, && is logical AND"""
    prompts = parse_memory_prompts(result_text)

    assert len(prompts) == 1
    assert "&" in prompts[0]
    assert "&&" in prompts[0]


def test_parse_qa_with_periods_in_answer():
    """Test parsing Q/A when answer contains periods"""
    result_text = """Q. What is Python used for?
A. Python is used for web dev., data analysis, etc. It's very versatile."""
    prompts = parse_memory_prompts(result_text)

    assert len(prompts) == 1
    assert "web dev." in prompts[0]
    assert "etc." in prompts[0]


def test_parse_qa_with_whitespace_variations():
    """Test parsing with various whitespace patterns"""
    result_text = """Q.What is Python?
A.A programming language"""
    prompts = parse_memory_prompts(result_text)

    # Should handle Q. and A. without space after
    assert len(prompts) == 1


def test_parse_qa_followed_by_next_qa():
    """Test that Q. delimiter properly separates prompts"""
    result_text = """Q. First question?
A. First answer
Q. Second question?
A. Second answer"""
    prompts = parse_memory_prompts(result_text)

    assert len(prompts) == 2
    # First prompt should not include second Q/A
    assert "Second question" not in prompts[0]
    assert "Second answer" not in prompts[0]
