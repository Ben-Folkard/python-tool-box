import ast
import difflib
from pathlib import Path
import textwrap
import re

__all__ = [
    "generate_markdown_diff",
    "generate_function_split_markdown_diffs"
]


def generate_markdown_diff(filename_1, filename_2, output_filename="diff.md"):
    file1_lines = Path(filename_1).read_text().splitlines()
    file2_lines = Path(filename_2).read_text().splitlines()

    diff = list(difflib.ndiff(file1_lines, file2_lines))

    md_lines = [f"# Diff `{filename_1}` → `{filename_2}`"]

    i = 0
    line_num1, line_num2 = 1, 1

    while i < len(diff):
        line = diff[i]

        # Removed lines from file1
        if line.startswith("- "):
            md_lines.append("\nFrom:")
            block_from = ["```python"]
            while i < len(diff) and diff[i].startswith("- "):
                block_from.append(f"{line_num1:4d}: {diff[i][2:]}")
                line_num1 += 1
                i += 1
            block_from.append("```")
            md_lines.extend(block_from)

            # Corresponding additions
            if i < len(diff) and diff[i].startswith("+ "):
                md_lines.append("\nTo:",)
                block_to = ["```python"]
                while i < len(diff) and diff[i].startswith("+ "):
                    block_to.append(f"{line_num2:4d}: {diff[i][2:]}")
                    line_num2 += 1
                    i += 1
                block_to.append("```")
                md_lines.extend(block_to)
            else:
                md_lines.append("\nTo:")

            md_lines.append("\n---")

        else:
            # Keep track of line numbers when no diff prefix
            if line.startswith("  "):
                line_num1 += 1
                line_num2 += 1
            elif line.startswith("+ "):
                line_num2 += 1
            elif line.startswith("- "):
                line_num1 += 1
            i += 1

    Path(output_filename).write_text("\n".join(md_lines))
    print(f"Markdown diff with line numbers written to {output_filename}")


def _extract_functions(filepath):
    """Extract function definitions (with decorators) from a Python file."""
    text = Path(filepath).read_text()
    tree = ast.parse(text)
    lines = text.splitlines()

    functions = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            # include decorator lines
            decorator_start = (
                min([d.lineno for d in node.decorator_list]) - 1
                if node.decorator_list
                else node.lineno - 1
            )
            start = decorator_start
            end = getattr(node, "end_lineno", None)

            if end is None:
                # fallback: find next def/class or EOF
                next_defs = [
                    n.lineno for n in ast.walk(tree)
                    if hasattr(n, "lineno") and n.lineno > node.lineno
                    and isinstance(n, (ast.FunctionDef, ast.ClassDef))
                ]
                end = min(next_defs) - 1 if next_defs else len(lines)

            functions[node.name] = lines[start:end]
    return functions


def _extract_function_calls(filepath, internal_function_names):
    """
    Extract full multi-line calls for *internal* functions only.
    Keeps assignment or dict key context (e.g., '"key": func(...)')
    but ensures each call snippet is isolated and unique.
    """
    text = Path(filepath).read_text()
    lines = text.splitlines()
    tree = ast.parse(text)

    calls = {}

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        # Only plain name calls (ignore external libs, attributes, etc.)
        func_name = (
            node.func.id
            if isinstance(node.func, ast.Name)
            else None
        )
        if func_name not in internal_function_names:
            continue

        # Line span
        start = node.lineno - 1
        end = getattr(node, "end_lineno", node.lineno) - 1

        # Expand to include assignment or dict key context if on same line
        context_start = start
        for i in range(start - 1, -1, -1):
            line = lines[i].rstrip()
            if re.match(r'^\s*(return|with|for|if|else|elif|def|class)\b', line):
                break
            if line.strip().endswith('('):
                break
            if re.search(r'[:=]\s*$', line):
                context_start = i
            elif re.search(r'["\']\s*:\s*$', line):
                context_start = i
            else:
                # stop if previous line seems unrelated
                if line.strip() and not line.strip().endswith(','):
                    break

        # Extract full text for just this call
        snippet = "\n".join(lines[context_start:end + 1])
        snippet = textwrap.dedent(snippet).rstrip()

        # Remove trailing commas if they’re part of a return dict
        snippet = re.sub(r',\s*$', '', snippet)

        # Record
        calls.setdefault(func_name, []).append((node.lineno, node.end_lineno, snippet))

    # Deduplicate overlapping or identical snippets
    for func in calls:
        seen = set()
        unique = []
        for lineno, end_lineno, snip in calls[func]:
            key = snip.strip()
            if key not in seen:
                seen.add(key)
                unique.append((lineno, end_lineno, snip))
        calls[func] = unique

    return calls


def _normalize_call_text(snippet):
    """Normalize call text for comparison: dedent, strip trailing ws, collapse multiple blank lines."""
    # Dedent and strip trailing spaces on each line
    lines = [ln.rstrip() for ln in textwrap.dedent(snippet).splitlines()]
    # Remove leading/trailing blank lines
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines).strip()


def _format_snippet_with_lineno(snippet, lineno):
    """
    Format multi-line snippet so first line is prefixed with the lineno,
    subsequent lines are indented to line up visually.
    """
    lines = snippet.splitlines()
    if not lines:
        return f"{lineno:4d}: "
    out = []
    first = f"{lineno:4d}: {lines[0]}"
    out.append(first)
    indent = " " * 6  # aligns after "#### " and lineno
    for ln in lines[1:]:
        out.append(f"{indent}{ln}")
    return "\n".join(out)


def _diff_function_calls(md_lines, name, calls1, calls2):
    """
    calls1/calls2 are dicts name -> list of (start_lineno, end_lineno, snippet).
    This will append properly formatted From/To or Added/Removed sections to md_lines.
    """
    c1 = calls1.get(name, [])
    c2 = calls2.get(name, [])

    # Normalize snippet text for comparison
    norm1 = [(_normalize_call_text(snip), (start, end, snip)) for start, end, snip in c1]
    norm2 = [(_normalize_call_text(snip), (start, end, snip)) for start, end, snip in c2]

    set1 = [t for t, _ in norm1]
    set2 = [t for t, _ in norm2]

    # unchanged normalized texts
    unchanged = set(set1) & set(set2)

    # changed/unique entries (preserve full info)
    c1_changed = [info for txt, info in norm1 if txt not in unchanged]
    c2_changed = [info for txt, info in norm2 if txt not in unchanged]

    if not c1_changed and not c2_changed:
        return  # nothing to report

    md_lines.append(f"#### Function call changes for `{name}`")

    # If counts match, assume a From/To mapping by order (sorted by start lineno)
    if c1_changed and c2_changed and len(c1_changed) == len(c2_changed):
        # sort both lists by start lineno for stable pairing
        c1_sorted = sorted(c1_changed, key=lambda t: t[0])
        c2_sorted = sorted(c2_changed, key=lambda t: t[0])
        for (ln1, end1, s1), (ln2, end2, s2) in zip(c1_sorted, c2_sorted):
            if _normalize_call_text(s1) != _normalize_call_text(s2):
                md_lines.append("\nFrom:\n```python")
                md_lines.append(_format_snippet_with_lineno(s1, ln1))
                md_lines.append("```\nTo:\n```python")
                md_lines.append(_format_snippet_with_lineno(s2, ln2))
                md_lines.append("```")
    else:
        # Added calls (in file2 but not file1)
        if c2_changed:
            md_lines.append("\n**Added calls:**")
            md_lines.append("```python")
            for ln, end, s in sorted(c2_changed, key=lambda t: t[0]):
                md_lines.append(_format_snippet_with_lineno(s, ln))
            md_lines.append("```")
        # Removed calls (in file1 but not file2)
        if c1_changed:
            md_lines.append("\n**Removed calls:**")
            md_lines.append("```python")
            for ln, end, s in sorted(c1_changed, key=lambda t: t[0]):
                md_lines.append(_format_snippet_with_lineno(s, ln))
            md_lines.append("```")

    md_lines.append("")  # spacing


def generate_function_split_markdown_diffs(file1, file2, output_filename="function_diff.md"):
    """Main entry: compare function defs + internal calls."""
    funcs1 = _extract_functions(file1)
    funcs2 = _extract_functions(file2)

    calls1 = _extract_function_calls(file1, set(funcs1.keys()))
    calls2 = _extract_function_calls(file2, set(funcs2.keys()))

    md_lines = [f"# Function Diff `{file1}` → `{file2}`", ""]

    all_funcs = sorted(set(funcs1.keys()) | set(funcs2.keys()))

    for name in all_funcs:
        f1_lines = funcs1.get(name)
        f2_lines = funcs2.get(name)

        if f1_lines is None and f2_lines is not None:
            md_lines.append(f"### Added function `{name}`\n")
            md_lines.append("```python")
            md_lines.extend(f2_lines)
            md_lines.append("```\n")
        elif f1_lines is not None and f2_lines is None:
            md_lines.append(f"### Removed function `{name}`\n")
            md_lines.append("```python")
            md_lines.extend(f1_lines)
            md_lines.append("```\n")
        elif f1_lines and f2_lines and f1_lines != f2_lines:
            md_lines.append(f"### Changed function `{name}`\n")
            md_lines.append("From:\n```python")
            md_lines.extend(f1_lines)
            md_lines.append("```\nTo:\n```python")
            md_lines.extend(f2_lines)
            md_lines.append("```\n")

        # Internal call differences
        _diff_function_calls(md_lines, name, calls1, calls2)

    Path(output_filename).write_text("\n".join(md_lines))
    print(f"Function-level Markdown diff (internal only) written to {output_filename}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    # Required positionals
    parser.add_argument("filename_1")
    parser.add_argument("filename_2")

    # Optionals
    parser.add_argument("-f1", "--filename_1_alt", help="override first")
    parser.add_argument("-f2", "--filename_2_alt", help="override second")
    parser.add_argument("-o", "--output_filename", help="name of output file")
    parser.add_argument("-t", "--type", help="function or standard breakdown")

    args = parser.parse_args()

    filename_1 = args.filename_1_alt or args.filename_1
    filename_2 = args.filename_2_alt or args.filename_2
    output_filename = args.output_filename or "diff.md"
    type_diff = args.type or "function"

    if type_diff.lower() == "function":
        generate_function_split_markdown_diffs(filename_1, filename_2, output_filename=output_filename)
    else:
        generate_markdown_diff(filename_1, filename_2, output_filename=output_filename)
