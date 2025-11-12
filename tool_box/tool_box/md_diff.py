import ast
import difflib
from pathlib import Path
import textwrap

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
    """Extract multi-line calls, filtered to internal functions only."""
    text = Path(filepath).read_text()
    lines = text.splitlines()
    tree = ast.parse(text)

    calls = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        func_name = None
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            # skip obj.method() or imported.module.func
            continue

        if not func_name or func_name not in internal_function_names:
            continue

        start = node.lineno - 1
        end = getattr(node, "end_lineno", None)
        if end is None:
            # fallback: read until parentheses balance
            open_parens = 0
            end = start
            for i in range(start, len(lines)):
                open_parens += lines[i].count("(")
                open_parens -= lines[i].count(")")
                if open_parens <= 0 and i > start:
                    end = i
                    break

        snippet = "\n".join(lines[start:end + 1]).rstrip()
        snippet = textwrap.dedent(snippet)
        calls.setdefault(func_name, []).append((node.lineno, snippet))

    return calls


def _normalize_call_text(snippet):
    """Normalize call text for comparison."""
    return "\n".join(line.rstrip() for line in textwrap.dedent(snippet).splitlines()).strip()


def _diff_function_calls(md_lines, name, calls1, calls2):
    """Add Markdown lines showing internal function call changes."""
    c1 = calls1.get(name, [])
    c2 = calls2.get(name, [])

    norm1 = [(_normalize_call_text(s), ln) for ln, s in c1]
    norm2 = [(_normalize_call_text(s), ln) for ln, s in c2]

    unchanged = set(sn for sn, _ in norm1) & set(sn for sn, _ in norm2)
    c1_changed = [(ln, s) for (ln, s), (ntext, _) in zip(c1, norm1) if ntext not in unchanged]
    c2_changed = [(ln, s) for (ln, s), (ntext, _) in zip(c2, norm2) if ntext not in unchanged]

    if not c1_changed and not c2_changed:
        return

    md_lines.append(f"#### Function call changes for `{name}`")

    if c1_changed and c2_changed and len(c1_changed) == len(c2_changed):
        for (ln1, s1), (ln2, s2) in zip(sorted(c1_changed), sorted(c2_changed)):
            if _normalize_call_text(s1) != _normalize_call_text(s2):
                md_lines.append(
                    f"\nFrom:\n```python\n{ln1:4d}: {s1}\n```\nTo:\n```python\n{ln2:4d}: {s2}\n```"
                )
    else:
        if c2_changed:
            md_lines.append("\n**Added calls:**")
            md_lines.append("```python")
            for ln, s in sorted(c2_changed):
                md_lines.append(f"{ln:4d}: {s}")
            md_lines.append("```")
        if c1_changed:
            md_lines.append("\n**Removed calls:**")
            md_lines.append("```python")
            for ln, s in sorted(c1_changed):
                md_lines.append(f"{ln:4d}: {s}")
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
