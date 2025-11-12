import ast
import difflib
from pathlib import Path

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
    """Extract function names, including decorators, and their source code from a Python file."""
    text = Path(filepath).read_text()
    tree = ast.parse(text)
    lines = text.splitlines()

    functions = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            # Include decorator lines above the function
            decorator_start = (
                min([d.lineno for d in node.decorator_list]) - 1
                if node.decorator_list
                else node.lineno - 1
            )
            start = decorator_start

            # Find end line (use end_lineno when available, else infer)
            end = getattr(node, "end_lineno", None)
            if end is None:
                # Fallback: scan forward until next function/class or EOF
                next_defs = [
                    n.lineno for n in ast.walk(tree)
                    if hasattr(n, "lineno") and n.lineno > node.lineno
                    and isinstance(n, (ast.FunctionDef, ast.ClassDef))
                ]
                end = min(next_defs) - 1 if next_defs else len(lines)

            functions[node.name] = lines[start:end]

    return functions


def generate_function_split_markdown_diffs(file1, file2, output_filename="function_diff.md"):
    funcs1 = _extract_functions(file1)
    funcs2 = _extract_functions(file2)

    md_lines = [f"# Function Diff `{file1}` → `{file2}`", ""]

    # All function names across both files
    all_funcs = sorted(set(funcs1.keys()) | set(funcs2.keys()))

    for name in all_funcs:
        f1_lines = funcs1.get(name)
        f2_lines = funcs2.get(name)

        # Added / Removed
        if f1_lines is None:
            md_lines.append(f"### Added function `{name}`\n")
            md_lines.append("```python")
            md_lines.extend(f2_lines)
            md_lines.append("```\n")
            continue
        elif f2_lines is None:
            md_lines.append(f"### Removed function `{name}`\n")
            md_lines.append("```python")
            md_lines.extend(f1_lines)
            md_lines.append("```\n")
            continue

        # Compare contents
        if f1_lines != f2_lines:
            diff = list(difflib.unified_diff(f1_lines, f2_lines, lineterm=""))
            if diff:
                md_lines.append(f"### Changed function `{name}`\n")
                md_lines.append("From:\n```python")
                md_lines.extend(f1_lines)
                md_lines.append("```\nTo:\n```python")
                md_lines.extend(f2_lines)
                md_lines.append("```\n")

    Path(output_filename).write_text("\n".join(md_lines))
    print(f"Function-level Markdown diff written to {output_filename}")


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
