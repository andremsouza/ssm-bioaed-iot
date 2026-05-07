#!/usr/bin/env python3
"""
Render mermaid diagrams in a Markdown file using mmdc and export to PDF.

Usage:
  python3 scripts/md2pdf_with_mermaid.py reports/pivot_d_report.md out.pdf

Prereqs: mmdc (@mermaid-js/mermaid-cli), pandoc, wkhtmltopdf
"""

from __future__ import annotations
import re
import sys
import shutil
import subprocess
import tempfile
from pathlib import Path


def which(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def ensure_cmd(cmd: str, hint: str) -> None:
    if not which(cmd):
        print(f"Missing required command: {cmd}\nInstall hint: {hint}")
        sys.exit(1)


def render_diagrams(md_text: str, tmpdir: str) -> str:
    pattern = re.compile(r"```mermaid\s*\n(.*?)\n```", re.DOTALL)
    idx = 0
    mmdc_present = which("mmdc")

    def repl(m):
        nonlocal idx
        idx += 1
        code = m.group(1)
        # If mmdc is available, try to render to a PNG image
        if mmdc_present:
            mmd = Path(tmpdir) / f"diagram{idx}.mmd"
            out_png = Path(tmpdir) / f"diagram{idx}.png"
            mmd.write_text(code, encoding="utf-8")
            print(f"Rendering diagram {idx} -> {out_png.name}")
            res = subprocess.run(["mmdc", "-i", str(mmd), "-o", str(out_png)])
            if res.returncode == 0 and out_png.exists():
                return f"![mermaid diagram {idx}](file://{out_png.resolve()})"
            else:
                print("mmdc failed or Chrome not found — falling back to inline Mermaid in HTML.")
        # Fallback: keep the diagram as an inline Mermaid block for client-side rendering
        return f'<div class="mermaid">\n{code}\n</div>'

    new_text = pattern.sub(repl, md_text)
    return new_text


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: md2pdf_with_mermaid.py input.md [output.pdf]")
        sys.exit(1)
    input_md = Path(sys.argv[1])
    if not input_md.exists():
        print(f"Input not found: {input_md}")
        sys.exit(1)
    output_pdf = Path(sys.argv[2]) if len(sys.argv) > 2 else input_md.with_suffix(".pdf")

    ensure_cmd("mmdc", "npm i -g @mermaid-js/mermaid-cli")
    pandoc_present = which("pandoc")
    wkhtml_present = which("wkhtmltopdf")

    if not pandoc_present:
        print("Warning: `pandoc` not found. Will try Python `markdown` fallback if available.")
    if not wkhtml_present:
        print("Info: `wkhtmltopdf` not found — will produce HTML instead of PDF.")

    md_text = input_md.read_text(encoding="utf-8")
    with tempfile.TemporaryDirectory() as tmp:
        converted_md = Path(tmp) / input_md.name
        new_text = render_diagrams(md_text, tmp)
        converted_md.write_text(new_text, encoding="utf-8")

        html_file = Path(tmp) / (input_md.stem + ".html")
        # Convert Markdown -> HTML using pandoc if available, otherwise try python-markdown
        if pandoc_present:
            print("Converting Markdown -> HTML (pandoc)...")
            subprocess.run(["pandoc", "-s", str(converted_md), "-o", str(html_file)], check=True)
        else:
            try:
                import markdown as pymd
            except Exception:
                print("Error: Neither `pandoc` nor Python package `markdown` is available.")
                print("Install pandoc (apt/brew) or `pip install markdown` and retry.")
                sys.exit(1)
            print("Converting Markdown -> HTML (python-markdown)...")
            text = converted_md.read_text(encoding="utf-8")
            html = pymd.markdown(text, extensions=["fenced_code", "tables"])
            html_file.write_text(html, encoding="utf-8")

        # Inject client-side Mermaid runtime so inline <div class="mermaid"> blocks render in browser
        MERMAID_SNIPPET = """
<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
<script>document.addEventListener('DOMContentLoaded', function(){ if(window.mermaid){mermaid.initialize({ startOnLoad:true }); } });</script>
"""
        html = html_file.read_text(encoding="utf-8")
        if re.search(r"</head>", html, flags=re.IGNORECASE):
            html = re.sub(r"(?i)</head>", MERMAID_SNIPPET + "\n</head>", html, count=1)
        else:
            html = MERMAID_SNIPPET + "\n" + html
        html_file.write_text(html, encoding="utf-8")

        # If the requested output path ends with .html, save the generated HTML and exit
        if str(output_pdf).lower().endswith(".html"):
            output_html = output_pdf
            output_html.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(str(html_file), str(output_html))
            print(f"Saved HTML: {output_html}")
            return

        if wkhtml_present:
            print("Converting HTML -> PDF (wkhtmltopdf)...")
            subprocess.run(
                ["wkhtmltopdf", "--enable-local-file-access", str(html_file), str(output_pdf)],
                check=True,
            )
            print(f"Saved PDF: {output_pdf}")
        else:
            # Save the HTML fallback next to the requested output path (use .html extension)
            if output_pdf.suffix.lower() == ".pdf":
                output_html = output_pdf.with_suffix(".html")
            else:
                output_html = output_pdf
            output_html.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(str(html_file), str(output_html))
            print(f"wkhtmltopdf not available — saved HTML: {output_html}")


if __name__ == "__main__":
    main()
