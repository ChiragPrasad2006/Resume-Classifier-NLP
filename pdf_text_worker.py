"""Lightweight PDF text-extraction worker used by further_train_longformer.py."""

import argparse
import json
import sys

from pypdf import PdfReader


def extract_text(pdf_path: str):
    try:
        reader = PdfReader(pdf_path)
        text = "\n".join(page.extract_text() or "" for page in reader.pages).strip()
        return text or None
    except Exception as exc:
        print(f"Could not read {pdf_path}: {exc}", file=sys.stderr)
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as input_file, open(
        args.output, "w", encoding="utf-8"
    ) as output_file:
        for line in input_file:
            task = json.loads(line)
            json.dump(
                {"index": task["index"], "text": extract_text(task["path"])},
                output_file,
                ensure_ascii=False,
            )
            output_file.write("\n")


if __name__ == "__main__":
    main()
