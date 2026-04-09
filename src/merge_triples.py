import json
from pathlib import Path

def merge_triples(input_files, output_file="triples.json"):
    all_triples = []

    for path in input_files:
        p = Path(path)
        if not p.exists():
            print(f"Skipping missing file: {p}")
            continue
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        triples = data.get("triples", data)
        if not isinstance(triples, list):
            print(f"File {p} does not contain a list under 'triples'; skipping.")
            continue
        all_triples.extend(triples)
        print(f"{p}: added {len(triples)} triples")

    print(f"Total merged triples: {len(all_triples)}")
    out = Path(output_file)
    with out.open("w", encoding="utf-8") as f:
        json.dump({"triples": all_triples}, f, ensure_ascii=False, indent=2)
    print(f"Saved merged triples to {out.resolve()}")

if __name__ == "__main__":
    # Edit this list with the files you exported from NotebookLM
    files = [
        "triples_1.json",
        "triples_2.json",
        "triples_3.json",
    ]
    merge_triples(files)
