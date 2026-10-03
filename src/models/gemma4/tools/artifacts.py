"""Record exact Gemma GGUF identities without loading tensor payloads into RAM."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import mmap
from pathlib import Path
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from tools.gufo.gguf import Reader, GGML_TYPE_NAME, tensor_bytes


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def inspect(path):
    with path.open("rb") as source, mmap.mmap(source.fileno(), 0,
                                              access=mmap.ACCESS_READ) as data:
        reader = Reader(data)
        if data[:4] != b"GGUF":
            raise ValueError(f"not GGUF: {path}")
        reader.pos = 4
        version, count, metadata_count = reader.u32(), reader.u64(), reader.u64()
        metadata = {}
        for _ in range(metadata_count):
            key = reader.string()
            metadata[key] = reader.read_value(reader.u32())
        tensors = []
        for _ in range(count):
            name = reader.string()
            shape = [reader.u64() for _ in range(reader.u32())]
            dtype, offset = reader.u32(), reader.u64()
            tensor = dict(name=name, shape=shape, type=dtype,
                          storage=GGML_TYPE_NAME.get(dtype, str(dtype)), offset=offset)
            tensor["bytes"] = tensor_bytes({}, tensor)
            tensors.append(tensor)
    tokenizer = {k: v for k, v in metadata.items() if k.startswith("tokenizer.")}
    template = tokenizer.get("tokenizer.chat_template", "")
    with path.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    return dict(
        file=path.relative_to(ROOT).as_posix(), bytes=path.stat().st_size,
        sha256=digest,
        gguf_version=version,
        metadata={k: v for k, v in metadata.items() if not k.startswith("tokenizer.")},
        tokenizer_sha256=hashlib.sha256(canonical(tokenizer)).hexdigest(),
        template_sha256=hashlib.sha256(template.encode()).hexdigest(),
        tokenizer_contract={k: v for k, v in tokenizer.items()
                            if not isinstance(v, list) and k != "tokenizer.chat_template"},
        storage_counts=dict(Counter(t["storage"] for t in tensors)), tensors=tensors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folder", type=Path, action="append", required=True)
    args = parser.parse_args()
    files = []
    for folder in args.folder:
        for path in sorted(folder.glob("*.gguf")):
            print(f"Hashing {path}", flush=True)
            files.append(inspect(path.resolve()))
    provenance = []
    for repository in ("unsloth/gemma-4-31B-it-GGUF", "unsloth/gemma-4-31B-it-qat-GGUF"):
        with urllib.request.urlopen(f"https://huggingface.co/api/models/{repository}?blobs=true") as response:
            info = json.load(response)
        matches = []
        for item in info.get("siblings", []):
            lfs = item.get("lfs", {})
            for file in files:
                if file["sha256"] == lfs.get("sha256"):
                    matches.append(dict(local=file["file"], remote=item["rfilename"], sha256=file["sha256"]))
        provenance.append(dict(repository=repository, revision=info["sha"], hash_matches=matches))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(files=files, provenance=provenance),
                                      indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Recorded {len(files)} files in {args.output}")


if __name__ == "__main__":
    main()
