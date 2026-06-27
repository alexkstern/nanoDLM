"""Download a corpus and tokenize it at the character level.

Two datasets:
  shakespeare (default): ~1 MB of tiny shakespeare from karpathy/char-rnn.
                         The canonical educational corpus.
  tinystories:           a capped slice of roneneldan/TinyStories from
                         HuggingFace. Purpose-built for tiny LMs to produce
                         coherent narratives — char-level Shakespeare is too
                         small and too archaic for the model to produce
                         readable modern English. Default cap is 50 MB
                         streamed from the start of the file, which gives
                         ~50K short stories. Pass --max-bytes to change.

Note on BPE: char-level is the default. `--tokenizer gpt2` switches to GPT-2
BPE (~50K vocab), which inflates the embedding table to ~19M params (~30M
total) but gives a genuine Zipfian *subword* vocabulary — the right granularity
for testing a token-frequency masking schedule against the "content-word"
story. We use it for a single robustness run on top of the char-level sweep.

Output:
  data/train.bin, data/val.bin   (uint16 token ids, 90/10 split)
  data/meta.pkl                  (vocab_size, stoi, itos, dataset name)

The mask token is NOT in the vocab — it's appended by the model.
"""
import argparse
import os
import pickle
import urllib.request

import numpy as np


SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/"
    "data/tinyshakespeare/input.txt"
)
TINYSTORIES_URL = (
    "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/"
    "TinyStoriesV2-GPT4-train.txt"
)
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")


def download_shakespeare(path: str) -> None:
    print(f"downloading tiny shakespeare -> {path}")
    urllib.request.urlretrieve(SHAKESPEARE_URL, path)


def download_tinystories(path: str, max_bytes: int) -> None:
    """Stream the first `max_bytes` of TinyStoriesV2-GPT4-train.txt."""
    print(f"streaming TinyStories (first {max_bytes // 1024 // 1024} MB) -> {path}")
    req = urllib.request.Request(
        TINYSTORIES_URL,
        headers={"Range": f"bytes=0-{max_bytes - 1}"},
    )
    written = 0
    with urllib.request.urlopen(req) as response, open(path, "wb") as f:
        while written < max_bytes:
            chunk = response.read(min(1024 * 1024, max_bytes - written))
            if not chunk:
                break
            f.write(chunk)
            written += len(chunk)
            if written % (10 * 1024 * 1024) == 0:
                print(f"  ...{written // 1024 // 1024} MB")
    # Trim back to the last complete story to avoid mid-story truncation
    with open(path, "rb") as f:
        text = f.read().decode("utf-8", errors="ignore")
    # TinyStories separates stories with the literal "<|endoftext|>" marker;
    # drop everything after the last one we have.
    last = text.rfind("<|endoftext|>")
    if last > 0:
        text = text[: last + len("<|endoftext|>")]
    # Strip the EOT markers entirely — char-level vocab can't include them as
    # a single token, and the model doesn't need them since each batch is a
    # random slice anyway.
    text = text.replace("<|endoftext|>", "\n")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"  wrote {len(text):,} chars")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="shakespeare",
                   choices=["shakespeare", "tinystories"])
    p.add_argument("--max-bytes", type=int, default=50 * 1024 * 1024,
                   help="byte cap for streaming TinyStories (default 50 MB)")
    p.add_argument("--tokenizer", default="char", choices=["char", "gpt2"],
                   help="char (default) or gpt2 BPE (needs tiktoken). BPE gives "
                        "a real Zipfian subword vocab for the frequency schedule "
                        "at the cost of a ~50K-row embedding table.")
    p.add_argument("--data-dir", default=DATA,
                   help="output dir for train.bin/val.bin/meta.pkl (default data/). "
                        "Use distinct dirs to keep char and BPE corpora side by side.")
    args = p.parse_args()

    # Resolve relative --data-dir against the repo root (HERE), matching how
    # train.py/eval.py/sample.py resolve cfg.data_dir via __file__, so a
    # relative name like "data_char" points to the same place from any CWD.
    data_out = args.data_dir if os.path.isabs(args.data_dir) else os.path.join(HERE, args.data_dir)
    os.makedirs(data_out, exist_ok=True)
    # Dataset-specific cache filename. Using a single "input.txt" for both
    # datasets means `prepare.py` then `prepare.py --dataset tinystories`
    # silently reuses the Shakespeare file — a real bug found in clean-room
    # testing. Separate names let each dataset cache independently.
    input_path = os.path.join(data_out, f"input_{args.dataset}.txt")

    if not os.path.exists(input_path) or os.path.getsize(input_path) < 1000:
        if args.dataset == "shakespeare":
            download_shakespeare(input_path)
        else:
            download_tinystories(input_path, args.max_bytes)
    else:
        print(f"reusing existing {input_path} ({os.path.getsize(input_path):,} bytes)")

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    print(f"corpus length: {len(text):,} chars")

    if args.tokenizer == "char":
        chars = sorted(set(text))
        vocab_size = len(chars)
        stoi = {c: i for i, c in enumerate(chars)}
        itos = {i: c for i, c in enumerate(chars)}
        print(f"vocab size: {vocab_size}")
        if vocab_size > 256:
            # uint16 still fits up to 65535; flag anyway because it's unusual.
            print(f"warning: vocab_size={vocab_size} is large for char-level "
                  f"(unicode in the corpus?). Stored as uint16.")
        ids = np.array([stoi[c] for c in text], dtype=np.uint16)
    else:  # gpt2 BPE
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        vocab_size = enc.n_vocab                       # 50257
        assert vocab_size < 2 ** 16, "vocab exceeds uint16"
        print(f"tokenizing {len(text):,} chars with gpt2 BPE...")
        ids = np.array(enc.encode_ordinary(text), dtype=np.uint16)
        # itos maps id -> decoded text piece (bytes -> utf-8, replacing partial
        # multibyte tokens) so sample.py can still print readable strings.
        itos = {i: enc.decode_single_token_bytes(i).decode("utf-8", errors="replace")
                for i in range(vocab_size)}
        stoi = None                                    # not needed at train time
        print(f"vocab size: {vocab_size} (gpt2 BPE)   "
              f"compression: {len(text) / len(ids):.2f} chars/token")

    n = len(ids)
    train_ids = ids[: int(n * 0.9)]
    val_ids = ids[int(n * 0.9):]
    train_ids.tofile(os.path.join(data_out, "train.bin"))
    val_ids.tofile(os.path.join(data_out, "val.bin"))
    with open(os.path.join(data_out, "meta.pkl"), "wb") as f:
        pickle.dump({"vocab_size": vocab_size, "stoi": stoi, "itos": itos,
                     "dataset": args.dataset, "tokenizer": args.tokenizer}, f)
    print(f"train: {len(train_ids):,} tokens   val: {len(val_ids):,} tokens")


if __name__ == "__main__":
    main()
