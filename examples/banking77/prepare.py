"""Build the banking77 proof-run files in OpenAI fine-tuning format.

banking77 (PolyAI, CC-BY-4.0): real online-banking customer messages, each labelled
with one of 77 intents. The kind of task teams fine-tuned gpt-4o-mini for.

Writes, deterministically:
  data/train.jsonl        40 messages per intent (3,080) in OpenAI chat format
  data/validation.jsonl    5 per intent (385), also from the train split
  data/golden.jsonl       10 per intent (770) from the test split, in `tunekeep record`
                          format. The "recorded" answer is the human label: here it stands
                          in for a fine-tune that answers every test message correctly, so
                          "score vs recorded" in the report is plain accuracy.

Standard library only. Source: PolyAI's original CSVs, pinned by commit and SHA-256.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import random
import urllib.request
from pathlib import Path

SOURCE = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/57ec275d8078af65b7731c2a98be812d844a6d6b/banking_data"
SHA256 = {
    "train.csv": "b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b",
    "test.csv": "d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d",
}
OUT = Path(__file__).parent / "data"


def rows(name: str) -> list[tuple[str, str]]:
    """(text, intent) pairs from one CSV, after checking its SHA-256."""
    with urllib.request.urlopen(f"{SOURCE}/{name}", timeout=60) as r:
        raw = r.read()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != SHA256[name]:
        raise SystemExit(f"{name}: unexpected SHA-256 {digest}")
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8")))
    return [(row["text"].strip(), row["category"].strip()) for row in reader]


def system_prompt(labels: list[str]) -> str:
    return ("You classify messages sent to an online bank's support chat. Choose exactly one intent from this "
            "list and reply with JSON only, like {\"intent\": \"card_arrival\"}.\nIntents: " + ", ".join(labels))


def main() -> None:
    train_rows, test_rows = rows("train.csv"), rows("test.csv")
    labels = sorted({lab for _, lab in train_rows})
    assert len(labels) == 77, len(labels)
    system = system_prompt(labels)
    rng = random.Random(77)

    by_label: dict[str, list[str]] = {}
    for text, lab in train_rows:
        by_label.setdefault(lab, []).append(text)
    train, val = [], []
    for lab, texts in sorted(by_label.items()):
        texts = sorted(set(texts))
        rng.shuffle(texts)
        answer = json.dumps({"intent": lab})
        for t in texts[:40]:
            train.append({"messages": [{"role": "system", "content": system}, {"role": "user", "content": t},
                                       {"role": "assistant", "content": answer}]})
        for t in texts[40:45]:
            val.append({"messages": [{"role": "system", "content": system}, {"role": "user", "content": t},
                                     {"role": "assistant", "content": answer}]})
    rng.shuffle(train)

    test: dict[str, list[str]] = {}
    for text, lab in test_rows:
        test.setdefault(lab, []).append(text)
    golden, line = [], 0
    for lab, texts in sorted(test.items()):
        texts = sorted(set(texts))
        rng.shuffle(texts)
        for t in texts[:10]:
            line += 1
            msgs = [{"role": "system", "content": system}, {"role": "user", "content": t}]
            answer = {"role": "assistant", "content": json.dumps({"intent": lab})}
            golden.append({
                "id": hashlib.sha256(json.dumps(msgs, sort_keys=True).encode()).hexdigest()[:16],
                "line": line,
                "model": "human labels (banking77 test split)",
                "request": {"messages": msgs, "temperature": 0.0},
                "response": {"message": answer, "finish_reason": "stop"},
                "reference": None,  # the label is already the recorded answer
            })

    OUT.mkdir(exist_ok=True)
    for name, data in (("train.jsonl", train), ("validation.jsonl", val), ("golden.jsonl", golden)):
        (OUT / name).write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in data))
    print(f"train {len(train)}, validation {len(val)}, golden {len(golden)} -> {OUT}")


if __name__ == "__main__":
    main()
