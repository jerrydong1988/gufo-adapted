"""Generate tokenizer/template goldens with the pinned independent reference."""

import argparse
import json
import mmap
from pathlib import Path
import subprocess
import sys

from jinja2 import Environment

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from tools.gufo.gguf import Reader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    args = parser.parse_args()
    args.scratch.mkdir(parents=True, exist_ok=True)
    with args.model.open("rb") as source, mmap.mmap(source.fileno(), 0,
                                                   access=mmap.ACCESS_READ) as data:
        reader = Reader(data)
        reader.pos = 16
        count = reader.u64()
        template = None
        for _ in range(count):
            key = reader.string()
            if key == "tokenizer.chat_template":
                template = reader.read_value(reader.u32())
            else:
                reader.skip_value(reader.u32())
    if template is None:
        raise ValueError("missing chat template")
    def tokenize(text, add_special):
        text_file, token_file = args.scratch / "input.txt", args.scratch / "tokens.txt"
        text_file.write_bytes(text.encode("utf-8"))
        with (args.scratch / "tokenize.log").open("ab") as log:
            subprocess.run([str(args.reference.resolve()), "--tokenize", str(args.model.resolve()),
                            str(text_file.resolve()), str(token_file.resolve()), str(int(add_special))],
                           stdout=log, stderr=log, check=True)
        return [int(item) for item in token_file.read_text().split()]
    texts = ["", "Hello world!", " hello  world ", "a\nb\n\n\nend", "\n" * 40,
             "1234567890 3.14159", "naïve café e\u0301", "中文 日本語 한국어",
             "🐈 🦉 🚀", "\t x\r\n y\u00a0z", "<bos><|turn>user\nHi<turn|>",
             '<|channel>thought\nreason<channel|>answer',
             'def f(x):\n    return x ** 2\n', 'foo\x00bar',
             'A' * 200, ' spaces ' * 20]
    tokenizer = [dict(text=text, add_special=add, tokens=tokenize(text, add))
                 for text in texts for add in (False, True)]
    environment = Environment()
    def reject(message):
        raise ValueError(message)
    environment.globals["raise_exception"] = reject
    compiled = environment.from_string(template)
    conversations = [
        [dict(role="user", content=" Say hello. ")],
        [dict(role="system", content=" Be concise. "), dict(role="user", content="2 + 2?")],
        [dict(role="developer", content="Answer in French."), dict(role="user", content="Hello")],
        [dict(role="user", content="First"), dict(role="assistant", content="One"), dict(role="user", content="Next")],
        [dict(role="user", content="First"), dict(role="assistant", content=" One "), dict(role="assistant", content="Two"), dict(role="user", content="Next")],
        [dict(role="user", content="First"), dict(role="assistant", content="<|channel>thought\nsecret<channel|>visible"), dict(role="user", content="Next")],
        [dict(role="user", content="\u2003\u00a0Hello\u2029")],
        [dict(role="user", content="Hi"), dict(role="developer", content="A later instruction"), dict(role="user", content="Next")],
        [dict(role="assistant", content="Answer", reasoning="Visible thought")],
        [dict(role="user", content="First"), dict(role="assistant", content="Answer", reasoning_content="Prior hidden thought"), dict(role="user", content="Next")],
        [dict(role="user", content="First"), dict(role="assistant", content=" Answer ", reasoning_content="Current thought")],
        [dict(role="user", content=[dict(type="text",text=" Before "),dict(type="image"),dict(type="text",text=" After ")])],
        [dict(role="user", content=[dict(type="image"),dict(type="image_url"),dict(type="text",text=" Compare ")])],
        [dict(role="user", content=[dict(type="image"),dict(type="text",text=" First ")]),dict(role="assistant",content="A shape"),dict(role="user",content="Remember the shape")],
        [dict(role="user",content="Inspect"),dict(role="assistant",content="",tool_calls=[dict(id="call_1",function=dict(name="inspect",arguments=dict(z=dict(nested=True),a="shape")))]),dict(role="tool",tool_call_id="call_1",content="Done")],
        [dict(role="user",content="Inspect"),dict(role="assistant",content="",tool_calls=[dict(id="call_1",function=dict(name="inspect",arguments=dict(path="image.png")))]),dict(role="tool",tool_call_id="call_1",content=[dict(type="text",text="Pixels"),dict(type="image"),dict(type="text",text="loaded"),dict(type="image_url")])],
        [dict(role="user",content="Inspect"),dict(role="assistant",content="",tool_calls=[dict(id="call_1",function=dict(name="inspect",arguments=dict()))]),dict(role="tool",tool_call_id="call_1",content="Done"),dict(role="assistant",content=" I see it "),dict(role="user",content="Continue")],
    ]
    chats = []
    for messages in conversations:
        for enabled in (False, True):
            rendered = compiled.render(messages=messages, tools=[], bos_token="<bos>",
                                       add_generation_prompt=True, enable_thinking=enabled)
            chats.append(dict(messages=messages, enabled=enabled, rendered=rendered,
                              tokens=tokenize(rendered, False)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(tokenizer=tokenizer, chat=chats),
                                     ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Recorded {len(tokenizer)} tokenizer and {len(chats)} template cases")


if __name__ == "__main__":
    main()
