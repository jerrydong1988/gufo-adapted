"""Real-model buffered/streaming Gemma HTTP qualification on an owned server."""
import argparse
import base64
import json
import io
from pathlib import Path
import subprocess
import time
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18180)
    parser.add_argument("--mmproj", type=Path)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--mtp-model", type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    base = f"http://127.0.0.1:{args.port}"
    def call(path, body=None):
        request = urllib.request.Request(base + path, data=None if body is None else json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        return urllib.request.urlopen(request, timeout=180)
    report = {"passed": False}
    with args.output.with_suffix(".server.log").open("w", encoding="utf-8") as log:
        command = [str(args.engine.resolve()), "serve", "llm", "--host", "127.0.0.1", "--port", str(args.port), "--model", str(args.model.resolve()), "--context", "4096", "--think", "off", "--speculative", "mtp" if args.mtp_model else "off"]
        if args.mtp_model: command += ["--mtp-model", str(args.mtp_model.resolve())]
        if args.mmproj: command += ["--mmproj", str(args.mmproj.resolve())]
        process = subprocess.Popen(command, stdout=log, stderr=log, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        try:
            deadline = time.monotonic() + 120
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"owned server exited: {process.returncode}")
                try:
                    with call("/v1/models") as response: report["models"] = json.load(response)
                    break
                except (OSError, urllib.error.URLError):
                    if time.monotonic() > deadline: raise
                    time.sleep(0.25)
            with call("/props") as response: report["props"] = json.load(response)
            modalities = report["models"]["data"][0]["architecture"]["input_modalities"]
            assert ("image" in modalities) == bool(args.mmproj)
            assert report["props"]["modalities"]["vision"] == bool(args.mmproj)
            for enabled in (False, True):
                request = {"model": report["models"]["data"][0]["id"], "messages": [{"role": "user", "content": "What is 2 + 2? Answer briefly."}], "max_tokens": 24, "temperature": 0, "chat_template_kwargs": {"enable_thinking": enabled}}
                with call("/v1/chat/completions", request) as response: buffered = json.load(response)
                request["stream"] = True
                content, reasoning, usage = "", "", None
                request["stream_options"] = {"include_usage": True}
                with call("/v1/chat/completions", request) as response:
                    for line in response:
                        if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]": continue
                        item = json.loads(line[6:])
                        if item.get("usage"): usage = item["usage"]
                        for choice in item.get("choices", []):
                            delta = choice.get("delta", {})
                            content += delta.get("content") or ""
                            reasoning += delta.get("reasoning_content") or ""
                message = buffered["choices"][0]["message"]
                assert content == (message.get("content") or ""), (content, message)
                assert reasoning == (message.get("reasoning_content") or ""), (reasoning, message)
                assert usage and usage["completion_tokens"] == buffered["usage"]["completion_tokens"]
                assert "<channel|>" not in content and "<|channel>" not in content
                report[f"thinking_{enabled}"] = {"buffered": buffered, "stream_content": content, "stream_reasoning": reasoning, "stream_usage": usage}
            with call("/v1/completions", {"prompt": "The capital of France is", "max_tokens": 8, "temperature": 0}) as response: report["raw"] = json.load(response)
            rejected = {"model": report["models"]["data"][0]["id"], "messages": [{"role": "user", "content": "Use a tool"}], "tools": [{"type": "function", "function": {"name": "test", "parameters": {"type": "object"}}}]}
            try:
                call("/v1/chat/completions", rejected).close()
                raise AssertionError("unsupported tools accepted")
            except urllib.error.HTTPError as error:
                assert 400 <= error.code < 500
                body = error.read().decode()
                assert "Gemma 4 tool requests are not supported" in body, body
                report["unsupported_tools"] = {"status": error.code, "body": body}
            if args.image:
                url = "data:image/png;base64," + base64.b64encode(args.image.read_bytes()).decode()
                image = {"type": "image_url", "image_url": {"url": url}}
                question = "Name the two shapes and their colors. Be brief."
                messages = [{"role": "user", "content": [image, {"type": "text", "text": question}]}]
                body = {"model": report["models"]["data"][0]["id"], "messages": messages, "temperature": 0, "max_tokens": 32}
                if not args.mmproj:
                    try:
                        call("/v1/chat/completions", body).close()
                        raise AssertionError("image accepted without an encoder")
                    except urllib.error.HTTPError as error:
                        assert 400 <= error.code < 500
                        report["unloaded_image_rejected"] = error.read().decode()
                else:
                    with call("/v1/chat/completions", body) as response: buffered = json.load(response)
                    visible = buffered["choices"][0]["message"].get("content") or ""
                    assert all(word in visible.lower() for word in ("red", "blue", "square", "circle")), visible
                    body["stream"] = True
                    streamed = ""
                    with call("/v1/chat/completions", body) as response:
                        for line in response:
                            if line.startswith(b"data: ") and line.strip() != b"data: [DONE]":
                                item = json.loads(line[6:])
                                for choice in item.get("choices", []): streamed += choice.get("delta", {}).get("content") or ""
                    assert streamed == visible, (streamed, visible)
                    report["direct_image"] = buffered
                    # Same token positions and image token count, different
                    # pixels: a cached frontier must not alias the images.
                    from PIL import Image, ImageDraw
                    alternate = Image.new("RGB", (256, 128), "#f4f4f4")
                    draw = ImageDraw.Draw(alternate)
                    draw.rectangle((24, 24, 104, 104), fill="blue")
                    draw.ellipse((152, 24, 232, 104), fill="red")
                    png = io.BytesIO()
                    alternate.save(png, format="PNG")
                    second = {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png.getvalue()).decode()}}
                    report["image_replacement"] = []
                    for attachment, expected in ((image, "red"), (second, "blue")):
                        replacement = {"model": body["model"], "messages": [{"role": "user", "content": [attachment, {"type": "text", "text": "What color is the square? Answer one word."}]}], "temperature": 0, "max_tokens": 8}
                        with call("/v1/chat/completions", replacement) as response: result = json.load(response)
                        assert expected in result["choices"][0]["message"]["content"].lower(), result
                        report["image_replacement"].append(result)
                    ordered = {"model": body["model"], "messages": [{"role": "user", "content": [image, second, {"type": "text", "text": "Which image has a blue square: first or second? Answer one word."}]}], "temperature": 0, "max_tokens": 12}
                    with call("/v1/chat/completions", ordered) as response: report["image_order"] = json.load(response)
                    assert "second" in report["image_order"]["choices"][0]["message"]["content"].lower(), report["image_order"]
                    history = messages + [{"role": "assistant", "content": visible}, {"role": "user", "content": "What color is the square? Answer one word."}]
                    with call("/v1/chat/completions", {"model": body["model"], "messages": history, "temperature": 0, "max_tokens": 8}) as response: report["image_history"] = json.load(response)
                    assert "red" in report["image_history"]["choices"][0]["message"]["content"].lower()
                    inputs = [{"role": "user", "content": question},
                              {"type": "function_call", "name": "read_image", "call_id": "gemma_image_1", "arguments": '{"path":"shapes.png"}'},
                              {"type": "function_call_output", "call_id": "gemma_image_1", "output": [{"type": "input_text", "text": "Loaded image"}, {"type": "input_image", "image_url": url}]}]
                    response_body = {"model": body["model"], "input": inputs, "temperature": 0, "max_output_tokens": 32}
                    with call("/v1/responses", response_body) as response: returned = json.load(response)
                    report["tool_result_image"] = returned
                    def response_text(value):
                        return "".join(part.get("text", "") for item in value.get("output", []) for part in item.get("content", []) if part.get("type") == "output_text")
                    returned_text = response_text(returned)
                    # The conventional artifact can end this immediate tool
                    # continuation after a newline. Preserve that result and
                    # require exact streaming equality, then ask explicitly
                    # about the image retained in the tool history.
                    if returned_text.strip():
                        assert all(word in returned_text.lower() for word in ("red", "blue", "square", "circle")), returned_text
                    response_body["stream"] = True
                    chunks = ""
                    with call("/v1/responses", response_body) as response:
                        for line in response:
                            if line.startswith(b"data: "):
                                event = json.loads(line[6:])
                                if event.get("type") == "response.output_text.delta": chunks += event["delta"]
                    assert chunks == returned_text, (chunks, returned_text)
                    followup_inputs = inputs + returned["output"] + [{"role": "user", "content": question}]
                    response_body = {"model": body["model"], "input": followup_inputs, "temperature": 0, "max_output_tokens": 32}
                    with call("/v1/responses", response_body) as response: followup = json.load(response)
                    report["tool_image_followup"] = followup
                    followup_text = response_text(followup)
                    assert all(word in followup_text.lower() for word in ("red", "blue", "square", "circle")), followup_text
                    response_body["stream"] = True
                    chunks = ""
                    with call("/v1/responses", response_body) as response:
                        for line in response:
                            if line.startswith(b"data: "):
                                event = json.loads(line[6:])
                                if event.get("type") == "response.output_text.delta": chunks += event["delta"]
                    assert chunks == followup_text, (chunks, followup_text)
                    response_body.pop("stream")
                    response_body["input"] = followup_inputs + followup["output"] + [{"role": "user", "content": "Which shape is blue? Answer briefly."}]
                    response_body["max_output_tokens"] = 12
                    with call("/v1/responses", response_body) as response: report["responses_image_replay"] = json.load(response)
                    assert "circle" in response_text(report["responses_image_replay"]).lower()
            report["passed"] = True
            args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print("PASS: raw, buffered/streaming text and reasoning, usage, explicit tool rejection")
        finally:
            args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            process.terminate()
            try: process.wait(timeout=20)
            except subprocess.TimeoutExpired: process.kill(); process.wait()


if __name__ == "__main__":
    main()
