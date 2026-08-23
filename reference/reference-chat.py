"""Ground-truth chat-template rendering via jinja2 (the engine transformers uses
under the hood), with HF's environment settings. chat-template.ts must reproduce
these exactly. Run: python3 reference-chat.py -> writes tests/chat-fixtures.json
"""
import json
from jinja2.sandbox import ImmutableSandboxedEnvironment

tmpl = json.load(open("models/tokenizer_config-qwen.json"))["chat_template"]
env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
template = env.from_string(tmpl)

cases = [
    {"messages": [{"role": "user", "content": "What is the capital of France?"}],
     "add_generation_prompt": True, "enable_thinking": False},
    {"messages": [{"role": "system", "content": "You are a concise assistant."},
                  {"role": "user", "content": "Hello!"}],
     "add_generation_prompt": True, "enable_thinking": False},
    {"messages": [{"role": "user", "content": "Hi"},
                  {"role": "assistant", "content": "Hello there!"},
                  {"role": "user", "content": "How are you?"}],
     "add_generation_prompt": True, "enable_thinking": False},
    {"messages": [{"role": "user", "content": "2+2?"}],
     "add_generation_prompt": False, "enable_thinking": False},
]

out = [{"input": c, "rendered": template.render(**c)} for c in cases]
json.dump(out, open("tests/chat-fixtures.json", "w"), ensure_ascii=False, indent=1)
print(f"wrote {len(out)} chat fixtures")
print(repr(out[0]["rendered"]))
