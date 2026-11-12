#!/usr/bin/env python3
"""
Step 9.1: OpenAI-Compatible Local & Cloud REST API Server for OdiaSLM (60M & 100M).
Loads the exported Hugging Face `LlamaForCausalLM` safetensors directory
(`artifacts/hf_export/odia-slm-60m-base` or `odia-slm-100m-base`) and serves:
  - GET  /health
  - POST /v1/completions
  - POST /v1/chat/completions
Works out-of-the-box with zero external web framework dependencies (uses threaded HTTPServer).
"""

import argparse
import json
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import sentencepiece as spm
import torch
from transformers import AutoModelForCausalLM

_MODEL = None
_SP = None
_DEVICE = None
_MODEL_NAME = "odia-slm-60m-base"


@torch.no_grad()
def generate_completion(
    prompt: str,
    max_tokens: int = 64,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> dict:
    t0 = time.time()
    prompt_ids = [1] + _SP.encode(prompt, out_type=int)
    max_ctx = getattr(_MODEL.config, "max_position_embeddings", 1024)
    if len(prompt_ids) > max_ctx - max_tokens:
        prompt_ids = [1] + prompt_ids[-(max_ctx - max_tokens - 1):]

    x = torch.tensor([prompt_ids], dtype=torch.long, device=_DEVICE)
    generated_ids = []

    for _ in range(max_tokens):
        logits = _MODEL(x).logits[:, -1, :].float()
        if temperature <= 1e-4:
            next_id = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            remove_mask = cum_probs > top_p
            remove_mask[..., 1:] = remove_mask[..., :-1].clone()
            remove_mask[..., 0] = False
            scatter_mask = remove_mask.scatter(1, sorted_indices, remove_mask)
            logits[scatter_mask] = float("-inf")
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

        tok_id = int(next_id.item())
        if tok_id == 2:  # EOS
            break
        generated_ids.append(tok_id)
        x = torch.cat([x, next_id], dim=-1)

    completion_text = _SP.decode(generated_ids)
    elapsed = max(1e-4, time.time() - t0)
    return {
        "completion_text": completion_text,
        "prompt_tokens": len(prompt_ids),
        "completion_tokens": len(generated_ids),
        "total_tokens": len(prompt_ids) + len(generated_ids),
        "tokens_per_sec": round(len(generated_ids) / elapsed, 2),
    }


class OdiaSLMAPIHandler(BaseHTTPRequestHandler):
    def _send_json(self, status_code: int, payload: dict):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health" or self.path == "/v1/models":
            self._send_json(200, {
                "status": "ok",
                "model": _MODEL_NAME,
                "architecture": "LlamaForCausalLM (GQA + SwiGLU + RoPE)",
                "vocab_size": _SP.get_piece_size(),
                "device": str(_DEVICE),
            })
        else:
            self._send_json(404, {"error": f"Unknown endpoint {self.path}"})

    def do_POST(self):
        content_len = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(content_len).decode("utf-8") if content_len > 0 else "{}"
        try:
            req = json.loads(raw_body)
        except Exception as e:
            self._send_json(400, {"error": f"Invalid JSON: {e}"})
            return

        max_tokens = int(req.get("max_tokens", 64))
        temperature = float(req.get("temperature", 0.7))
        top_p = float(req.get("top_p", 0.9))

        if self.path == "/v1/completions":
            prompt = str(req.get("prompt", ""))
            gen = generate_completion(prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p)
            self._send_json(200, {
                "id": f"cmpl-{uuid.uuid4().hex[:12]}",
                "object": "text_completion",
                "created": int(time.time()),
                "model": _MODEL_NAME,
                "choices": [{
                    "text": gen["completion_text"],
                    "index": 0,
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": gen["prompt_tokens"],
                    "completion_tokens": gen["completion_tokens"],
                    "total_tokens": gen["total_tokens"],
                    "tokens_per_sec": gen["tokens_per_sec"],
                },
            })
        elif self.path == "/v1/chat/completions":
            messages = req.get("messages", [])
            prompt_parts = []
            for m in messages:
                content = str(m.get("content", "")).strip()
                if content:
                    prompt_parts.append(content)
            prompt = "\n".join(prompt_parts)
            gen = generate_completion(prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p)
            self._send_json(200, {
                "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": _MODEL_NAME,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": gen["completion_text"],
                    },
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": gen["prompt_tokens"],
                    "completion_tokens": gen["completion_tokens"],
                    "total_tokens": gen["total_tokens"],
                    "tokens_per_sec": gen["tokens_per_sec"],
                },
            })
        else:
            self._send_json(404, {"error": f"Unknown POST endpoint {self.path}"})


def main():
    global _MODEL, _SP, _DEVICE, _MODEL_NAME
    parser = argparse.ArgumentParser(description="OpenAI-Compatible REST API Server for OdiaSLM")
    parser.add_argument("--model-dir", type=str, default="artifacts/hf_export/odia-slm-60m-base")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    assert model_dir.exists(), f"Exported model directory not found: {model_dir}"
    _MODEL_NAME = model_dir.name

    _DEVICE = torch.device("cpu")
    if torch.cuda.is_available():
        try:
            _ = torch.zeros(1, device="cuda")
            _DEVICE = torch.device("cuda:0")
        except Exception:
            _DEVICE = torch.device("cpu")

    print(f"[API Server] Loading {model_dir} on {_DEVICE} ...")
    _SP = spm.SentencePieceProcessor(model_file=str(model_dir / "tokenizer.model"))
    _MODEL = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.float32, trust_remote_code=False
    ).to(_DEVICE)
    _MODEL.eval()

    server = ThreadingHTTPServer((args.host, args.port), OdiaSLMAPIHandler)
    print(f"[API Server] Ready at http://{args.host}:{args.port}")
    print("  - GET  /health")
    print("  - POST /v1/completions")
    print("  - POST /v1/chat/completions")
    server.serve_forever()


if __name__ == "__main__":
    main()
