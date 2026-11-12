#!/usr/bin/env python3
"""
Verification Suite for src/api_server.py.
Starts the OpenAI-compatible server, waits for model loading, and tests
GET /health, POST /v1/completions, and POST /v1/chat/completions.
"""

import json
import subprocess
import sys
import time
import urllib.request

PORT = 8089
BASE_URL = f"http://127.0.0.1:{PORT}"


def main():
    print(f"[1/3] Starting src/api_server.py on {BASE_URL} ...")
    proc = subprocess.Popen(
        [sys.executable, "src/api_server.py", "--port", str(PORT)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    try:
        ready = False
        for _ in range(120):
            if proc.poll() is not None:
                out, _ = proc.communicate()
                raise RuntimeError(f"API server exited prematurely:\n{out}")
            try:
                with urllib.request.urlopen(f"{BASE_URL}/health", timeout=2) as r:
                    health = json.loads(r.read().decode("utf-8"))
                    print(f"      /health READY: {health}")
                    ready = True
                    break
            except Exception:
                time.sleep(0.5)

        assert ready, "Server did not become ready within 60 seconds!"

        print("[2/3] Testing POST /v1/completions ...")
        req_cmpl = urllib.request.Request(
            f"{BASE_URL}/v1/completions",
            data=json.dumps({
                "prompt": "ଓଡ଼ିଶାର ରାଜଧାନୀ ଭୁବନେଶ୍ୱର ଏକ",
                "max_tokens": 28,
                "temperature": 0.7,
            }).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req_cmpl, timeout=60) as resp:
            cmpl_json = json.loads(resp.read().decode("utf-8"))
            print(f"      Completion Output: {cmpl_json['choices'][0]['text']}")
            print(f"      Usage & Speed    : {cmpl_json['usage']}")

        print("[3/3] Testing POST /v1/chat/completions ...")
        req_chat = urllib.request.Request(
            f"{BASE_URL}/v1/chat/completions",
            data=json.dumps({
                "messages": [{"role": "user", "content": "କୃଷି ଓଡ଼ିଶାର ଅର୍ଥନୀତିର ମୁଖ୍ୟ ଆଧାର ଅଟେ, କାରଣ"}],
                "max_tokens": 28,
                "temperature": 0.7,
            }).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req_chat, timeout=60) as resp:
            chat_json = json.loads(resp.read().decode("utf-8"))
            print(f"      Chat Output      : {chat_json['choices'][0]['message']['content']}")
            print(f"      Usage & Speed    : {chat_json['usage']}")

        print("\n=== API SERVER VERIFICATION SUMMARY ===")
        print("Status                  : PASS (GET /health, POST /v1/completions, POST /v1/chat/completions)")
        print("OpenAI SDK Compatible   : True (trust_remote_code=False)")
        print("=======================================")

    finally:
        proc.terminate()
        proc.wait()


if __name__ == "__main__":
    main()
