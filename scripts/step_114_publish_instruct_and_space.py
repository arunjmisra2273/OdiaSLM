#!/usr/bin/env python3
"""
Phase 5 (Step 11.4): Public Release of arunjmisra2273/OdiaSLM-100M-Instruct on Hugging Face Hub
+ Automatic Deployment of Interactive Gradio Web Chat Space (arunjmisra2273/OdiaSLM-Chat).
Author: Arun Misra | AI Researcher, Bhubaneswar, Odisha, India
"""

import json
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

os.environ["HF_HUB_DISABLE_XET"] = "1"


def load_env_file(env_path: str = ".env") -> None:
    p = Path(env_path)
    if not p.exists():
        return
    for raw in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip().replace("\r", "")
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = v.strip().strip('"').strip("'")
        if k.strip() and v:
            os.environ[k.strip()] = v


load_env_file(".env")
HF_TOKEN = os.environ.get("HF_TOKEN")

from huggingface_hub import HfApi

EXPORT_DIR = Path("artifacts/hf_export/odia-slm-100m-instruct")
SPACE_DIR = Path("artifacts/hf_space_odia_slm")
MODEL_REPO = "arunjmisra2273/OdiaSLM-100M-Instruct"
SPACE_REPO = "arunjmisra2273/OdiaSLM-Chat"
CB = "`" * 3


def build_instruct_model_card() -> str:
    lines = [
        "---",
        "language:",
        "- or",
        "- en",
        "license: other",
        "license_name: odiaslm-dual-nc-1.0",
        "license_link: LICENSE",
        "library_name: transformers",
        "pipeline_tag: text-generation",
        "tags:",
        "- odia",
        "- oriya",
        "- indic-llm",
        "- small-language-model",
        "- instruct",
        "- sft",
        "- dpo",
        "- llama",
        "- from-scratch",
        "- odisha",
        "base_model: arunjmisra2273/OdiaSLM-100M-Base",
        "---",
        "",
        "# OdiaSLM-100M-Instruct: Sovereign Instruction-Tuned Small Language Model for Odia",
        "",
        "**`OdiaSLM-100M-Instruct`** (`100,685,568` parameters) is the instruction-tuned and preference-aligned conversational edition of **[`OdiaSLM-100M-Base`](https://huggingface.co/arunjmisra2273/OdiaSLM-100M-Base)**—the first sovereign decoder-only Small Language Model pre-trained **from scratch** for the **Odia language (`ory_Orya`)**.",
        "",
        "Developed in **Bhubaneswar, Odisha, India** by AI Researcher **Arun Misra**, `OdiaSLM-100M-Instruct` combines a morphology-aware 16,384-token Odia BPE vocabulary (`1.4809 tokens/word` fertility—**3.6x more efficient** than frontier multilingual LLMs) with a 3-stage post-training alignment pipeline (**Full-Parameter SFT + Direct Preference Optimization + Balanced Odisha Heritage & EOS Calibration**).",
        "",
        "* **Live Web Chat Space:** [**`https://huggingface.co/spaces/arunjmisra2273/OdiaSLM-Chat`**](https://huggingface.co/spaces/arunjmisra2273/OdiaSLM-Chat)",
        "* **Preprint Paper (Zenodo DOI):** [**`10.5281/zenodo.24167433`**](https://doi.org/10.5281/zenodo.24167433)",
        "* **GitHub Repository:** [**`https://github.com/arunjmisra2273/OdiaSLM`**](https://github.com/arunjmisra2273/OdiaSLM)",
        "* **Base Models:** [`OdiaSLM-100M-Base`](https://huggingface.co/arunjmisra2273/OdiaSLM-100M-Base) | [`OdiaSLM-60M-Base`](https://huggingface.co/arunjmisra2273/OdiaSLM-60M-Base)",
        "",
        "---",
        "",
        "## 1. Architecture & Post-Training Pipeline Summary",
        "",
        "| Specification | `OdiaSLM-100M-Instruct` |",
        "| :--- | :--- |",
        "| **Architecture** | `LlamaForCausalLM` (`14` Layers, `d_model = 768`, `12` Q Heads / `4` KV Heads GQA, `2048` SwiGLU, RoPE theta=10,000, Pre-RMSNorm) |",
        "| **Total Parameters** | **`100,685,568`** (`100.69M`) |",
        "| **Tokenizer** | Custom 16,384-vocab Morphology-Aware Odia SentencePiece BPE (`1.4809` tokens/word, `0` `<unk>`) |",
        "| **Stage 1: Pretraining** | `1.57B` tokens (`9.62M` deduplicated & 13-gram decontaminated Odia docs; Base Val PPL `17.14`) |",
        "| **Stage 2: Supervised Fine-Tuning (SFT)** | **Full-parameter SFT** on `127,966` decontaminated Odia instruction/QA pairs packed into `25,988` sequences of `1,024` tokens (`1,220` steps, NEFTune alpha=5.0, Z-loss 1e-4, assistant-only loss masking; **SFT Val PPL = `10.32`**, **Top-1 Acc = `51.77%`**) |",
        "| **Stage 3: Preference Alignment (DPO)** | **Direct Preference Optimization (`DPO`) + Chosen-NLL Anchor** on `15,000` `(prompt, chosen, rejected)` triplets targeting anti-repetition, completeness, and script purity (**DPO Val Preference Accuracy = `99.56%`**), followed by **`0.70 DPO + 0.30 SFT` Model Soup blending** & Odisha cultural/EOS calibration |",
        "| **CPU Inference Speed** | **30-34.5 tokens/sec (~140+ Odia chars/sec)** on a standard laptop CPU (`trust_remote_code=False`) |",
        "",
        "---",
        "",
        "## 2. Quickstart Usage (Hugging Face `transformers`)",
        "",
        "`OdiaSLM-100M-Instruct` uses standard `LlamaForCausalLM` weights and requires **zero custom modeling code** (`trust_remote_code=False`):",
        "",
        f"{CB}python",
        "import torch",
        "from transformers import AutoModelForCausalLM, AutoTokenizer",
        "",
        'model_id = "arunjmisra2273/OdiaSLM-100M-Instruct"',
        "tokenizer = AutoTokenizer.from_pretrained(model_id)",
        "model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32).eval()",
        "",
        'question = "ଓଡ଼ିଶାର ରାଜଧାନୀ କ\'ଣ ଏବଂ ଏହାର ମୁଖ୍ୟ ବିଶେଷତ୍ୱ କ\'ଣ?"',
        'prompt = f"<|user|>\\n{question}\\n<|assistant|>\\n"',
        "",
        'inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)',
        "outputs = model.generate(",
        "    **inputs,",
        "    max_new_tokens=110,",
        "    do_sample=False,",
        "    repetition_penalty=1.18,",
        "    no_repeat_ngram_size=4,",
        "    eos_token_id=tokenizer.eos_token_id,",
        "    pad_token_id=tokenizer.pad_token_id,",
        ")",
        "",
        'response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)',
        "print(response.strip())",
        f"{CB}",
        "",
        "---",
        "",
        "## 3. Sample Odia Responses (`OdiaSLM-100M-Instruct`)",
        "",
        "| User Prompt (`<|user|>`) | Model Response (`<|assistant|>`) |",
        "| :--- | :--- |",
        "| **ଆପଣ କିଏ?** | ମୁଁ ହେଉଛି ‘OdiaSLM-100M-Instruct’, ଓଡ଼ିଆ ଭାଷା ପାଇଁ ସମ୍ପୂର୍ଣ୍ଣ ମୂଳରୁ ପ୍ରଶିକ୍ଷିତ ଏକ ସ୍ୱଦେଶୀ କୃତ୍ରିମ ବୁଦ୍ଧିମତ୍ତା (AI) ଭାଷା ମଡେଲ୍। ମୋତେ ଭୁବନେଶ୍ୱରର ଏଆଇ ଗବେଷକ ଅରୁଣ ମିଶ୍ର (Arun Misra) ବିକଶିତ କରିଛନ୍ତି। |",
        "| **ଓଡ଼ିଶାର ରାଜଧାନୀ କ'ଣ ଏବଂ ଏହାର ମୁଖ୍ୟ ବିଶେଷତ୍ୱ କ'ଣ?** | ଓଡ଼ିଶାର ରାଜଧାନୀ ହେଉଛି ଭୁବନେଶ୍ୱର। ଏହାକୁ ଭାରତର ‘ମନ୍ଦିର ମାଳିନୀ ସହର’ କୁହାଯାଏ। ପ୍ରାଚୀନ ଲିଙ୍ଗରାଜ ମନ୍ଦିର, ଧଉଳି ଶାନ୍ତିସ୍ତୂପ, ଖଣ୍ଡଗିରି-ଉଦୟଗିରି ଗୁମ୍ଫା ସହିତ ଆଧୁନିକ ଶିକ୍ଷା ଏବଂ ସୂଚନା ପ୍ରଯୁକ୍ତିବିଦ୍ୟା (IT) କେନ୍ଦ୍ର ଭାବେ ଭୁବନେଶ୍ୱର ପ୍ରସିଦ୍ଧ। |",
        "| **କୋଣାର୍କ ସୂର୍ଯ୍ୟ ମନ୍ଦିର ବିଷୟରେ ସଂକ୍ଷେପରେ କୁହନ୍ତୁ।** | କୋଣାର୍କ ସୂର୍ଯ୍ୟ ମନ୍ଦିର ଓଡ଼ିଶାର ପୁରୀ ଜିଲ୍ଲାରେ ଚନ୍ଦ୍ରଭାଗା ନଦୀ କୂଳରେ ଅବସ୍ଥିତ ଏକ ବିଶ୍ୱପ୍ରସିଦ୍ଧ ଐତିହାସିକ ମନ୍ଦିର। ତ୍ରୟୋଦଶ ଶତାବ୍ଦୀରେ ଗଙ୍ଗ ବଂଶର ରାଜା ପ୍ରଥମ ନରସିଂହ ଦେବ ଏହାକୁ ନିର୍ମାଣ କରିଥିଲେ। କଳିଙ୍ଗ ସ୍ଥାପତ୍ୟ କଳାରେ ନିର୍ମିତ ଏହି ରଥ ଆକୃତିର ମନ୍ଦିର ୟୁନେସ୍କୋ (UNESCO) ବିଶ୍ୱ ଐତିହ୍ୟ ତାଲିକାରେ ଅନ୍ତର୍ଭୁକ୍ତ। |",
        "| **ଓଡ଼ିଶାର ପ୍ରସିଦ୍ଧ ଖାଦ୍ୟ ‘ପଖାଳ’ ବିଷୟରେ ସଂକ୍ଷେପରେ କୁହନ୍ତୁ।** | ପଖାଳ ହେଉଛି ଓଡ଼ିଶାର ସବୁଠାରୁ ଲୋକପ୍ରିୟ ଏବଂ ପାରମ୍ପରିକ ଖାଦ୍ୟ। ରନ୍ଧା ଭାତରେ ପାଣି ଓ ଦହି ମିଶାଇ ପ୍ରସ୍ତୁତ କରାଯାଉଥିବା ଏହି ଖାଦ୍ୟ ଗ୍ରୀଷ୍ମ ଋତୁରେ ଶରୀରକୁ ଶୀତଳତା ଓ ଶକ୍ତି ପ୍ରଦାନ କରେ। ପ୍ରତିବର୍ଷ ମାର୍ଚ୍ଚ ୨୦ ତାରିଖକୁ ‘ବିଶ୍ୱ ପଖାଳ ଦିବସ’ ଭାବେ ପାଳନ କରାଯାଏ। |",
        "| **ସୌରମଣ୍ଡଳରେ କେତୋଟି ଗ୍ରହ ଅଛନ୍ତି ଏବଂ ସବୁଠାରୁ ବଡ଼ ଗ୍ରହର ନାମ କ'ଣ?** | ସୌରମଣ୍ଡଳରେ ମୋଟ ୮ଟି ଗ୍ରହ ଅଛନ୍ତି—ବୁଧ, ଶୁକ୍ର, ପୃଥିବୀ, ମଙ୍ଗଳ, ବୃହସ୍ପତି, ଶନି, ୟୁରାନସ୍ ଏବଂ ନେପଚ୍ୟୁନ୍। ସେମାନଙ୍କ ମଧ୍ୟରୁ ‘ବୃହସ୍ପତି’ (Jupiter) ହେଉଛି ସୌରମଣ୍ଡଳର ସବୁଠାରୁ ବଡ଼ ଗ୍ରହ। |",
        "",
        "---",
        "",
        "## 4. Citation",
        "",
        f"{CB}bibtex",
        "@techreport{misra2026odiaslm,",
        "  author      = {Arun Misra},",
        "  title       = {{OdiaSLM: From-Scratch Pretraining, Morphology-Aware Tokenization, and Calibrated Evaluation of 60M and 100M Small Language Models for Odia}},",
        "  institution = {Independent AI Research, Bhubaneswar, Odisha, India},",
        "  year        = {2026},",
        "  month       = nov,",
        "  doi         = {10.5281/zenodo.24167433},",
        "  url         = {[https://doi.org/10.5281/zenodo.24167433](https://doi.org/10.5281/zenodo.24167433)}",
        "}",
        f"{CB}",
        "",
        "## 5. License & Commercial Inquiries",
        "",
        "Released under the **OdiaSLM Non-Commercial Research & Educational License v1.0** (`LICENSE`). Free for academic research, classroom education, personal experimentation, and non-profit cultural preservation. For commercial licensing, contact **Arun Misra** (`arunjmisra2273@gmail.com`).",
    ]
    return "\n".join(lines) + "\n"


def build_space_files():
    SPACE_DIR.mkdir(parents=True, exist_ok=True)
    readme_lines = [
        "---",
        "title: OdiaSLM 100M Instruct Chat",
        "emoji: 🏛️",
        "colorFrom: orange",
        "colorTo: red",
        "sdk: gradio",
        "sdk_version: 5.9.1",
        "app_file: app.py",
        "pinned: true",
        "---",
        "",
        "# 🏛️ OdiaSLM-100M-Instruct — Live Odia AI Assistant",
        "Interactive demo for **`arunjmisra2273/OdiaSLM-100M-Instruct`**, developed from scratch in Bhubaneswar, Odisha by AI Researcher **Arun Misra**.",
    ]
    (SPACE_DIR / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")

    reqs = "torch\ntransformers>=4.44.0\nsentencepiece\nsafetensors\naccelerate\n"
    (SPACE_DIR / "requirements.txt").write_text(reqs, encoding="utf-8")

    app_lines = [
        "import time",
        "import gradio as gr",
        "import sentencepiece as spm",
        "import torch",
        "from huggingface_hub import hf_hub_download",
        "from transformers import AutoModelForCausalLM",
        "",
        'MODEL_ID = "arunjmisra2273/OdiaSLM-100M-Instruct"',
        "",
        'print(f"Loading {MODEL_ID} on CPU...")',
        'tok_file = hf_hub_download(repo_id=MODEL_ID, filename="tokenizer.model")',
        "sp = spm.SentencePieceProcessor(model_file=tok_file)",
        "model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32, trust_remote_code=False).eval()",
        'print("Model loaded successfully!")',
        "",
        "",
        "@torch.no_grad()",
        "def generate_odia_response(prompt: str, max_new_tokens: int = 110, repetition_penalty: float = 1.18):",
        '    q = (prompt or "").strip()',
        "    if not q:",
        '        return "ଦୟାକରି ଓଡ଼ିଆରେ ଏକ ପ୍ରଶ୍ନ ଲେଖନ୍ତୁ।", "0.0 tok/s"',
        "",
        '    formatted = f"<|user|>\\n{q}\\n<|assistant|>\\n"',
        "    in_ids = torch.tensor([[sp.bos_id()] + sp.encode(formatted, out_type=int)], dtype=torch.long)",
        "",
        "    t0 = time.time()",
        "    out_ids = model.generate(",
        "        in_ids,",
        "        max_new_tokens=int(max_new_tokens),",
        "        do_sample=False,",
        "        repetition_penalty=float(repetition_penalty),",
        "        no_repeat_ngram_size=4,",
        "        eos_token_id=sp.eos_id(),",
        "        pad_token_id=sp.pad_id(),",
        "    )",
        "    elapsed = time.time() - t0",
        "    new_tokens = out_ids[0, in_ids.size(1):].tolist()",
        "    if sp.eos_id() in new_tokens:",
        "        new_tokens = new_tokens[: new_tokens.index(sp.eos_id())]",
        "    answer = sp.decode(new_tokens).strip()",
        "    tps = len(new_tokens) / max(elapsed, 1e-4)",
        '    stats = f"Generated {len(new_tokens)} Odia tokens in {elapsed:.2f}s ({tps:.1f} tokens/sec on CPU)"',
        "    return answer, stats",
        "",
        "",
        "EXAMPLES = [",
        '    ["ଆପଣ କିଏ?"],',
        '    ["OdiaSLM କ\'ଣ ଏବଂ ଏହାର ବିଶେଷତ୍ୱ କ\'ଣ?"],',
        '    ["ଓଡ଼ିଶାର ରାଜଧାନୀ କ\'ଣ ଏବଂ ଏହାର ମୁଖ୍ୟ ବିଶେଷତ୍ୱ କ\'ଣ?"],',
        '    ["କୋଣାର୍କ ସୂର୍ଯ୍ୟ ମନ୍ଦିର ବିଷୟରେ ସଂକ୍ଷେପରେ କୁହନ୍ତୁ।"],',
        '    ["ପୁରୀ ଶ୍ରୀଜଗନ୍ନାଥ ମନ୍ଦିର ଏବଂ ରଥଯାତ୍ରା ବିଷୟରେ କୁହନ୍ତୁ।"],',
        '    ["ଓଡ଼ିଶାର ପ୍ରସିଦ୍ଧ ଖାଦ୍ୟ ‘ପଖାଳ’ ବିଷୟରେ ସଂକ୍ଷେପରେ କୁହନ୍ତୁ।"],',
        '    ["ଚିଲିକା ହ୍ରଦ କାହିଁକି ପ୍ରସିଦ୍ଧ?"],',
        '    ["ସୌରମଣ୍ଡଳରେ କେତୋଟି ଗ୍ରହ ଅଛନ୍ତି ଏବଂ ସବୁଠାରୁ ବଡ଼ ଗ୍ରହର ନାମ କ\'ଣ?"],',
        '    ["ମଧୁମେହ (ଡାଇବେଟିସ୍) ରୋଗରୁ ରକ୍ଷା ପାଇବା ପାଇଁ କ\'ଣ କରିବା ଉଚିତ୍?"],',
        '    ["ଭାରତର ସ୍ୱାଧୀନତା ସଂଗ୍ରାମରେ ମହାତ୍ମା ଗାନ୍ଧୀଙ୍କ ଭୂମିକା କ\'ଣ ଥିଲା?"],',
        "]",
        "",
        'HEADER_MD = (',
        '    "# 🏛️ OdiaSLM-100M-Instruct: Sovereign Odia AI Assistant\\n"',
        '    "**First from-scratch 100M Small Language Model for the Odia language (`ory_Orya`), instruction-tuned with SFT + DPO.**\\n"',
        '    "Developed in Bhubaneswar, Odisha by **Arun Misra** | "',
        '    "[Paper (DOI: 10.5281/zenodo.24167433)](https://doi.org/10.5281/zenodo.24167433) | "',
        '    "[Model Weights](https://huggingface.co/arunjmisra2273/OdiaSLM-100M-Instruct) | "',
        '    "[GitHub](https://github.com/arunjmisra2273/OdiaSLM)"',
        ')',
        "",
        'with gr.Blocks(title="OdiaSLM-100M-Instruct Live Demo") as demo:',
        "    gr.Markdown(HEADER_MD)",
        "    with gr.Row():",
        "        with gr.Column(scale=3):",
        "            inp = gr.Textbox(",
        '                label="ଆପଣଙ୍କର ପ୍ରଶ୍ନ ଲେଖନ୍ତୁ (Enter your question in Odia)",',
        '                placeholder="ଉଦାହରଣ: ଓଡ଼ିଶାର ରାଜଧାନୀ କ\'ଣ ଏବଂ ଏହାର ମୁଖ୍ୟ ବିଶେଷତ୍ୱ କ\'ଣ?",',
        "                lines=3,",
        "            )",
        "            with gr.Row():",
        '                submit_btn = gr.Button("ଉତ୍ତର ପାଆନ୍ତୁ (Generate Answer)", variant="primary")',
        '                clear_btn = gr.Button("ସଫା କରନ୍ତୁ (Clear)")',
        '            with gr.Accordion("Advanced Generation Settings", open=False):',
        '                max_toks = gr.Slider(32, 180, value=110, step=10, label="Max New Tokens")',
        '                rep_pen = gr.Slider(1.0, 1.4, value=1.18, step=0.02, label="Repetition Penalty")',
        "        with gr.Column(scale=4):",
        '            out_box = gr.Textbox(label="OdiaSLM-100M-Instruct ଉତ୍ତର (Response)", lines=6)',
        '            stat_box = gr.Textbox(label="CPU Telemetry", interactive=False)',
        "",
        "    gr.Examples(",
        "        examples=EXAMPLES,",
        "        inputs=[inp],",
        "        outputs=[out_box, stat_box],",
        "        fn=generate_odia_response,",
        "        cache_examples=False,",
        '        label="Click any sample Odia question below to test immediately:",',
        "    )",
        "",
        "    submit_btn.click(fn=generate_odia_response, inputs=[inp, max_toks, rep_pen], outputs=[out_box, stat_box])",
        "    inp.submit(fn=generate_odia_response, inputs=[inp, max_toks, rep_pen], outputs=[out_box, stat_box])",
        '    clear_btn.click(lambda: ("", ""), outputs=[inp, out_box])',
        "",
        'if __name__ == "__main__":',
        "    demo.launch()",
    ]
    (SPACE_DIR / "app.py").write_text("\n".join(app_lines) + "\n", encoding="utf-8")


def main():
    assert HF_TOKEN, "HF_TOKEN missing in .env!"
    api = HfApi(token=HF_TOKEN)

    (EXPORT_DIR / "README.md").write_text(build_instruct_model_card(), encoding="utf-8")
    lic_src = Path("LICENSE")
    if lic_src.exists():
        (EXPORT_DIR / "LICENSE").write_text(lic_src.read_text(encoding="utf-8"), encoding="utf-8")

    print(f"[1/2] Publishing public model repository: [https://huggingface.co/](https://huggingface.co/){MODEL_REPO} ...")
    api.create_repo(repo_id=MODEL_REPO, repo_type="model", private=False, exist_ok=True)
    api.upload_folder(
        folder_path=str(EXPORT_DIR),
        repo_id=MODEL_REPO,
        repo_type="model",
        commit_message="Release OdiaSLM-100M-Instruct (SFT + DPO + Odisha Heritage Calibrated SafeTensors)",
    )
    print(f"  -> Published Model: [https://huggingface.co/](https://huggingface.co/){MODEL_REPO}")

    print(f"[2/2] Deploying live interactive Gradio Space: [https://huggingface.co/spaces/](https://huggingface.co/spaces/){SPACE_REPO} ...")
    build_space_files()
    api.create_repo(
        repo_id=SPACE_REPO,
        repo_type="space",
        space_sdk="gradio",
        private=False,
        exist_ok=True,
    )
    api.upload_folder(
        folder_path=str(SPACE_DIR),
        repo_id=SPACE_REPO,
        repo_type="space",
        commit_message="Deploy live OdiaSLM-100M-Instruct Gradio web demo",
    )
    print(f"  -> Deployed Live Web Space: [https://huggingface.co/spaces/](https://huggingface.co/spaces/){SPACE_REPO}")

    report = {
        "step": "step_114_publish_instruct_and_space",
        "model_repo_url": f"[https://huggingface.co/](https://huggingface.co/){MODEL_REPO}",
        "live_space_url": f"[https://huggingface.co/spaces/](https://huggingface.co/spaces/){SPACE_REPO}",
    }
    Path("artifacts/step_114_publish_instruct_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\n=== STEP 11.4 PUBLIC MODEL & LIVE WEB SPACE DEPLOYMENT COMPLETE ===")
    print(f"Public Model Hub : [https://huggingface.co/](https://huggingface.co/){MODEL_REPO}")
    print(f"Live Web Demo    : [https://huggingface.co/spaces/](https://huggingface.co/spaces/){SPACE_REPO}")
    print("===================================================================")


if __name__ == "__main__":
    main()
