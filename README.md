

https://github.com/user-attachments/assets/1d9d1373-0920-4b8a-bfda-486b82c7953a

<p align="center">
  
  <a href="https://pypi.org/project/neuronet-ai-cognitiveshell">
    <img src="https://img.shields.io/pypi/v/neuronet-ai-cognitiveshell?color=blue&label=PyPI&logo=python&logoColor=white" alt="PyPI">
  </a>
  
  <a href="https://pypi.org/project/neuronet-ai-cognitiveshell">
    <img src="https://img.shields.io/pypi/dm/neuronet-ai-cognitiveshell?label=downloads&color=success" alt="Downloads">
  </a>

  <a href="https://github.com/muhammad-robitulloh/NeuroNet-AI-NodeSystem_Cognitive-Shell.v2.0/actions/workflows/python-publish.yml">
    <img src="https://github.com/muhammad-robitulloh/NeuroNet-AI-NodeSystem_Cognitive-Shell.v2.0/actions/workflows/python-publish.yml/badge.svg" alt="Build Status">
  </a>

  <a href="https://github.com/muhammad-robitulloh/NeuroNet-AI-NodeSystem_Cognitive-Shell.v2.0/blob/main/LICENSE">
    <img src="https://img.shields.io/github/license/muhammad-robitulloh/NeuroNet-AI-NodeSystem_Cognitive-Shell.v2.0?color=lightgrey" alt="License">
  </a>

  <a href="https://github.com/muhammad-robitulloh/NeuroNet-AI-NodeSystem_Cognitive-Shell.v2.0">
    <img src="https://img.shields.io/github/last-commit/muhammad-robitulloh/NeuroNet-AI-NodeSystem_Cognitive-Shell.v2.0?color=yellow" alt="Last Commit">
  </a>

</p>

# 🧠 NeuroNet AI Node System: Cognitive Shell v2.0

A semi-agentic AI terminal shell that **thinks**, **suggests**, and **acts** — from detecting terminal errors to auto-fixing them using AI, all running on lightweight environments like Termux or Google Cloud Shell.

> **Terminal meets AI. Debugging gets intelligent.**

---

## ✨ Features

- ✅ Real-time **log monitoring**
- 🔍 Error detection via keyword and pattern recognition
- 🤖 **LLM-powered suggestions** via OpenRouter or custom API
- 🔁 Optional **auto-execution** of AI-recommended fixes
- 💬 **Telegram Bot Integration** (send/receive terminal commands via Telegram)
- 📱 Optimized for **low-resource devices** like Android Termux or cloud shells


---

## 📦 Requirements

- Python `>=3.10`
- `pexpect`, `requests`, `re`, `os`, `time`
- Telegram bot token (you can get from [@BotFather](https://t.me/botfather))
- A free OpenRouter API key (https://openrouter.ai/)

---

## 🚀 Installation

1. **Install dependencies:**

   tested on `Termux` &  `Ubuntu Jammy Jellyfish`

```bash
pkg update && pkg upgrade
pkg install python git
git clone https://github.com/muhammad-robitulloh/NeuroNet-AI-NodeSystem_Cognitive-Shell.v2.0.git
cd NeuroNet-AI-NodeSystem_Cognitive-Shell.v2.0
pip install -r requirements.txt
python quick_start.py
python cognitive_shell/cognitiveshell.py
```
2. **Install via PyPI(Python Package Index)**

```bash
pip install neuronet-ai-cognitiveshell
quickstart
cognitiveshell
```

## ⚙️ Under Development

The current version of **Cognitive Shell** is still under active development.

Planned improvements include:
- 🔐 `.env` configuration for API keys and tokens (Telegram, LLM, etc.)
- 🧠 Multi-model support (switching between OpenRouter, local LLMs, etc.)
- 💬 Customizable prompt templates per user
- 📂 Logging and error history system
- 📦 Packaged CLI setup with one-liner installer

This early MVP is functional but minimal — perfect for testing agentic capabilities in lightweight environments like Termux or remote shells.

Expect frequent updates and breaking changes during this phase. Contributions and feedback are welcome!

