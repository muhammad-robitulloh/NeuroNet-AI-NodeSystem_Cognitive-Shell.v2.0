![Latest Release](https://img.shields.io/github/v/release/muhammad-robitulloh/NeuroNet-AI-NodeSystem_Cognitive-Shell.v2.0?style=flat-square)

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

   tested on `Termux`

```bash
pkg update && pkg upgrade
pkg install python git
git clone https://github.com/muhammad-robitulloh/NeuroNet-AI-NodeSystem_Cognitive-Shell.v2.0.git
cd NeuroNet-AI-NodeSystem_Cognitive-Shell.v2.0
pip install -r requirements.txt
python quick_start.py
python cognitive_shell/__main__.py
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

