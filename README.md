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

```bash
pkg update && pkg upgrade

pkg install python git

git clone https://github.com/muhammad-robitulloh/NeuroNet-AI-NodeSystem_Cognitive-Shell.v2.0.git

cd NeuroNet-AI-NodeSystem_Cognitive-Shell.v2.0

pip install -r requirements.txt

bash setup.sh

python Cognitive_Shell_v2.0/cognitive_shell.py
