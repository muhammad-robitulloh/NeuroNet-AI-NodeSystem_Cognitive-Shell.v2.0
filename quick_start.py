import os
import time
from pathlib import Path

def print_header():
    try:
        os.system("neofetch")
    except Exception:
        print("📦 CognitiveShell Setup\n------------------------")
    time.sleep(2)
    print("\n📦 Setting up CognitiveShell environment...\n")

def prompt_env_variable(prompt, var_name, default=None):
    value = input(f"{prompt} ")
    if not value and default is not None:
        return default
    return value

def main():
    print_header()

    telegram_token = prompt_env_variable("🔑 Masukkan Telegram Bot Token:", "TELEGRAM_BOT_TOKEN")
    telegram_chat_id = prompt_env_variable("🆔 Masukkan Telegram Chat ID:", "TELEGRAM_CHAT_ID")
    llm_api = prompt_env_variable("🌐 Masukkan LLM API URL (Default: OpenRouter):", "LLM_API", default="https://openrouter.ai/api/v1/chat/completions")
    openrouter_api_key = prompt_env_variable("🔐 Masukkan OpenRouter API Key:", "OPENROUTER_API_KEY")

    env_content = f"""TELEGRAM_BOT_TOKEN={telegram_token}
TELEGRAM_CHAT_ID={telegram_chat_id}
LLM_API={llm_api}
OPENROUTER_API_KEY={openrouter_api_key}
"""

    env_path = Path(".env")
    env_path.write_text(env_content)



if __name__ == "__main__":
    main()
