#!/bin/bash
echo "📦 Setting up CognitiveShell environment..."

read -p "🔑 Masukkan Telegram Bot Token: " TELEGRAM_BOT_TOKEN
read -p "🆔 Masukkan Telegram Chat ID: " TELEGRAM_CHAT_ID
read -p "🌐 Masukkan LLM API URL (Default: OpenRouter): " LLM_API
read -p "🔐 Masukkan OpenRouter API Key: " OPENROUTER_API_KEY

echo "Membuat file .env ..."
cat > .env <<EOL
TELEGRAM_BOT_TOKEN=$TELEGRAM_BOT_TOKEN
TELEGRAM_CHAT_ID=$TELEGRAM_CHAT_ID
LLM_API=${LLM_API:-https://openrouter.ai/api/v1/chat/completions}
OPENROUTER_API_KEY=$OPENROUTER_API_KEY
EOL

echo "✅ Konfigurasi selesai. Jalankan dengan: python cognitive_shell.py"
