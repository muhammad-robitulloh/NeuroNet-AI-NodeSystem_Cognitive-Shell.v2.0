import pexpect
import requests
import re
import os
import time
import subprocess
import logging
import shlex  # Untuk sanitasi perintah shell
from telegram import Update
from telegram.constants import ChatAction, ParseMode # Import ChatAction dan ParseMode untuk MarkdownV2
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    CallbackContext,
    ConversationHandler,
    JobQueue # Import JobQueue secara eksplisit
)
from dotenv import load_dotenv
import asyncio # Tetap impor jika dibutuhkan untuk tugas async lain, meskipun tidak untuk inisialisasi loop utama
import pytz # Import pytz untuk penanganan zona waktu (saat ini tidak digunakan, tetapi disimpan)

# Muat variabel lingkungan dari file .env
load_dotenv()

# === Konfigurasi Global ===
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "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")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1/chat/completions")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "6504150585")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "7873624234:AAHJ_Wm6EFgV4W7JdeehfhHjYbVdHiPOk_M")

# Konfigurasi model LLM untuk berbagai tugas
CODE_GEN_MODEL = os.getenv("CODE_GEN_MODEL", "moonshotai/kimi-dev-72b:free")
ERROR_FIX_MODEL = os.getenv("ERROR_FIX_MODEL", "nvidia/llama-3.3-nemotron-super-49b-v1:free")
CONVERSATION_MODEL = os.getenv("CONVERSATION_MODEL", "mistralai/mistral-small-3.2-24b-instruct")
COMMAND_CONVERSION_MODEL = os.getenv("COMMAND_CONVERSION_MODEL", "nvidia/llama-3.3-nemotron-super-49b-v1:free")
FILENAME_GEN_MODEL = os.getenv("FILENAME_GEN_MODEL", "mistralai/mistral-small-3.2-24b-instruct")
INTENT_DETECTION_MODEL = os.getenv("INTENT_DETECTION_MODEL", "mistralai/mistral-small-3.2-24b-instruct")

# Warna ANSI untuk output konsol Termux (hanya untuk log internal)
COLOR_GREEN = "\033[92m"
COLOR_YELLOW = "\033[93m"
COLOR_RED = "\033[91m"
COLOR_BLUE = "\033[94m"
COLOR_PURPLE = "\033[95m"
COLOR_RESET = "\033[0m"

# Status untuk ConversationHandler (debugging)
DEBUGGING_STATE = 1

# --- Konfigurasi Logging ---
# Mengubah format logger agar lebih ringkas dan mudah dibaca
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    level=logging.INFO,
    datefmt='%H:%M:%S' # Menambahkan format waktu yang lebih ringkas
)
logger = logging.getLogger(__name__)

# --- Fungsi Global untuk Penyimpanan Konteks ---
user_contexts: dict = {}
chat_histories: dict = {}

def get_user_context(chat_id: int) -> dict:
    """Mengambil konteks pengguna. Menginisialisasi jika belum ada."""
    if chat_id not in user_contexts:
        user_contexts[chat_id] = {
            "last_error_log": None,
            "last_command_run": None,
            "last_generated_code": None,
            "awaiting_debug_response": False,
            "full_error_output": [],
            "last_user_message_intent": None,
            "last_ai_response_type": None,
            "last_generated_code_language": None
        }
    return user_contexts[chat_id]

def get_chat_history(chat_id: int) -> list:
    """Mengambil riwayat obrolan pengguna. Menginisialisasi jika belum ada."""
    if chat_id not in chat_histories:
        chat_histories[chat_id] = []
    return chat_histories[chat_id]

def _escape_plaintext_markdown_v2(text: str) -> str:
    """
    Meng-escape karakter khusus MarkdownV2 dalam teks biasa (non-kode),
    tetapi secara khusus TIDAK meng-escape '*' dan '_' karena diasumsikan
    digunakan untuk pemformatan tebal dan miring.
    Fungsi ini untuk segmen teks yang BUKAN blok kode.
    """
    # Karakter yang harus di-escape saat muncul secara literal di MarkdownV2.
    # Daftar ini TIDAK termasuk '*' dan '_', karena kita ingin mereka
    # diinterpretasikan sebagai pemformatan.
    chars_to_escape_regex = r'[\[\]()~`>#+\-=|{}.!]' 

    # Escape backslash literal terlebih dahulu, agar tidak mengganggu escape berikutnya
    text = text.replace('\\', '\\\\')

    # Gunakan re.sub untuk meng-escape karakter spesifik
    escaped_text = re.sub(chars_to_escape_regex, r'\\\g<0>', text)
    
    return escaped_text

# === Fungsi Umum: Panggil LLM ===
def call_llm(messages: list, model: str, api_key: str, max_tokens: int = 512, temperature: float = 0.7, extra_headers: dict = None) -> tuple[bool, str]:
    """
    Fungsi generik untuk mengirim permintaan ke model LLM (OpenRouter).
    Mengembalikan tuple: (True, hasil) jika berhasil, (False, pesan_error) jika gagal.
    """
    if not api_key or not LLM_BASE_URL:
        logger.error("[LLM ERROR] Kunci API atau URL Dasar LLM tidak diatur.")
        return False, "Kunci API atau URL Dasar LLM tidak diatur. Harap periksa konfigurasi."

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    if extra_headers:
        headers.update(extra_headers)

    try:
        res = requests.post(LLM_BASE_URL, json=payload, headers=headers, timeout=300)
        res.raise_for_status()
        data = res.json()
        if "choices" in data and data["choices"]:
            return True, data["choices"][0]["message"]["content"]
        else:
            logger.error(f"[LLM] Respons LLM tidak mengandung 'choices'. Debug respons: {data}")
            return False, f"Respons LLM tidak dalam format yang diharapkan. Debug respons: {data}"
    except requests.exceptions.Timeout:
        logger.error(f"[LLM] Permintaan API LLM waktu habis ({LLM_BASE_URL}).")
        return False, f"Permintaan API LLM waktu habis. Silakan coba lagi."
    except requests.exceptions.RequestException as e:
        logger.error(f"[LLM] Gagal terhubung ke API LLM ({LLM_BASE_URL}): {e}")
        return False, f"Gagal terhubung ke API LLM: {e}"
    except KeyError as e:
        logger.error(f"[LLM] Respons LLM tidak dalam format yang diharapkan (tidak ada 'choices' atau 'message'): {e}. Debug respons: {data}")
        return False, f"Respons LLM tidak dalam format yang diharapkan: {e}. Debug respons: {data}"
    except Exception as e:
        logger.error(f"[LLM] Terjadi kesalahan tak terduga saat memanggil LLM: {e}")
        return False, f"Terjadi kesalahan tak terduga saat memanggil LLM: {e}"

# === Fungsi: Ekstrak Kode dari Respons LLM ===
def ekstrak_kode_dari_llm(text_response: str, target_language: str = None) -> tuple[str, str]:
    """
    Mengekstrak blok kode Markdown dari respons LLM.
    Mengembalikan tuple: (kode_bersih, bahasa_terdeteksi)
    """
    code_block_pattern = r"```(?P<lang>\w+)?\n(?P<content>.*?)```"
    matches = re.findall(code_block_pattern, text_response, re.DOTALL)
    
    if matches:
        if target_language:
            for lang, content in matches:
                if lang and lang.lower() == target_language.lower():
                    logger.info(f"{COLOR_GREEN}[LLM] ✔ Kode {target_language} terdeteksi dan diekstrak.{COLOR_RESET}")
                    return content.strip(), lang.lower()
        
        for lang, content in matches:
            if lang and lang.lower() in ["python", "bash", "javascript", "js", "sh", "py", "node"]:
                logger.info(f"{COLOR_GREEN}[LLM] ✔ Kode {lang} terdeteksi dan diekstrak.{COLOR_RESET}")
                return content.strip(), lang.lower().replace("js", "javascript").replace("sh", "bash").replace("py", "python")
        
        if matches:
            logger.info(f"{COLOR_YELLOW}[LLM] ⚠ Blok kode Markdown ditemukan tanpa indikator bahasa spesifik. Mengekstrak dan mencoba menebak bahasa.{COLOR_RESET}")
            first_content = matches[0][1].strip()
            detected_lang = deteksi_bahasa_pemrograman_dari_konten(first_content)
            return first_content, detected_lang
    
    logger.warning(f"{COLOR_YELLOW}[LLM] ⚠ Tidak ada blok kode Markdown terdeteksi. Melakukan pembersihan teks agresif.{COLOR_RESET}")
    lines = text_response.strip().split('\n')
    cleaned_lines = []
    in_potential_code_block = False
    
    for line in lines:
        stripped_line = line.strip()
        if stripped_line.startswith(('#', 'import ', 'from ', 'function ', 'def ', 'const ', 'let ', 'var ', 'echo ', '#!/')):
            cleaned_lines.append(line)
            in_potential_code_block = True
        elif re.match(r'^(def|class|if|for|while|try|with|function|const|let|var)\s+', stripped_line):
            cleaned_lines.append(line)
            in_potential_code_block = True
        elif any(char in stripped_line for char in ['=', '(', ')', '{', '}', '[', ']']) and not stripped_line.startswith('- '):
            cleaned_lines.append(line)
            in_potential_code_block = True
        elif in_potential_code_block and not stripped_line:
            cleaned_lines.append(line)
        elif len(stripped_line) > 0 and not re.match(r'^[a-zA-Z\s,;.:-]*$', stripped_line):
             cleaned_lines.append(line)
             in_potential_code_block = True
        else:
            if in_potential_code_block and stripped_line:
                break 
            pass

    final_code = "\n".join(cleaned_lines).strip()
    final_code = re.sub(r'```(.*?)```', r'\1', final_code, flags=re.DOTALL)
    
    detected_lang = deteksi_bahasa_pemrograman_dari_konten(final_code)
    logger.info(f"{COLOR_YELLOW}[LLM] ⚠ Kode diekstrak dengan pembersihan agresif. Bahasa terdeteksi: {detected_lang}{COLOR_RESET}")
    return final_code.strip(), detected_lang


# === Fungsi: Deteksi Bahasa Pemrograman dari Konten Kode ===
def deteksi_bahasa_pemrograman_dari_konten(code_content: str) -> str:
    """
    Mendeteksi bahasa pemrograman dari konten kode berdasarkan heuristik.
    """
    if not code_content:
        return "txt"

    code_content_lower = code_content.lower()

    if "import" in code_content_lower or "def " in code_content_lower or "class " in code_content_lower or ".py" in code_content_lower:
        return "python"
    if "bash" in code_content_lower or "#!/bin/bash" in code_content or "#!/bin/sh" in code_content or "echo " in code_content_lower or ".sh" in code_content_lower:
        return "bash"
    if "function" in code_content_lower or "console.log" in code_content_lower or "const " in code_content_lower or "let " in code_content_lower or "var " in code_content_lower or ".js" in code_content_lower:
        return "javascript"
    if "<html" in code_content_lower or "<body" in code_content_lower or "<div" in code_content_lower:
        return "html"
    if "body {" in code_content_lower or "background-color" in code_content_lower or "color:" in code_content_lower:
        return "css"
    if "<?php" in code_content_lower or (re.search(r'\becho\b', code_content_lower) and not re.search(r'\bbash\b', code_content_lower)):
        return "php"
    if "public class" in code_content_lower or "public static void main" in code_content_lower or ".java" in code_content_lower:
        return "java"
    if "#include <" in code_content_lower or "int main()" in code_content_lower or ".c" in code_content_lower or ".cpp" in code_content_lower:
        return "c"
    return "txt"


# === Fungsi: Deteksi Niat Pengguna ===
def deteksi_niat_pengguna(pesan_pengguna: str) -> str:
    """
    Mendeteksi niat pengguna (menjalankan perintah shell, membuat program, atau percakapan umum).
    Mengembalikan string: "shell", "program", atau "konversasi".
    """
    messages = [
        {"role": "system", "content": """Anda adalah detektor niat. Identifikasi apakah pesan pengguna bermaksud untuk:
- "shell": Jika pengguna ingin menjalankan perintah sistem atau melakukan operasi file (misalnya, "hapus file", "tampilkan direktori", "jalankan", "buka", "instal", "kompres").
- "program": Jika pengguna ingin membuat atau memperbaiki kode (misalnya, "buat fungsi python", "tulis kode javascript", "perbaiki kesalahan ini", "tulis program").
- "konversasi": Untuk semua jenis pertanyaan atau interaksi lain yang bukan perintah langsung atau pembuatan kode.

Kembalikan hanya satu kata dari kategori di atas. Jangan berikan penjelasan tambahan.
"""},
        {"role": "user", "content": f"Deteksi niat untuk: '{pesan_pengguna}'"}
    ]
    logger.info(f"{COLOR_BLUE}[AI] Mendeteksi niat pengguna untuk '{pesan_pengguna}' ({INTENT_DETECTION_MODEL})...{COLOR_RESET}\n")
    
    success, niat = call_llm(messages, INTENT_DETECTION_MODEL, OPENROUTER_API_KEY, max_tokens=10, temperature=0.0)
    
    if success:
        niat_cleaned = niat.strip().lower()
        if not niat_cleaned:
            logger.warning(f"[AI] Niat kosong dari LLM. Default ke 'konversasi'.")
            return "konversasi"
        elif niat_cleaned in ["shell", "program", "konversasi"]:
            return niat_cleaned
        else:
            logger.warning(f"[AI] Niat tidak dikenal dari LLM: '{niat_cleaned}'. Default ke 'konversasi'.")
            return "konversasi"
    else:
        logger.error(f"[AI] Gagal mendeteksi niat: {niat}. Default ke 'konversasi'.")
        return "konversasi"

# === Fungsi: Deteksi Bahasa Pemrograman yang diminta dalam Prompt ===
def deteksi_bahasa_dari_prompt(prompt: str) -> str | None:
    """
    Mendeteksi bahasa pemrograman yang diminta dalam prompt pengguna.
    Mengembalikan string bahasa (misalnya, "python", "bash", "javascript") atau None jika tidak spesifik.
    """
    prompt_lower = prompt.lower()
    if "python" in prompt_lower or "script python" in prompt_lower or "fungsi python" in prompt_lower:
        return "python"
    elif "bash" in prompt_lower or "shell" in prompt_lower or "script shell" in prompt_lower or "program sh" in prompt_lower:
        return "bash"
    elif "javascript" in prompt_lower or "js" in prompt_lower or "nodejs" in prompt_lower:
        return "javascript"
    elif "html" in prompt_lower or "web page" in prompt_lower:
        return "html"
    elif "css" in prompt_lower or "stylesheet" in prompt_lower:
        return "css"
    elif "php" in prompt_lower:
        return "php"
    elif "java" in prompt_lower:
        return "java"
    elif "c++" in prompt_lower or "cpp" in prompt_lower:
        return "cpp"
    elif "c#" in prompt_lower or "csharp" in prompt_lower:
        return "csharp"
    elif "ruby" in prompt_lower or "rb" in prompt_lower:
        return "ruby"
    elif "go lang" in prompt_lower or "golang" in prompt_lower or "go " in prompt_lower:
        return "go"
    elif "swift" in prompt_lower:
        return "swift"
    elif "kotlin" in prompt_lower:
        return "kotlin"
    elif "rust" in prompt_lower:
        return "rust"
    return None


# === Fungsi: Minta Kode dari LLM ===
def minta_kode(prompt: str, error_context: str = None, chat_id: int = None, target_language: str = None) -> tuple[bool, str, str | None]:
    """
    Meminta LLM untuk menghasilkan kode berdasarkan prompt dalam bahasa tertentu.
    Jika error_context disediakan, ini adalah permintaan debugging.
    Menyertakan konteks percakapan terbaru jika tersedia.
    """
    messages = []
    
    history = get_chat_history(chat_id) if chat_id else []
    recent_history = history[-10:]
    for msg in recent_history:
        messages.append(msg)

    system_message_content = "Anda adalah asisten pengkodean AI yang mahir dalam berbagai bahasa pemrograman. Hasil kode harus *hanya* berupa blok kode Markdown dengan tag bahasa yang sesuai (misalnya, ```python, ```bash, ```javascript, ```html, ```css, ```php, ```java, dll.). JANGAN tambahkan penjelasan, intro, kesimpulan, atau teks tambahan di luar blok kode Markdown. Sertakan semua impor/dependensi yang diperlukan dalam blok kode. Jika ada bagian yang memerlukan input pengguna, berikan komentar yang jelas di dalam kode."

    if error_context:
        messages.append({
                "role": "system", 
                "content": system_message_content + " Anda sedang memperbaiki kode. Berdasarkan log error dan riwayat percakapan yang diberikan, berikan *hanya* kode yang sudah diperbaiki secara lengkap atau kode baru. Pastikan kode dapat langsung dijalankan."
            })
        messages.append({
                "role": "user",
                "content": f"Ada kesalahan saat menjalankan kode/perintah:\n\n{error_context}\n\nPerbaiki atau berikan kode baru yang lengkap. Fokus pada bahasa {target_language if target_language else 'yang relevan'}."
            })
        logger.info(f"{COLOR_BLUE}[AI] Meminta perbaikan/kode baru ({target_language if target_language else 'universal'}) dari model AI ({CODE_GEN_MODEL}) berdasarkan kesalahan...{COLOR_RESET}\n")
    else:
        messages.append({
                "role": "system", 
                "content": system_message_content
            })
        prompt_with_lang = f"Instruksi: {prompt}"
        if target_language:
            prompt_with_lang += f" (dalam bahasa {target_language})"
        messages.append({
                "role": "user",
                "content": prompt_with_lang
            })
        logger.info(f"{COLOR_BLUE}[AI] Meminta kode ({target_language if target_language else 'universal'}) dari model AI ({CODE_GEN_MODEL})...{COLOR_RESET}\n")
    
    success, response_content = call_llm(messages, CODE_GEN_MODEL, OPENROUTER_API_KEY, max_tokens=2048, temperature=0.7)

    if success:
        cleaned_code, detected_language = ekstrak_kode_dari_llm(response_content, target_language)
        return True, cleaned_code, detected_language
    else:
        return False, response_content, None

# === Fungsi: Hasilkan Nama File ===
def generate_filename(prompt: str, detected_language: str = "txt") -> str:
    """
    Menghasilkan nama file yang relevan berdasarkan prompt pengguna dan bahasa yang terdeteksi.
    """
    extension_map = {
        "python": ".py", "bash": ".sh", "javascript": ".js", "html": ".html",
        "css": ".css", "php": ".php", "java": ".java", "c": ".c",
        "cpp": ".cpp", "csharp": ".cs", "ruby": ".rb", "go": ".go",
        "swift": ".swift", "kotlin": ".kt", "rust": ".rs", "txt": ".txt"
    }
    
    messages = [
        {"role": "system", "content": f"Anda adalah generator nama file. Berikan satu nama file singkat, relevan, dan deskriptif (tanpa spasi, gunakan garis bawah, semua huruf kecil, tanpa ekstensi) berdasarkan deskripsi kode berikut dan bahasa '{detected_language}'. Contoh: 'factorial_function' atau 'cli_calculator'. Tanpa penjelasan, hanya nama file."},
        {"role": "user", "content": f"Deskripsi kode: {prompt}"}
    ]
    logger.info(f"{COLOR_BLUE}[AI] Menghasilkan nama file untuk '{prompt}' ({FILENAME_GEN_MODEL}) dengan bahasa {detected_language}...{COLOR_RESET}\n")
    
    success, filename = call_llm(messages, FILENAME_GEN_MODEL, OPENROUTER_API_KEY, max_tokens=20, temperature=0.5)
    
    if not success:
        logger.warning(f"[AI] Gagal menghasilkan nama file dari LLM: {filename}. Menggunakan nama default.")
        return f"generated_code{extension_map.get(detected_language, '.txt')}"

    filename = filename.strip()
    filename = re.sub(r'[^\w-]', '', filename).lower().replace(' ', '_')
    
    for ext in extension_map.values():
        if filename.endswith(ext):
            filename = filename[:-len(ext)]
            break
            
    if not filename:
        filename = "generated_code"
        
    return filename + extension_map.get(detected_language, '.txt')


# === Fungsi: Konversi Bahasa Alami ke Perintah Shell ===
def konversi_ke_perintah_shell(bahasa_natural: str, chat_id: int = None) -> tuple[bool, str]:
    """
    Mengonversi bahasa alami pengguna menjadi perintah shell yang dapat dieksekusi.
    Menyertakan konteks percakapan terbaru jika tersedia.
    """
    messages = []

    history = get_chat_history(chat_id) if chat_id else []
    recent_history = history[-10:] 
    for msg in recent_history:
        messages.append(msg)

    messages.append({"role": "system", "content": "Anda adalah penerjemah bahasa alami ke perintah shell. Konversi instruksi bahasa alami berikut menjadi satu baris perintah shell Linux Termux yang paling relevan. Jangan berikan penjelasan, hanya perintahnya. Jika instruksi tidak jelas atau tidak dapat dikonversi menjadi perintah shell, balas dengan 'CANNOT_CONVERT'."})
    messages.append({"role": "user", "content": f"Konversi ini ke perintah shell: {bahasa_natural}"})

    logger.info(f"{COLOR_BLUE}[AI] Mengonversi bahasa alami ke perintah shell ({COMMAND_CONVERSION_MODEL})...{COLOR_RESET}\n")
    return call_llm(messages, COMMAND_CONVERSION_MODEL, OPENROUTER_API_KEY, max_tokens=128, temperature=0.3)


# === Fungsi: Kirim Error ke LLM untuk Saran ===
def kirim_error_ke_llm_for_suggestion(log_error: str, chat_id: int = None) -> tuple[bool, str]:
    """
    Mengirim log error ke LLM untuk mendapatkan saran perbaikan.
    Menyertakan konteks percakapan terbaru jika tersedia.
    """
    messages = []

    history = get_chat_history(chat_id) if chat_id else []
    recent_history = history[-10:] 
    for msg in recent_history:
        messages.append(msg)

    messages.append({"role": "user", "content": f"Ada error berikut:\n\n{log_error}\n\nApa saran terbaik untuk memperbaikinya dalam konteks sistem Linux Termux? Berikan saran dalam format shell yang dapat dijalankan jika memungkinkan, atau dalam blok kode Markdown. Jika tidak, berikan penjelasan singkat."})
    
    headers = {"HTTP-Referer": "[https://t.me/dseAI_bot](https://t.me/dseAI_bot)"}
    
    logger.info(f"{COLOR_BLUE}[AI] Mengirim error ke model AI ({ERROR_FIX_MODEL}) untuk saran...{COLOR_RESET}\n")
    return call_llm(messages, ERROR_FIX_MODEL, OPENROUTER_API_KEY, max_tokens=512, temperature=0.7, extra_headers=headers)

# === Fungsi: Minta Jawaban Percakapan Umum dari LLM ===
def minta_jawaban_konversasi(chat_id: int, prompt: str) -> tuple[bool, str]:
    """
    Meminta jawaban percakapan umum dari LLM, sambil menyimpan riwayat
    dan menyertakan referensi dari interaksi sebelumnya (kode, perintah).
    """
    history = get_chat_history(chat_id)
    user_context = get_user_context(chat_id)
    
    system_context_messages = []

    if user_context["last_command_run"] and user_context["last_ai_response_type"] == "shell":
        system_context_messages.append(
            {"role": "system", "content": f"Pengguna baru saja menjalankan perintah shell: `{user_context['last_command_run']}`. Pertimbangkan konteks ini dalam jawaban Anda."}
        )
    if user_context["last_generated_code"] and user_context["last_ai_response_type"] == "program":
        lang_display = user_context["last_generated_code_language"] if user_context["last_generated_code_language"] else "code"
        system_context_messages.append(
            {"role": "system", "content": f"Pengguna baru saja menerima kode {lang_display}:\n```{lang_display}\n{user_context['last_generated_code']}\n```. Pertimbangkan konteks ini dalam jawaban Anda."}
        )
    if user_context["last_error_log"] and user_context["last_user_message_intent"] == "shell":
        system_context_messages.append(
            {"role": "system", "content": f"Pengguna mengalami kesalahan setelah menjalankan perintah: `{user_context['last_command_run']}` dengan log error:\n```\n{user_context['full_error_output'][-500:]}\n```. Pertimbangkan ini dalam jawaban Anda."}
        )
    elif user_context["last_error_log"] and user_context["last_user_message_intent"] == "program":
         system_context_messages.append(
            {"role": "system", "content": f"Pengguna mengalami kesalahan setelah berinteraksi dengan program:\n```\n{user_context['full_error_output'][-500:]}\n```. Pertimbangkan ini dalam jawaban Anda."}
        )

    messages_to_send = []
    messages_to_send.extend(system_context_messages)

    max_history_length = 10
    recent_history = history[-max_history_length:]
    messages_to_send.extend(recent_history)
    
    messages_to_send.append({"role": "user", "content": prompt})

    logger.info(f"{COLOR_BLUE}[AI] Meminta jawaban percakapan dari model AI ({CONVERSATION_MODEL})...{COLOR_RESET}\n")
    success, response = call_llm(messages_to_send, CONVERSATION_MODEL, OPENROUTER_API_KEY, max_tokens=256, temperature=0.7)

    if success:
        history.append({"role": "user", "content": prompt})
        history.append({"role": "assistant", "content": response})
        chat_histories[chat_id] = history
    return success, response


# === Fungsi: Simpan ke file ===
def simpan_ke_file(nama_file: str, isi: str) -> bool:
    """
    Menyimpan konten string ke file.
    Mengembalikan True jika berhasil, False jika gagal.
    """
    try:
        with open(nama_file, "w") as f:
            f.write(isi)
        logger.info(f"{COLOR_GREEN}[FILE] ✅ Kode berhasil disimpan ke file: {nama_file}{COLOR_RESET}")
        return True
    except IOError as e:
        logger.error(f"[FILE] 🔴 Gagal menyimpan file {nama_file}: {e}")
        return False

# === Fungsi: Kirim notifikasi Telegram ===
async def kirim_ke_telegram(chat_id: int, context: CallbackContext, pesan_raw: str):
    """
    Mengirim pesan ke Telegram. Menghapus warna ANSI dan menerapkan escaping MarkdownV2
    ke seluruh konten pesan, secara khusus menangani blok kode dengan tidak meng-escape konten internalnya.
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning(f"[Telegram] ⚠ Token BOT Telegram atau ID Obrolan tidak ditemukan. Notifikasi tidak dikirim.")
        return

    # Hapus kode warna ANSI terlebih dahulu
    pesan_bersih_tanpa_ansi = re.sub(r'\033\[[0-9;]*m', '', pesan_raw)
    
    final_message_parts = []
    
    # 1. Pisahkan berdasarkan blok kode multiline (```)
    multiline_split = re.split(r'(```(?:\w+)?\n.*?```)', pesan_bersih_tanpa_ansi, flags=re.DOTALL)
    
    for ml_part in multiline_split:
        if ml_part.startswith('```') and ml_part.endswith('```'):
            # Ini adalah blok kode multiline, tambahkan seperti apa adanya (konten di dalamnya tidak di-escape)
            final_message_parts.append(ml_part)
        else:
            # Ini adalah bagian teks di luar blok kode multiline, sekarang periksa blok kode inline (`)
            inline_split = re.split(r'(`[^`]+`)', ml_part) # Tangkap `...`
            for il_part in inline_split:
                if il_part.startswith('`') and il_part.endswith('`'):
                    # Ini adalah blok kode inline, tambahkan seperti apa adanya (konten di dalamnya tidak di-escape)
                    final_message_parts.append(il_part)
                else:
                    # Ini adalah teks biasa, terapkan escape MarkdownV2 menggunakan fungsi baru
                    final_message_parts.append(_escape_plaintext_markdown_v2(il_part))
            
    pesan_final = "".join(final_message_parts)

    try:
        await context.bot.send_message(chat_id=chat_id, text=pesan_final, parse_mode=ParseMode.MARKDOWN_V2)
        logger.info(f"[Telegram] Notifikasi berhasil dikirim ke {chat_id}.")
    except Exception as e:
        logger.error(f"[Telegram] 🔴 Gagal mengirim pesan ke Telegram: {e}")

# === Fungsi: Deteksi perintah shell dalam saran AI ===
def deteksi_perintah_shell(saran_ai: str) -> str | None:
    """
    Mendeteksi baris perintah shell dari saran AI, termasuk yang ada di
    blok kode Markdown atau kutipan inline.
    Prioritas: Blok kode Markdown > Kutipan inline > Pola Regex reguler
    """
    code_block_pattern = r"```(?:bash|sh|zsh|\w+)?\n(.*?)```"
    inline_code_pattern = r"`([^`]+)`"

    code_blocks = re.findall(code_block_pattern, saran_ai, re.DOTALL)
    for block in code_blocks:
        lines_in_block = [line.strip() for line in block.split('\n') if line.strip()]
        if lines_in_block:
            first_line = lines_in_block[0]
            if any(first_line.startswith(kw) for kw in ["sudo", "apt", "pkg", "pip", "python", "bash", "sh", "./", "chmod", "chown", "mv", "cp", "rmdir", "mkdir", "cd", "ls", "git", "curl", "wget", "tar", "unzip", "zip", "export"]):
                return first_line

    inline_codes = re.findall(inline_code_pattern, saran_ai)
    for code in inline_codes:
        code = code.strip()
        if code and any(code.startswith(kw) for kw in ["sudo", "apt", "pkg", "pip", "python", "bash", "sh", "./", "chmod", "chown", "mv", "cp", "rmdir", "mkdir", "cd", "ls", "git", "curl", "wget", "tar", "unzip", "zip", "export"]):
            return code

    shell_command_patterns = [
        r"^(sudo|apt|pkg|dpkg|pip|python|bash|sh|./|chmod|chown|mv|cp|rmdir|mkdir|cd|ls|grep|find|nano|vi|vim|git|curl|wget|tar|unzip|zip|export|alias)\s+",
        r"^(\S+\.sh)\s+",
        r"^\S+\s+(--\S+|\S+)+",
    ]
    lines = saran_ai.strip().split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
        for pattern in shell_command_patterns:
            if re.match(pattern, line, re.IGNORECASE):
                return line
                
    return None

# === Fungsi Keamanan: Filter Perintah Berbahaya ===
def is_command_dangerous(command: str) -> bool:
    """
    Memeriksa apakah perintah shell mengandung kata kunci terlarang.
    """
    command_lower = command.lower()
    
    dangerous_patterns = [
        r'\brm\b\s+-rf',
        r'\brm\b\s+/\s*',
        r'\bpkg\s+uninstall\b',
        r'\bmv\b\s+/\s*',
        r'\bchown\b\s+root',
        r'\bchmod\b\s+\d{3}\s+/\s*',
        r'\bsu\b',
        r'\bsudo\b\s+poweroff',
        r'\breboot\b',
        r'\bformat\b',
        r'\bmkfs\b',
        r'\bdd\b',
        r'\bfdisk\b',
        r'\bparted\b',
        r'\bwipefs\b',
        r'\bcrontab\b\s+-r',
    ]

    for pattern in dangerous_patterns:
        if re.search(pattern, command_lower):
            logger.warning(f"[SECURITY] ❗ Perintah berbahaya terdeteksi: {command}")
            return True
    return False

# === Fungsi Mode: Pengamatan Shell dan Koreksi Error (untuk Telegram) ===
async def run_shell_observer_telegram(command_to_run: str, update: Update, context: CallbackContext):
    """
    Menjalankan perintah shell, memantau output, dan mengirim log/saran error ke Telegram.
    Non-interaktif.
    """
    chat_id = update.effective_chat.id
    user_context = get_user_context(chat_id)
    user_context["last_command_run"] = command_to_run
    user_context["full_error_output"] = []
    
    telegram_log_buffer = [] 
    async def send_telegram_chunk():
        nonlocal telegram_log_buffer
        if telegram_log_buffer:
            # Ini diformat sebagai blok kode dengan tag 'log'
            escaped_log_content = "\n".join(telegram_log_buffer)
            # Karena _escape_plaintext_markdown_v2 tidak meng-escape *, kita perlu manual
            # menambahkan tanda bintang untuk pemformatan di sini jika diinginkan.
            # Namun, untuk log, biarkan apa adanya dan Telegram akan menampilkannya sebagai blok kode.
            message = f"```log\n{escaped_log_content}\n```" 
            await kirim_ke_telegram(chat_id, context, message)
            telegram_log_buffer = []

    # Pesan awal saat menjalankan perintah. Command_to_run dimasukkan langsung ke backtick.
    await kirim_ke_telegram(chat_id, context, f"*⚙️ SHELL* Memulai perintah: `{command_to_run}`")
    logger.info(f"\n{COLOR_BLUE}[Shell] 🟢 Menjalankan perintah: `{command_to_run}`{COLOR_RESET}\n")

    # shlex.quote hanya digunakan untuk menjalankan perintah secara aman di shell, bukan untuk ditampilkan di Markdown.
    safe_command_to_run = shlex.quote(command_to_run)

    try:
        child = pexpect.spawn(f"bash -c {safe_command_to_run}", encoding='utf-8', timeout=None)
    except pexpect.exceptions.ExceptionPexpect as e:
        error_msg = f"*❗ ERROR SHELL* Gagal menjalankan perintah: `{str(e)}`. Pastikan perintah valid, bash tersedia, dan pexpect terinstal dengan benar."
        await kirim_ke_telegram(chat_id, context, error_msg)
        logger.error(f"[Shell] 🔴 Gagal menjalankan perintah: {e}")
        return ConversationHandler.END # Pastikan mengembalikan ConversationHandler.END jika ada error di sini

    error_detected_in_stream = False
    error_line_buffer = []
    user_context["last_error_log"] = None

    while True:
        try:
            line = await asyncio.to_thread(child.readline)

            if not line:
                if child.eof():
                    logger.info(f"{COLOR_GREEN}[Shell] ✅ Proses shell selesai.{COLOR_RESET}")
                    await send_telegram_chunk()
                    await kirim_ke_telegram(chat_id, context, f"*⚙️ SHELL* Perintah shell selesai.")
                    
                    if user_context["last_error_log"]:
                        await kirim_ke_telegram(chat_id, context, f"*❗ ERROR* Error terdeteksi pada eksekusi terakhir. Apakah Anda ingin men-debug program ini dengan bantuan AI? (Ya/Tidak)")
                        user_context["awaiting_debug_response"] = True
                        return DEBUGGING_STATE
                    break
                continue

            cleaned_line = line.strip()
            logger.info(f"{COLOR_YELLOW}[Shell Log] {cleaned_line}{COLOR_RESET}") # Logger yang lebih rapi
            
            telegram_log_buffer.append(cleaned_line)
            if len(telegram_log_buffer) >= 10:
                await send_telegram_chunk()

            error_line_buffer.append(cleaned_line)
            if len(error_line_buffer) > 10:
                error_line_buffer.pop(0)
            
            user_context["full_error_output"].append(cleaned_line)

            is_program_execution_command = bool(re.match(r"^(python|sh|bash|node|\./)\s+\S+\.(py|sh|js|rb|pl|php)", command_to_run, re.IGNORECASE))
            
            if is_program_execution_command and any(keyword in cleaned_line.lower() for keyword in ["error", "exception", "not found", "failed", "permission denied", "command not found", "no such file or directory", "segmentation fault", "fatal"]):
                if not error_detected_in_stream:
                    error_detected_in_stream = True
                    await send_telegram_chunk()
                    
                    user_context["last_error_log"] = "\n".join(user_context["full_error_output"])
                    
                    await kirim_ke_telegram(chat_id, context, f"*🧠 AI DEBUGGING* Error terdeteksi. Meminta saran AI...")
                    logger.info(f"{COLOR_RED}[AI] Error terdeteksi. Mengirim konteks ke model...{COLOR_RESET}\n")
                    
                    success_saran, saran = kirim_error_ke_llm_for_suggestion(user_context["last_error_log"], chat_id)

                    # Mendeteksi bahasa dari saran untuk syntax highlighting yang tepat
                    saran_lang = deteksi_bahasa_pemrograman_dari_konten(saran) if success_saran else "text"

                    if success_saran:
                        telegram_msg = f"""*❗ ERROR TERDETEKSI*
*Log Error Terbaru:*
```log
{user_context["full_error_output"][-2000:]}
```

---

*💡 SARAN AI*
```{saran_lang}
{saran}
```
"""
                    else:
                        telegram_msg = f"*❗ ERROR TERDETEKSI*\n*Log Error Terbaru:*\n```log\n{user_context['full_error_output'][-2000:]}\n```\n\n*🔴 ERROR AI* Gagal mendapatkan saran dari AI: {saran}"
                    
                    await kirim_ke_telegram(chat_id, context, telegram_msg)

        except pexpect.exceptions.EOF:
            logger.info(f"{COLOR_GREEN}[Shell] ✅ Proses shell selesai.{COLOR_RESET}")
            await send_telegram_chunk()
            await kirim_ke_telegram(chat_id, context, f"*⚙️ SHELL* Perintah shell selesai.")
            if user_context["last_error_log"]:
                await kirim_ke_telegram(chat_id, context, f"*❗ ERROR* Error terdeteksi pada eksekusi terakhir. Apakah Anda ingin men-debug program ini dengan bantuan AI? (Ya/Tidak)")
                user_context["awaiting_debug_response"] = True
                return DEBUGGING_STATE
            break
        except KeyboardInterrupt:
            logger.warning(f"\n{COLOR_YELLOW}[Shell] ✋ Diinterupsi oleh pengguna Termux.{COLOR_RESET}")
            child.sendline('\x03')
            child.close()
            await kirim_ke_telegram(chat_id, context, f"*⚙️ SHELL* Proses shell dihentikan secara manual.")
            break
        except Exception as e:
            error_msg = f"*🔴 ERROR INTERNAL* Terjadi kesalahan tak terduga pada `shell_observer`: `{str(e)}`"
            await kirim_ke_telegram(chat_id, context, error_msg)
            logger.error(f"[Shell] 🔴 Error tak terduga: {e}")
            if child.isalive():
                child.close()
            break
    
    return ConversationHandler.END

# === Penangan Perintah Telegram ===

async def start_command(update: Update, context: CallbackContext):
    """Mengirim pesan selamat datang ketika perintah /start diberikan."""
    chat_id = update.effective_chat.id
    pesan_raw = f"""
*Halo! Saya AI Asisten Shell & Kode Anda.*

Saya bisa membantu Anda dengan beberapa hal:

* *⚙️ SHELL*
    Jalankan perintah sistem atau operasi file. Cukup ketik perintah Anda atau instruksi alami (misal: `tampilkan isi direktori`).
* *✨ PROGRAM*
    Hasilkan atau perbaiki kode program. Cukup berikan instruksi kode (misal: `buatkan fungsi python untuk menghitung faktorial`, `bikinin program bash simple berfungsi sebagai kalkulator`, `tulis kode javascript untuk DOM`). Saya akan mendeteksi bahasanya.
* *💬 KONVERSASI*
    Ajukan pertanyaan umum atau mulai percakapan santai.

---

*Perintah Tambahan:*
* `/listfiles` - Melihat daftar file yang dihasilkan.
* `/deletefile <nama_file>` - Menghapus file yang dihasilkan.
* `/clear_chat` - Menghapus riwayat percakapan.

*Penting:* Pastikan bot saya berjalan di Termux dan semua variabel lingkungan sudah diatur!
    """
    await kirim_ke_telegram(chat_id, context, pesan_raw)
    logger.info(f"[Telegram] Pesan /start dikirim ke {chat_id}.")

async def handle_listfiles_command(update: Update, context: CallbackContext):
    """Menangani perintah /listfiles untuk menampilkan daftar file yang dihasilkan."""
    chat_id = update.effective_chat.id

    if str(chat_id) != TELEGRAM_CHAT_ID:
        await kirim_ke_telegram(chat_id, context, f"*❗ AKSES DITOLAK* Anda tidak diizinkan untuk menggunakan fitur ini. Hubungi admin bot.")
        logger.warning(f"[Auth] ⚠ Upaya akses tidak sah /listfiles dari {chat_id}.")
        return

    allowed_extensions = [
        '.py', '.sh', '.js', '.html', '.css', '.php', '.java', '.c', '.cpp',
        '.cs', '.rb', '.go', '.swift', '.kt', '.rs', '.txt'
    ]
    
    files = [f for f in os.listdir('.') if os.path.isfile(f) and any(f.endswith(ext) for ext in allowed_extensions) and f != os.path.basename(__file__)]
    
    if files:
        # Gunakan backtick untuk nama file agar terlihat seperti kode inline
        file_list_msg = "*📄 FILE SAYA* Daftar file program yang tersedia:\n" + "\n".join([f"- `{f}`" for f in files])
    else:
        file_list_msg = "*📄 FILE SAYA* Tidak ada file program yang dihasilkan oleh bot."
    
    await kirim_ke_telegram(chat_id, context, file_list_msg)
    logger.info(f"[Telegram] Daftar file dikirim ke {chat_id}.")

async def handle_deletefile_command(update: Update, context: CallbackContext):
    """Menangani perintah /deletefile untuk menghapus file tertentu."""
    chat_id = update.effective_chat.id
    filename_to_delete = " ".join(context.args).strip()

    if str(chat_id) != TELEGRAM_CHAT_ID:
        await kirim_ke_telegram(chat_id, context, f"*❗ AKSES DITOLAK* Anda tidak diizinkan untuk menggunakan fitur ini. Hubungi admin bot.")
        logger.warning(f"[Auth] ⚠ Upaya akses tidak sah /deletefile dari {chat_id}.")
        return

    if not filename_to_delete:
        await kirim_ke_telegram(chat_id, context, f"*❓ PERINTAH* Mohon berikan nama file yang ingin dihapus. Contoh: `/deletefile nama_program_anda.py`")
        return

    allowed_extensions = [
        '.py', '.sh', '.js', '.html', '.css', '.php', '.java', '.c', '.cpp',
        '.cs', '.rb', '.go', '.swift', '.kt', '.rs', '.txt'
    ]
    is_allowed_extension = any(filename_to_delete.endswith(ext) for ext in allowed_extensions)

    if not is_allowed_extension or filename_to_delete == os.path.basename(__file__):
        await kirim_ke_telegram(chat_id, context, f"*❗ DITOLAK* Hanya file program yang dihasilkan yang bisa dihapus. Anda tidak bisa menghapus file bot itu sendiri atau file dengan ekstensi yang tidak diizinkan.")
        logger.warning(f"[Security] ❗ Upaya menghapus file tidak valid: {filename_to_delete} dari {chat_id}")
        return

    try:
        if os.path.exists(filename_to_delete) and os.path.isfile(filename_to_delete):
            os.remove(filename_to_delete)
            await kirim_ke_telegram(chat_id, context, f"*✅ SUKSES* File `{filename_to_delete}` berhasil dihapus.")
            logger.info(f"[File] File {filename_to_delete} dihapus oleh {chat_id}.")
        else:
            await kirim_ke_telegram(chat_id, context, f"*❗ TIDAK DITEMUKAN* File `{filename_to_delete}` tidak ditemukan.")
    except Exception as e:
        await kirim_ke_telegram(chat_id, context, f"*🔴 ERROR* Gagal menghapus file `{filename_to_delete}`: `{str(e)}`")
        logger.error(f"[File] 🔴 Gagal menghapus file {filename_to_delete}: {e}")

async def handle_clear_chat_command(update: Update, context: CallbackContext):
    """Menangani perintah /clear_chat untuk menghapus riwayat percakapan."""
    chat_id = update.effective_chat.id
    if str(chat_id) != TELEGRAM_CHAT_ID:
        await kirim_ke_telegram(chat_id, context, f"*❗ AKSES DITOLAK* Anda tidak diizinkan untuk menggunakan fitur ini. Hubungi admin bot.")
        logger.warning(f"[Auth] ⚠ Upaya akses tidak sah /clear_chat dari {chat_id}.")
        return

    if chat_id in chat_histories:
        del chat_histories[chat_id]
        await kirim_ke_telegram(chat_id, context, f"*✅ SUKSES* Riwayat percakapan Anda telah dihapus.")
        logger.info(f"[Chat] Riwayat obrolan untuk {chat_id} dihapus.")
    else:
        await kirim_ke_telegram(chat_id, context, f"*💬 INFO* Tidak ada riwayat percakapan untuk dihapus.")


async def handle_text_message(update: Update, context: CallbackContext):
    """
    Menangani semua pesan teks non-perintah dari Telegram.
    Akan mendeteksi niat pengguna dan memanggil fungsi yang sesuai.
    """
    chat_id = update.effective_chat.id
    user_message = update.message.text.strip()
    user_context = get_user_context(chat_id)
    
    if str(chat_id) != TELEGRAM_CHAT_ID:
        await kirim_ke_telegram(chat_id, context, f"*❗ AKSES DITOLAK* Anda tidak diizinkan untuk berinteraksi dengan bot ini. Hubungi admin bot.")
        logger.warning(f"[Auth] ⚠ Upaya akses tidak sah dari {chat_id}: {user_message}")
        return

    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
    
    niat = deteksi_niat_pengguna(user_message)
    user_context["last_user_message_intent"] = niat
    logger.info(f"[Intent] Pengguna {chat_id} -> Niat: {niat}")

    if niat == "shell":
        await kirim_ke_telegram(chat_id, context, f"*⚙️ SHELL* Niat terdeteksi: Perintah Shell. Menerjemahkan instruksi: `{user_message}`")
        success_konversi, perintah_shell = konversi_ke_perintah_shell(user_message, chat_id)
        perintah_shell = perintah_shell.strip()

        if not success_konversi:
            await kirim_ke_telegram(chat_id, context, f"*🔴 ERROR KONVERSI* Terjadi masalah saat mengonversi perintah:\n```\n{perintah_shell}\n```")
            logger.error(f"[Error] Gagal Konversi: {perintah_shell}")
            user_context["last_ai_response_type"] = None
            user_context["last_generated_code"] = None
            user_context["last_generated_code_language"] = None
            return
        elif perintah_shell == "CANNOT_CONVERT":
            await kirim_ke_telegram(chat_id, context, f"*❗ PERINTAH TIDAK JELAS* Maaf, saya tidak dapat mengonversi instruksi tersebut menjadi perintah shell yang jelas. Harap berikan instruksi yang lebih spesifik.")
            user_context["last_ai_response_type"] = None
            user_context["last_generated_code"] = None
            user_context["last_generated_code_language"] = None
            return

        if is_command_dangerous(perintah_shell):
            await kirim_ke_telegram(chat_id, context, f"*🚫 DILARANG* Perintah ini tidak diizinkan untuk dieksekusi: `{perintah_shell}`. Harap gunakan perintah lain.")
            logger.warning(f"[Security] ❗ Upaya menjalankan perintah berbahaya: {perintah_shell} dari {chat_id}")
            user_context["last_ai_response_type"] = None
            user_context["last_generated_code"] = None
            user_context["last_generated_code_language"] = None
            return
        
        await kirim_ke_telegram(chat_id, context, f"*⚙️ SHELL* Perintah shell yang diterjemahkan: `{perintah_shell}`")
        user_context["last_ai_response_type"] = "shell"
        user_context["last_command_run"] = perintah_shell
        user_context["last_generated_code"] = None
        user_context["last_generated_code_language"] = None
        # Panggil run_shell_observer_telegram dan tangani return value untuk ConversationHandler
        return await run_shell_observer_telegram(perintah_shell, update, context)

    elif niat == "program":
        await kirim_ke_telegram(chat_id, context, f"*✨ PROGRAM* Niat terdeteksi: Pembuatan Program. Mulai menghasilkan kode untuk: `{user_message}`")
        
        target_lang_from_prompt = deteksi_bahasa_dari_prompt(user_message)

        success_code, kode_tergenerasi, detected_language = minta_kode(user_message, chat_id=chat_id, target_language=target_lang_from_prompt)

        if not success_code:
            await kirim_ke_telegram(chat_id, context, f"*🔴 ERROR GENERASI KODE* Terjadi masalah saat menghasilkan kode:\n```\n{kode_tergenerasi}\n```")
            logger.error(f"[Error] Gagal Gen Kode: {kode_tergenerasi}")
            user_context["last_ai_response_type"] = None
            user_context["last_generated_code"] = None
            user_context["last_generated_code_language"] = None
            return
        
        generated_file_name = generate_filename(user_message, detected_language)
        simpan_ok = simpan_ke_file(generated_file_name, kode_tergenerasi)

        if simpan_ok:
            user_context["last_generated_code"] = kode_tergenerasi
            user_context["last_generated_code_language"] = detected_language
            user_context["last_ai_response_type"] = "program"
            user_context["last_command_run"] = None

            await kirim_ke_telegram(chat_id, context, f"*✅ SUKSES* Kode *{detected_language.capitalize()}* berhasil dihasilkan dan disimpan ke `{generated_file_name}`.")
            await kirim_ke_telegram(chat_id, context, f"*Anda bisa membukanya di Termux dengan:* `nano {generated_file_name}`")
            
            run_command_suggestion = ""
            if detected_language == "python":
                run_command_suggestion = f"`python {generated_file_name}`"
            elif detected_language == "bash":
                run_command_suggestion = f"`bash {generated_file_name}` atau `chmod +x {generated_file_name} && ./{generated_file_name}`"
            elif detected_language == "javascript":
                run_command_suggestion = f"`node {generated_file_name}` (pastikan Node.js terinstal)"
            elif detected_language == "html":
                run_command_suggestion = f"Buka file ini di browser web Anda."
            elif detected_language == "php":
                run_command_suggestion = f"`php {generated_file_name}` (pastikan PHP terinstal)"
            elif detected_language == "java":
                run_command_suggestion = f"Kompres dengan `javac {generated_file_name}` lalu jalankan dengan `java {generated_file_name.replace('.java', '')}`"
            elif detected_language in ["c", "cpp"]:
                run_command_suggestion = f"Kompres dengan `gcc {generated_file_name} -o a.out` lalu jalankan dengan `./a.out`"
            
            if run_command_suggestion:
                await kirim_ke_telegram(chat_id, context, f"*Dan menjalankan dengan:* {run_command_suggestion}")

            await kirim_ke_telegram(chat_id, context, f"*📋 KODE TERGENERASI*\n```{detected_language}\n{kode_tergenerasi}\n```")
        else:
            await kirim_ke_telegram(chat_id, context, f"*🔴 ERROR FILE* Gagal menyimpan kode yang dihasilkan ke file.")
            user_context["last_ai_response_type"] = None
            user_context["last_generated_code"] = None
            user_context["last_generated_code_language"] = None
        return ConversationHandler.END # Akhiri percakapan setelah pembuatan program
            

    else: # niat == "konversasi"
        await kirim_ke_telegram(chat_id, context, f"*💬 KONVERSASI UMUM* Niat terdeteksi: Percakapan Umum. Meminta jawaban AI...")
        success_response, jawaban_llm = minta_jawaban_konversasi(chat_id, user_message)
        user_context["last_ai_response_type"] = "konversasi"
        user_context["last_command_run"] = None
        user_context["last_generated_code"] = None
        user_context["last_generated_code_language"] = None
        
        if not success_response:
            await kirim_ke_telegram(chat_id, context, f"*🔴 ERROR AI* Terjadi masalah saat memproses percakapan:\n```\n{jawaban_llm}\n```")
            logger.error(f"[Error] Gagal Konversasi: {jawaban_llm}")
        else:
            await kirim_ke_telegram(chat_id, context, f"*💬 RESPON AI*\n{jawaban_llm}")
        return ConversationHandler.END # Akhiri percakapan setelah konversasi umum


async def handle_unknown_command(update: Update, context: CallbackContext):
    """Menanggapi perintah yang tidak dikenal (misalnya, /foo bar)."""
    chat_id = update.effective_chat.id
    await kirim_ke_telegram(chat_id, context, f"*❓ TIDAK DIKENAL* Perintah tidak dikenal. Silakan gunakan `/start` untuk melihat perintah yang tersedia.")
    logger.warning(f"[Command] ⚠ Perintah tidak dikenal dari {chat_id}: {update.message.text}")

# === Penangan Percakapan Debugging ===
async def ask_for_debug_response(update: Update, context: CallbackContext):
    """Meminta respons Ya/Tidak dari pengguna untuk debugging."""
    chat_id = update.effective_chat.id
    user_context = get_user_context(chat_id)
    
    if user_context["awaiting_debug_response"]:
        user_response = update.message.text.strip().lower()
        if user_response in ["ya", "yes"]:
            await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
            await kirim_ke_telegram(chat_id, context, f"*🧠 AI DEBUGGING* Memulai sesi debugging...")
            logger.info(f"{COLOR_BLUE}[Debug] Memulai debugging untuk {chat_id}{COLOR_RESET}")
            
            error_log = user_context["last_error_log"]
            last_command = user_context["last_command_run"]
            last_generated_code_lang = user_context["last_generated_code_language"]

            if error_log:
                await kirim_ke_telegram(chat_id, context, f"*🧠 AI DEBUGGING* Meminta LLM untuk menganalisis error dan memberikan perbaikan/kode baru...")
                success_debug, debug_saran, debug_lang = minta_kode(prompt="", error_context=error_log, chat_id=chat_id, target_language=last_generated_code_lang)
                
                if not success_debug:
                    await kirim_ke_telegram(chat_id, context, f"*🔴 ERROR DEBUGGING* Terjadi masalah saat melakukan debugging:\n```\n{debug_saran}\n```")
                    logger.error(f"[Debug] Gagal Debug: {debug_saran}")
                else:
                    # Mencoba mengekstrak nama file dari perintah terakhir yang dieksekusi
                    debug_file_name = None
                    if last_command:
                        match = re.search(r"^(python|sh|bash|node|php|\./)\s+(\S+\.(py|sh|js|rb|pl|php|java|c|cpp|html|css|txt))", last_command, re.IGNORECASE)
                        if match:
                            debug_file_name = match.group(2)
                    
                    if not debug_file_name:
                        # Jika tidak dapat mengekstrak dari perintah, buat nama file baru
                        debug_file_name = generate_filename("bug_fix", debug_lang)


                    simpan_ok = simpan_ke_file(debug_file_name, debug_saran)
                    if simpan_ok:
                        user_context["last_generated_code"] = debug_saran
                        user_context["last_generated_code_language"] = debug_lang
                        user_context["last_ai_response_type"] = "program"
                        await kirim_ke_telegram(chat_id, context, f"*✅ SUKSES* AI telah menghasilkan perbaikan/kode baru ke `{debug_file_name}`.")
                        
                        run_command_suggestion = ""
                        if debug_lang == "python":
                            run_command_suggestion = f"`python {debug_file_name}`"
                        elif debug_lang == "bash":
                            run_command_suggestion = f"`bash {debug_file_name}` atau `chmod +x {debug_file_name} && ./{debug_file_name}`"
                        elif debug_lang == "javascript":
                            run_command_suggestion = f"`node {debug_file_name}` (pastikan Node.js terinstal)"
                        elif debug_lang == "html":
                            run_command_suggestion = f"Buka file ini di browser web Anda."
                        elif debug_lang == "php":
                            run_command_suggestion = f"`php {debug_file_name}` (pastikan PHP terinstal)"
                        elif debug_lang == "java":
                            run_command_suggestion = f"Kompres dengan `javac {debug_file_name}` lalu jalankan dengan `java {debug_file_name.replace('.java', '')}`"
                        elif debug_lang in ["c", "cpp"]:
                            run_command_suggestion = f"Kompres dengan `gcc {debug_file_name} -o a.out` lalu jalankan dengan `./a.out`"

                        if run_command_suggestion:
                            await kirim_ke_telegram(chat_id, context, f"*Silakan tinjau dan coba jalankan kembali dengan:* {run_command_suggestion}\n\n*📋 KODE PERBAIKAN*\n```{debug_lang}\n{debug_saran}\n```")
                        else:
                            await kirim_ke_telegram(chat_id, context, f"*Silakan tinjau dan coba jalankan kembali. *\n\n*📋 KODE PERBAIKAN*\n```{debug_lang}\n{debug_saran}\n```")

                    else:
                        await kirim_ke_telegram(chat_id, context, f"*🔴 ERROR FILE* Gagal menyimpan kode perbaikan yang dihasilkan ke file.")
                        user_context["last_generated_code"] = None
                        user_context["last_ai_response_type"] = None
                        user_context["last_generated_code_language"] = None
            else:
                await kirim_ke_telegram(chat_id, context, f"*💬 INFO* Tidak ada log error yang tersedia untuk debugging.")
            
            user_context["last_error_log"] = None
            user_context["last_command_run"] = None
            user_context["awaiting_debug_response"] = False
            user_context["full_error_output"] = []
            return ConversationHandler.END
        elif user_response in ["tidak", "no"]:
            await kirim_ke_telegram(chat_id, context, f"*💬 INFO* Debugging dibatalkan.")
            logger.info(f"{COLOR_GREEN}[Debug] Debugging dibatalkan oleh {chat_id}{COLOR_RESET}")
            user_context["last_error_log"] = None
            user_context["last_command_run"] = None
            user_context["awaiting_debug_response"] = False
            user_context["full_error_output"] = []
            return ConversationHandler.END
        else:
            await kirim_ke_telegram(chat_id, context, f"*❓ RESPON INVALID* Mohon jawab 'Ya' atau 'Tidak'.")
            return DEBUGGING_STATE
    else:
        # Jika bukan dalam keadaan debugging, lewati ke handle_text_message biasa
        return await handle_text_message(update, context)


def main():
    """Fungsi utama untuk memulai bot Telegram."""
    if not TELEGRAM_BOT_TOKEN:
        logger.error(f"ERROR: TELEGRAM_BOT_TOKEN tidak diatur. Harap atur variabel lingkungan atau masukkan langsung.")
        return
    if not TELEGRAM_CHAT_ID:
        logger.error(f"ERROR: TELEGRAM_CHAT_ID tidak diatur. Harap atur variabel lingkungan atau masukkan langsung.")
        return
    if not OPENROUTER_API_KEY:
        logger.error(f"ERROR: OPENROUTER_API_KEY tidak diatur. Harap atur variabel lingkungan atau masukkan langsung.")
        return

    logger.info(f"{COLOR_GREEN}Memulai Bot Telegram...{COLOR_RESET}")
    logger.info(f"Menggunakan TOKEN: {'*' * (len(TELEGRAM_BOT_TOKEN) - 5) + TELEGRAM_BOT_TOKEN[-5:] if len(TELEGRAM_BOT_TOKEN) > 5 else TELEGRAM_BOT_TOKEN}")
    logger.info(f"ID Obrolan yang Diizinkan: {TELEGRAM_CHAT_ID}")

    # Bangun Aplikasi dengan JobQueue tanpa tzinfo eksplisit di konstruktornya
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).job_queue(JobQueue()).build()

    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("listfiles", handle_listfiles_command))
    application.add_handler(CommandHandler("deletefile", handle_deletefile_command))
    application.add_handler(CommandHandler("clear_chat", handle_clear_chat_command))
    
    conv_handler = ConversationHandler(
        entry_points=[MessageHandler(filters.TEXT & ~filters.COMMAND, ask_for_debug_response)],
        states={
            DEBUGGING_STATE: [MessageHandler(filters.TEXT & ~filters.COMMAND, ask_for_debug_response)],
        },
        fallbacks=[CommandHandler('cancel', lambda update, context: ConversationHandler.END)]
    )
    application.add_handler(conv_handler)
    
    # Handler pesan teks harus ditempatkan setelah ConversationHandler agar ConversationHandler memiliki prioritas
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message))
    
    application.add_handler(MessageHandler(filters.COMMAND, handle_unknown_command))

    logger.info(f"{COLOR_GREEN}Bot sedang berjalan. Tekan Ctrl+C untuk berhenti.{COLOR_RESET}")
    # Gunakan run_polling secara langsung, karena ia mengelola loop event-nya sendiri
    application.run_polling(allowed_updates=Update.ALL_TYPES)
    logger.info(f"{COLOR_GREEN}Bot berhenti.{COLOR_RESET}")

if __name__ == "__main__":
    main() # Panggil main secara sinkron
