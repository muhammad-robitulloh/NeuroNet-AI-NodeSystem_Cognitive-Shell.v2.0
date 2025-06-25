import pexpect
import requests
import re
import os
import time
import subprocess
import logging
import shlex # Untuk sanitasi perintah shell
from telegram import Update, ChatAction # Import ChatAction untuk indikator mengetik
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext, ConversationHandler
from dotenv import load_dotenv # Import untuk memuat variabel lingkungan dari .env

# Muat variabel lingkungan dari file .env
load_dotenv()

# === Konfigurasi Global ===
# Sangat disarankan untuk mengatur variabel lingkungan ini untuk keamanan.
# Pastikan Anda telah mengatur variabel lingkungan ini, atau nilai default akan digunakan.
# Contoh di Termux:
# export OPENROUTER_API_KEY="sk-or-v1-YOUR_OPENROUTER_API_KEY_DI_SINI"
# export TELEGRAM_CHAT_ID="YOUR_TELEGRAM_CHAT_ID_DI_SINI" # ID chat Telegram Anda (dapatkan dari @userinfobot)
# export TELEGRAM_BOT_TOKEN="YOUR_TELEGRAM_BOT_TOKEN_DI_SINI" # Token dari BotFather

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "sk-or-v1 OPENROUTER_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1/chat/completions")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "CHAT_ID_TELEGRAM") # ID chat yang diizinkan untuk interaksi
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "BOT_TOKEN")

# Konfigurasi model LLM untuk tugas yang berbeda
CODE_GEN_MODEL = os.getenv("CODE_GEN_MODEL", "moonshotai/kimi-dev-72b:free")
ERROR_FIX_MODEL = os.getenv("ERROR_FIX_MODEL", "nvidia/llama-3.3-nemotron-super-49b-v1:free") # Menggunakan model yang sama untuk konsistensi
CONVERSATION_MODEL = os.getenv("CONVERSATION_MODEL", "mistralai/mistral-small-3.2-24b-instruct") # Model untuk percakapan umum
COMMAND_CONVERSION_MODEL = os.getenv("COMMAND_CONVERSION_MODEL", "nvidia/llama-3.3-nemotron-super-49b-v1:free") # Model untuk konversi bahasa natural ke perintah shell
FILENAME_GEN_MODEL = os.getenv("FILENAME_GEN_MODEL", "mistralai/mistral-small-3.2-24b-instruct") # Model untuk generasi nama file
INTENT_DETECTION_MODEL = os.getenv("INTENT_DETECTION_MODEL", "mistralai/mistral-small-3.2-24b-instruct") # Model baru untuk deteksi niat

# Warna ANSI untuk output konsol Termux (hanya untuk log internal, tidak dikirim ke Telegram)
COLOR_GREEN = "\033[92m"
COLOR_YELLOW = "\033[93m"
COLOR_RED = "\033[91m"
COLOR_BLUE = "\033[94m"
COLOR_PURPLE = "\033[95m"
COLOR_RESET = "\033[0m"

# State untuk ConversationHandler (debugging)
DEBUGGING_STATE = 1

# --- Konfigurasi Logging ---
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# --- Fungsi Global untuk Penyimpanan Konteks ---
user_contexts = {} # Menyimpan konteks error dan interaksi terakhir untuk setiap pengguna
chat_histories = {} # Menyimpan riwayat percakapan per pengguna

def get_user_context(chat_id):
    """Mendapatkan konteks pengguna. Menginisialisasi jika belum ada."""
    if chat_id not in user_contexts:
        user_contexts[chat_id] = {
            "last_error_log": None,
            "last_command_run": None, # Perintah shell terakhir yang dijalankan
            "last_generated_code": None, # Kode terakhir yang dihasilkan
            "awaiting_debug_response": False,
            "full_error_output": [], # Menyimpan output lengkap untuk debugging
            "last_user_message_intent": None, # Menyimpan niat dari pesan terakhir pengguna
            "last_ai_response_type": None # Menyimpan tipe respons AI terakhir (shell, program, konversasi)
        }
    return user_contexts[chat_id]

def get_chat_history(chat_id):
    """Mendapatkan riwayat chat pengguna. Menginisialisasi jika belum ada."""
    if chat_id not in chat_histories:
        chat_histories[chat_id] = []
    return chat_histories[chat_id]

def escape_telegram_markdown_v1(text: str) -> str:
    """
    Mengubah karakter khusus Markdown V1 menjadi karakter yang di-escape.
    Karakter yang perlu di-escape: _, *, `, [,
    Ini penting untuk mencegah error parsing Markdown di Telegram
    jika respons LLM mengandung karakter ini secara tidak terduga di luar blok kode.
    """
    special_chars = ['_', '*', '`', '[']
    # Urutan penting: escape backticks dulu karena mereka digunakan untuk kode
    text = text.replace('`', '\\`') 
    for char in special_chars:
        if char != '`': # Hindari mengulang backtick
            text = text.replace(char, f'\\{char}')
    return text

# === Fungsi Umum: Memanggil LLM ===
def call_llm(messages, model, api_key, max_tokens=512, temperature=0.7, extra_headers=None):
    """
    Fungsi generik untuk mengirim permintaan ke model LLM (OpenRouter).
    Mengembalikan tuple: (True, hasil) jika berhasil, (False, pesan_error) jika gagal.
    """
    if not api_key or not LLM_BASE_URL:
        logger.error("[LLM ERROR] API Key atau Base URL LLM tidak diatur.")
        return False, f"API Key atau Base URL LLM tidak diatur. Harap periksa konfigurasi."

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
        res = requests.post(LLM_BASE_URL, json=payload, headers=headers)
        res.raise_for_status() # Akan memunculkan HTTPError untuk status kode 4xx/5xx
        data = res.json()
        if "choices" in data and data["choices"]:
            return True, data["choices"][0]["message"]["content"]
        else:
            logger.error(f"[LLM ERROR] Respon LLM tidak mengandung 'choices'. Debug respon: {data}")
            return False, f"Respon LLM tidak sesuai format yang diharapkan."
    except requests.exceptions.RequestException as e:
        logger.error(f"[LLM ERROR] Gagal terhubung ke LLM API ({LLM_BASE_URL}): {e}")
        return False, f"Gagal terhubung ke LLM API: {e}"
    except KeyError:
        logger.error(f"[LLM ERROR] Respon LLM tidak sesuai format yang diharapkan (missing 'choices' or 'message').")
        return False, f"Respon LLM tidak sesuai format yang diharapkan."
    except Exception as e:
        logger.error(f"[LLM ERROR] Terjadi kesalahan tidak terduga saat memanggil LLM: {e}")
        return False, f"Terjadi kesalahan tidak terduga saat memanggil LLM: {e}"

# === Fungsi Baru: Mengekstrak Kode dari Respon LLM ===
def ekstrak_kode_dari_llm(text_response):
    """
    Mengekstrak blok kode Python dari respons LLM.
    Mencari format Markdown ```python ... ``` dan mengembalikan isinya.
    Jika tidak ditemukan, akan mencoba mengembalikan respons secara keseluruhan,
    namun dengan filtering tambahan untuk membuang teks non-kode di awal/akhir.
    """
    # Regex untuk mencari blok kode Markdown Python
    code_block_pattern = r"```python\n(.*?)```"
    match = re.search(code_block_pattern, text_response, re.DOTALL)
    
    if match:
        return match.group(1).strip()
    else:
        # Fallback yang lebih agresif untuk membersihkan teks non-kode.
        # Membuang baris yang jelas-jelas bukan kode (misal: kalimat pengantar, penutup).
        lines = text_response.strip().split('\n')
        cleaned_lines = []
        in_code_block = False # Tambahkan flag untuk mendeteksi potensi blok kode tanpa format Markdown
        
        for line in lines:
            stripped_line = line.strip()
            # Jika baris adalah komentar atau import, mungkin itu kode
            if stripped_line.startswith(('#', 'import ', 'from ')):
                cleaned_lines.append(line)
                in_code_block = True
            # Jika baris adalah definisi fungsi, kelas, atau kontrol flow, itu kode
            elif re.match(r'^(def|class|if|for|while|try|with)\s+', stripped_line):
                cleaned_lines.append(line)
                in_code_block = True
            # Jika baris berisi `=` atau `(`, `)`, kemungkinan kode
            elif '=' in stripped_line or '(' in stripped_line or ')' in stripped_line:
                cleaned_lines.append(line)
                in_code_block = True
            # Jika sedang di dalam blok kode, tambahkan baris kosong juga (untuk format)
            elif in_code_block and not stripped_line:
                cleaned_lines.append(line)
            # Batasi penalaran yang masuk ke file: hanya baris yang tampak seperti kode
            # Jika ada baris panjang yang tidak tampak seperti kode (misal: penjelasan), abaikan
            elif len(stripped_line) > 0 and not re.match(r'^[a-zA-Z\s,;.:-]*$', stripped_line): # Coba hindari kalimat biasa
                 cleaned_lines.append(line)
                 in_code_block = True
            else:
                # Jika tidak terdeteksi sebagai bagian dari kode, dan sebelumnya ada kode, hentikan
                if in_code_block and stripped_line: # Jika ada teks non-kosong setelah kode
                    break 
                # Jika belum dalam blok kode, dan baris ini tidak terlihat seperti kode, abaikan
                pass

        # Hapus baris kosong di awal atau akhir dari hasil akhir
        final_code = "\n".join(cleaned_lines).strip()
        # Jika hasil akhir masih mengandung "```" yang tersisa (misal, kalau model kasih ``` tapi tanpa bahasa), buang
        final_code = re.sub(r'```(.*?)```', r'\1', final_code, flags=re.DOTALL)
        return final_code.strip()


# === Fungsi Khusus: Mendeteksi Niat Pengguna ===
def deteksi_niat_pengguna(pesan_pengguna):
    """
    Mendeteksi niat pengguna (jalankan perintah shell, buat program, atau percakapan umum).
    Mengembalikan string: "shell", "program", atau "konversasi".
    """
    messages = [
        {"role": "system", "content": """Anda adalah detektor niat. Identifikasi apakah pesan pengguna bermaksud:
- "shell": Jika pengguna ingin menjalankan perintah sistem atau melakukan operasi file (contoh: "hapus file", "tampilkan direktori", "jalankan", "buka", "instal", "kompres").
- "program": Jika pengguna ingin membuat atau memperbaiki kode program (contoh: "buatkan fungsi python", "tulis kode javascript", "perbaiki error ini", "tulis program").
- "konversasi": Untuk semua jenis pertanyaan atau interaksi lainnya yang bukan perintah langsung atau pembuatan kode.

Hanya kembalikan satu kata dari kategori di atas. Jangan berikan penjelasan tambahan.
"""},
        {"role": "user", "content": f"Deteksi niat untuk: '{pesan_pengguna}'"}
    ]
    logger.info(f"{COLOR_BLUE}[🧠] Mendeteksi niat pengguna untuk '{pesan_pengguna}' ({INTENT_DETECTION_MODEL})...{COLOR_RESET}\n")
    
    success, niat = call_llm(messages, INTENT_DETECTION_MODEL, OPENROUTER_API_KEY, max_tokens=10, temperature=0.0) # Temperature rendah untuk hasil yang konsisten
    
    if success:
        niat_cleaned = niat.strip().lower()
        if not niat_cleaned: # Menangani kasus LLM mengembalikan string kosong
            logger.warning(f"Niat kosong dari LLM. Default ke 'konversasi'.")
            return "konversasi"
        elif niat_cleaned in ["shell", "program", "konversasi"]:
            return niat_cleaned
        else:
            logger.warning(f"Niat tidak dikenal dari LLM: '{niat_cleaned}'. Default ke 'konversasi'.")
            return "konversasi"
    else:
        logger.error(f"Gagal mendeteksi niat: {niat}. Default ke 'konversasi'.")
        return "konversasi"


# === Fungsi Khusus: Meminta Kode dari LLM ===
def minta_kode(prompt, error_context=None, chat_id=None):
    """
    Meminta LLM untuk menghasilkan kode Python berdasarkan prompt.
    Jika error_context diberikan, ini adalah permintaan debugging.
    Menyertakan konteks percakapan terakhir jika tersedia.
    """
    messages = []
    
    # Tambahkan riwayat chat sebagai konteks
    history = get_chat_history(chat_id) if chat_id else []
    # Ambil 5 pasangan pesan terakhir (10 pesan)
    recent_history = history[-10:] 
    for msg in recent_history:
        messages.append(msg)

    if error_context:
        messages.append({
                "role": "system", 
                "content": "Anda adalah asisten AI yang ahli dalam memperbaiki kode Python. Berdasarkan log error yang diberikan dan riwayat percakapan, berikan *hanya* kode Python lengkap yang telah diperbaiki atau kode baru. Sertakan semua import yang diperlukan. JANGAN tambahkan penjelasan atau teks tambahan di luar blok kode Markdown. Format respons Anda sebagai blok kode Markdown Python."
            })
        messages.append({
                "role": "user",
                "content": f"Ada error dalam menjalankan kode/perintah:\n\n{error_context}\n\nPerbaiki atau berikan kode Python lengkap yang baru. Pastikan kode dapat dijalankan langsung."
            })
        logger.info(f"{COLOR_BLUE}[🤖] Meminta perbaikan/kode baru dari model AI ({CODE_GEN_MODEL}) berdasarkan error...{COLOR_RESET}\n")
    else:
        messages.append({
                "role": "system", 
                "content": "Anda adalah asisten AI coding. Buatkan *hanya* kode Python lengkap. Jika ada bagian yang perlu diisi pengguna, berikan komentar yang jelas di dalam kode. Sertakan semua import yang diperlukan. JANGAN tambahkan penjelasan atau teks tambahan di luar blok kode Markdown. Format respons Anda sebagai blok kode Markdown Python."
            })
        messages.append({
                "role": "user",
                "content": f"Instruksi: {prompt}"
            })
        logger.info(f"{COLOR_BLUE}[🤖] Meminta kode dari model AI ({CODE_GEN_MODEL})...{COLOR_RESET}\n")
    
    success, response_content = call_llm(messages, CODE_GEN_MODEL, OPENROUTER_API_KEY, max_tokens=1024, temperature=0.7)

    if success:
        # Ekstrak hanya kode dari respons LLM
        cleaned_code = ekstrak_kode_dari_llm(response_content)
        return True, cleaned_code
    else:
        return False, response_content # Kembalikan pesan error jika LLM call gagal


# === Fungsi Khusus: Konversi Bahasa Natural ke Perintah Shell ===
def konversi_ke_perintah_shell(bahasa_natural, chat_id=None):
    """
    Mengonversi bahasa natural pengguna menjadi perintah shell yang dapat dijalankan.
    Menyertakan konteks percakapan terakhir jika tersedia.
    """
    messages = []

    # Tambahkan riwayat chat sebagai konteks
    history = get_chat_history(chat_id) if chat_id else []
    recent_history = history[-10:] 
    for msg in recent_history:
        messages.append(msg)

    messages.append({"role": "system", "content": "Anda adalah penerjemah bahasa alami ke perintah shell. Konversi instruksi bahasa alami berikut menjadi satu baris perintah shell Linux Termux yang paling relevan. Jangan berikan penjelasan, hanya perintahnya saja. Jika instruksinya tidak jelas atau tidak dapat diubah menjadi perintah shell, jawab dengan 'TIDAK_DAPAT_DIKONVERSI'."})
    messages.append({"role": "user", "content": f"Konversi ini ke perintah shell: {bahasa_natural}"})

    logger.info(f"{COLOR_BLUE}[🧠] Mengonversi bahasa natural ke perintah shell ({COMMAND_CONVERSION_MODEL})...{COLOR_RESET}\n")
    return call_llm(messages, COMMAND_CONVERSION_MODEL, OPENROUTER_API_KEY, max_tokens=128, temperature=0.3)


# === Fungsi Khusus: Menggenerasi Nama File ===
def generate_filename(prompt):
    """
    Menggenerasi nama file yang relevan berdasarkan prompt pengguna.
    """
    messages = [
        {"role": "system", "content": "Anda adalah generator nama file. Berikan satu nama file Python yang singkat, relevan, dan deskriptif (tanpa spasi, gunakan underscore, semua huruf kecil, tanpa ekstensi .py) berdasarkan deskripsi kode berikut. Contoh: 'fungsi_faktorial' atau 'game_ular'. Jangan berikan penjelasan, hanya nama filenya."},
        {"role": "user", "content": f"Deskripsi kode: {prompt}"}
    ]
    logger.info(f"{COLOR_BLUE}[🧠] Menggenerasi nama file untuk '{prompt}' ({FILENAME_GEN_MODEL})...{COLOR_RESET}\n")
    
    success, filename = call_llm(messages, FILENAME_GEN_MODEL, OPENROUTER_API_KEY, max_tokens=20, temperature=0.5)
    
    if not success:
        logger.warning(f"Gagal menghasilkan nama file dari LLM: {filename}. Menggunakan nama default.")
        return "generated_code.py"

    # Bersihkan nama file dari karakter yang tidak valid dan pastikan tidak ada .py
    filename = filename.strip()
    filename = re.sub(r'[^\w-]', '', filename).lower().replace(' ', '_').replace('.py', '')
    if not filename: # Fallback jika LLM gagal menghasilkan nama
        filename = "generated_code"
    return filename + ".py"


# === Fungsi Khusus: Kirim Error ke LLM untuk Saran Perbaikan ===
def kirim_error_ke_llm_for_suggestion(log_error, chat_id=None):
    """
    Mengirim log error ke LLM untuk mendapatkan saran perbaikan.
    Menyertakan konteks percakapan terakhir jika tersedia.
    """
    messages = []

    # Tambahkan riwayat chat sebagai konteks
    history = get_chat_history(chat_id) if chat_id else []
    recent_history = history[-10:] 
    for msg in recent_history:
        messages.append(msg)

    messages.append({"role": "user", "content": f"Ada error berikut:\n\n{log_error}\n\nApa saran terbaik untuk memperbaikinya dalam konteks sistem Linux Termux? Berikan saran dalam format yang bisa langsung dijalankan di shell jika memungkinkan, atau dalam blok kode Markdown. Jika tidak, berikan penjelasan singkat."})
    
    headers = {"HTTP-Referer": "[https://t.me/dseAI_bot](https://t.me/dseAI_bot)"} # Referer untuk OpenRouter
    
    logger.info(f"{COLOR_BLUE}[🧠] Mengirim error ke model AI ({ERROR_FIX_MODEL}) untuk saran...{COLOR_RESET}\n")
    return call_llm(messages, ERROR_FIX_MODEL, OPENROUTER_API_KEY, max_tokens=512, temperature=0.7, extra_headers=headers)

# === Fungsi Khusus: Meminta Jawaban Konversasi Umum dari LLM ===
def minta_jawaban_konversasi(chat_id, prompt):
    """
    Meminta LLM untuk jawaban konversasi umum, dengan menyimpan riwayat
    dan menyertakan rujukan dari interaksi sebelumnya (kode, perintah).
    """
    history = get_chat_history(chat_id)
    user_context = get_user_context(chat_id)
    
    # Bangun pesan sistem untuk memberikan konteks dari interaksi sebelumnya
    system_context_messages = []

    if user_context["last_command_run"] and user_context["last_ai_response_type"] == "shell":
        system_context_messages.append(
            {"role": "system", "content": f"Pengguna baru saja menjalankan perintah shell: `{user_context['last_command_run']}`. Pertimbangkan konteks ini dalam jawaban Anda."}
        )
    if user_context["last_generated_code"] and user_context["last_ai_response_type"] == "program":
        system_context_messages.append(
            {"role": "system", "content": f"Pengguna baru saja menerima kode Python berikut:\n```python\n{user_context['last_generated_code']}\n```. Pertimbangkan konteks ini dalam jawaban Anda."}
        )
    if user_context["last_error_log"] and user_context["last_user_message_intent"] == "shell": # Jika error dari perintah shell
        system_context_messages.append(
            {"role": "system", "content": f"Pengguna mengalami error setelah menjalankan perintah: `{user_context['last_command_run']}` dengan log error:\n```\n{user_context['last_error_log'][-500:]}\n```. Pertimbangkan ini dalam jawaban Anda."}
        )
    elif user_context["last_error_log"] and user_context["last_user_message_intent"] == "program": # Jika error dari program
         system_context_messages.append(
            {"role": "system", "content": f"Pengguna mengalami error setelah berinteraksi dengan program:\n```\n{user_context['last_error_log'][-500:]}\n```. Pertimbangkan ini dalam jawaban Anda."}
        )


    # Gabungkan pesan sistem kontekstual dengan riwayat percakapan utama
    messages_to_send = []
    messages_to_send.extend(system_context_messages) # Tambahkan konteks sistem terlebih dahulu

    # Tambahkan riwayat chat (dibatasi) ke pesan
    # Batasi panjang riwayat untuk mencegah token overruns
    # Misalnya, simpan 10 pesan terakhir (5 user, 5 assistant)
    max_history_length = 10
    recent_history = history[-max_history_length:]
    messages_to_send.extend(recent_history)
    
    # Tambahkan pesan pengguna saat ini
    messages_to_send.append({"role": "user", "content": prompt})

    logger.info(f"{COLOR_BLUE}[💬] Meminta jawaban konversasi dari model AI ({CONVERSATION_MODEL})...{COLOR_RESET}\n")
    success, response = call_llm(messages_to_send, CONVERSATION_MODEL, OPENROUTER_API_KEY, max_tokens=256, temperature=0.7)

    if success:
        history.append({"role": "user", "content": prompt}) # Tambahkan prompt pengguna ke riwayat
        history.append({"role": "assistant", "content": response}) # Tambahkan respons asisten ke riwayat
        chat_histories[chat_id] = history # Simpan riwayat yang diperbarui
    return success, response


# === Fungsi: Simpan ke file ===
def simpan_ke_file(nama_file, isi):
    """
    Menyimpan isi string ke dalam file.
    Mengembalikan True jika berhasil, False jika gagal.
    """
    try:
        with open(nama_file, "w") as f:
            f.write(isi)
        logger.info(f"{COLOR_GREEN}📄 Kode berhasil disimpan ke file: {nama_file}{COLOR_RESET}")
        return True
    except IOError as e:
        logger.error(f"[ERROR] Gagal menyimpan file {nama_file}: {e}")
        return False

# === Fungsi: Kirim notifikasi ke Telegram ===
def kirim_ke_telegram(chat_id, context: CallbackContext, pesan):
    """
    Mengirim pesan notifikasi ke Telegram.
    Pesan akan diformat untuk tampilan Telegram, tidak ada warna ANSI.
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning(f"[Telegram WARN] Token BOT atau Chat ID Telegram tidak ditemukan. Notifikasi tidak dikirim.")
        return

    # Menghilangkan kode warna ANSI sebelum mengirim ke Telegram
    pesan_bersih = re.sub(r'\033\[[0-9;]*m', '', pesan)

    try:
        context.bot.send_message(chat_id=chat_id, text=pesan_bersih, parse_mode="Markdown")
        logger.info(f"[Telegram] Notifikasi berhasil dikirim ke {chat_id}.")
    except Exception as e:
        logger.error(f"[Telegram ERROR] Gagal mengirim pesan ke Telegram: {e}")

# === Fungsi: Deteksi perintah shell dalam saran AI ===
def deteksi_perintah_shell(saran_ai):
    """
    Mendeteksi baris perintah shell dari saran AI, termasuk yang ada
    dalam blok kode Markdown atau kutipan inline.
    Prioritas: Blok kode Markdown > Kutipan Inline > Pola Regex Biasa
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
    Memeriksa apakah perintah shell mengandung keyword yang dilarang.
    Menggunakan regex untuk mencocokkan kata kunci sebagai kata utuh
    atau frasa tertentu, mengabaikan spasi dan argumen.
    """
    command_lower = command.lower()
    
    # regex \b untuk mencocokkan batas kata
    # \s* untuk mencocokkan nol atau lebih spasi
    dangerous_patterns = [
        r'\brm\b\s+-rf', # "rm -rf"
        r'\brm\b\s+/\s*', # "rm /"
        r'\bpkg\s+uninstall\b', # "pkg uninstall" sebagai frasa
        r'\bmv\b\s+/\s*', # "mv /"
        r'\bchown\b\s+root', # "chown root"
        r'\bchmod\b\s+\d{3}\s+/\s*', # "chmod 000 /"
    ]

    for pattern in dangerous_patterns:
        if re.search(pattern, command_lower):
            logger.warning(f"[SECURITY ALERT] Dangerous command detected: {command}")
            return True
    return False

# === Fungsi Mode: Observasi Shell dan Koreksi Error (untuk Telegram) ===
def run_shell_observer_telegram(command_to_run: str, update: Update, context: CallbackContext):
    """
    Menjalankan perintah shell, memantau output, dan mengirim log/saran error ke Telegram.
    Tidak interaktif.
    """
    chat_id = update.effective_chat.id
    user_context = get_user_context(chat_id)
    user_context["last_command_run"] = command_to_run # Simpan perintah yang dijalankan
    user_context["full_error_output"] = [] # Reset full error output
    
    # Buffer untuk mengumpulkan output sebelum dikirim ke Telegram agar tidak terlalu banyak pesan
    telegram_log_buffer = [] 
    def send_telegram_chunk():
        nonlocal telegram_log_buffer
        if telegram_log_buffer:
            message = "*[LOG SHELL]*\n```\n" + "\n".join(telegram_log_buffer) + "\n```"
            kirim_ke_telegram(chat_id, context, message)
            telegram_log_buffer = [] # Kosongkan buffer setelah dikirim

    kirim_ke_telegram(chat_id, context, f"*{COLOR_BLUE}Menjalankan perintah: `{shlex.quote(command_to_run)}`{COLOR_RESET}*")
    logger.info(f"\n{COLOR_BLUE}[🟢] Menjalankan perintah Telegram: `{command_to_run}`{COLOR_RESET}\n")

    # Sanitasi perintah untuk eksekusi shell yang lebih aman
    # Meskipun pexpect.spawn dengan bash -c sudah menyediakan isolasi, shlex.quote adalah lapisan tambahan
    safe_command_to_run = shlex.quote(command_to_run)

    try:
        child = pexpect.spawn(f"bash -c {safe_command_to_run}", encoding='utf-8', timeout=None)
    except pexpect.exceptions.ExceptionPexpect as e:
        error_msg = f"*{COLOR_RED}[ERROR] Gagal menjalankan perintah:* {e}. Pastikan perintah valid, bash tersedia, dan pexpect terinstal dengan benar."
        kirim_ke_telegram(chat_id, context, error_msg)
        logger.error(error_msg)
        return

    error_detected_in_stream = False # Flag untuk deteksi error saat streaming
    error_line_buffer = [] # Menyimpan beberapa baris terakhir untuk konteks error LLM
    user_context["last_error_log"] = None # Reset error log sebelum menjalankan perintah baru

    while True:
        try:
            line = child.readline()
            if not line:
                if child.eof():
                    logger.info(f"{COLOR_GREEN}[✅] Proses shell selesai.{COLOR_RESET}")
                    send_telegram_chunk() # Kirim sisa log jika ada
                    kirim_ke_telegram(chat_id, context, f"*{COLOR_GREEN}Perintah shell selesai.{COLOR_RESET}*")
                    
                    # Setelah perintah selesai, cek apakah ada error dan tawarkan debugging
                    if user_context["last_error_log"]: # Jika ada error yang disimpan dari stream
                        kirim_ke_telegram(chat_id, context, f"*{COLOR_YELLOW}Terdeteksi error pada eksekusi terakhir. Apakah Anda ingin mencoba melakukan debugging pada program ini dengan bantuan AI? (Ya/Tidak){COLOR_RESET}*")
                        user_context["awaiting_debug_response"] = True
                        return DEBUGGING_STATE # Mengubah state handler
                    break
                continue

            cleaned_line = line.strip()
            logger.info(f"{COLOR_YELLOW}[📤 LOG]{COLOR_RESET} {cleaned_line}")
            
            # Tambahkan ke buffer Telegram
            telegram_log_buffer.append(cleaned_line)
            if len(telegram_log_buffer) >= 10: # Kirim setiap 10 baris atau lebih
                send_telegram_chunk()

            # Tambahkan baris ke buffer konteks error
            error_line_buffer.append(cleaned_line)
            if len(error_line_buffer) > 10: # Simpan 10 baris terakhir untuk konteks cepat
                error_line_buffer.pop(0)
            
            # Selalu tambahkan ke full_error_output jika program sedang dieksekusi
            user_context["full_error_output"].append(cleaned_line)

            # Deteksi kata kunci error (kasus-insensitive)
            # Memastikan error hanya dideteksi untuk program yang dijalankan (misal: "python script.py", "sh script.sh")
            is_program_execution_command = bool(re.match(r"^(python|sh|bash|\./)\s+\S+\.(py|sh|js|rb|pl|php)", command_to_run, re.IGNORECASE))
            
            if is_program_execution_command and any(keyword in cleaned_line.lower() for keyword in ["error", "exception", "not found", "failed", "permission denied", "command not found", "no such file or directory", "segmentation fault", "fatal"]):
                if not error_detected_in_stream: # Pastikan hanya mengirim satu kali per deteksi error awal
                    error_detected_in_stream = True
                    send_telegram_chunk() # Kirim log yang tertunda sebelum pesan error
                    
                    # Simpan error log lengkap yang terakumulasi
                    user_context["last_error_log"] = "\n".join(user_context["full_error_output"])
                    
                    kirim_ke_telegram(chat_id, context, f"*{COLOR_RED}[🧠] Error terdeteksi. Meminta saran AI...{COLOR_RESET}*")
                    logger.info(f"{COLOR_RED}[🧠] Error terdeteksi. Mengirim konteks ke model...{COLOR_RESET}\n")
                    
                    success_saran, saran = kirim_error_ke_llm_for_suggestion(user_context["last_error_log"], chat_id)

                    if success_saran:
                        telegram_msg = f"""*{COLOR_RED}🧠 Error Terdeteksi:*{COLOR_RESET}
```
{user_context["last_error_log"][-2000:]} # Potong untuk menghindari pesan terlalu panjang
```

*{COLOR_BLUE}💡 Saran AI:*{COLOR_RESET}
```
{saran}
```
"""
                    else:
                        telegram_msg = f"*{COLOR_RED}🧠 Error Terdeteksi:*{COLOR_RESET}\n{user_context['last_error_log'][-2000:]}\n\n*{COLOR_RED}Gagal mendapatkan saran dari AI: {saran}{COLOR_RESET}*"
                    
                    kirim_ke_telegram(chat_id, context, telegram_msg)
                    # Tidak mereset error_detected_in_stream di sini agar pertanyaan debugging muncul di akhir, dan full_error_output tetap terisi.

        except pexpect.exceptions.EOF:
            logger.info(f"{COLOR_GREEN}[✅] Proses shell selesai.{COLOR_RESET}")
            send_telegram_chunk()
            kirim_ke_telegram(chat_id, context, f"*{COLOR_GREEN}Perintah shell selesai.{COLOR_RESET}*")
            if user_context["last_error_log"]: # Tanya untuk debugging hanya jika ada error
                kirim_ke_telegram(chat_id, context, f"*{COLOR_YELLOW}Terdeteksi error pada eksekusi terakhir. Apakah Anda ingin mencoba melakukan debugging pada program ini dengan bantuan AI? (Ya/Tidak){COLOR_RESET}*")
                user_context["awaiting_debug_response"] = True
                return DEBUGGING_STATE
            break
        except KeyboardInterrupt:
            logger.warning(f"\n{COLOR_YELLOW}[✋] Dihentikan oleh pengguna Termux.{COLOR_RESET}")
            child.sendline('\x03') # Mengirim Ctrl+C
            child.close()
            kirim_ke_telegram(chat_id, context, f"*{COLOR_YELLOW}Proses shell dihentikan secara manual.{COLOR_RESET}*")
            break
        except Exception as e:
            error_msg = f"*{COLOR_RED}[🔴 ERROR] Terjadi kesalahan tidak terduga dalam observer_shell: {e}{COLOR_RESET}"
            kirim_ke_telegram(chat_id, context, error_msg)
            logger.error(error_msg)
            if child.isalive():
                child.close()
            break
    
    return ConversationHandler.END # Mengakhiri conversation jika tidak ada error atau jika EOF tercapai tanpa error

# === Handler Perintah Telegram ===

def start_command(update: Update, context: CallbackContext):
    """Mengirim pesan selamat datang saat perintah /start diberikan."""
    chat_id = update.effective_chat.id
    pesan = f"""
*{COLOR_GREEN}Halo! Saya AI Asisten Shell & Kode Anda.{COLOR_RESET}*
Saya bisa membantu Anda dengan beberapa hal:

1.  *{COLOR_BLUE}Jalankan Perintah Shell:* Cukup ketik perintah Anda atau instruksi alami (misal: `tampilkan isi direktori`).
2.  *{COLOR_BLUE}Hasilkan Kode Python:* Cukup berikan instruksi kode (misal: `buatkan fungsi python untuk menghitung faktorial`).
3.  *{COLOR_BLUE}Percakapan Umum:* Kirim pesan teks biasa untuk percakapan umum.

*Perintah Tambahan:*
* `/listfiles` - Melihat daftar file yang dihasilkan.
* `/deletefile <nama_file>` - Menghapus file yang dihasilkan.
* `/clear_chat` - Menghapus riwayat percakapan.

Pastikan bot saya berjalan di Termux dan semua variabel lingkungan sudah diatur!
    """
    kirim_ke_telegram(chat_id, context, pesan)
    logger.info(f"[Telegram] Pesan /start dikirim ke {chat_id}.")

# Handle fungsi ini sekarang akan menjadi internal, tidak lagi di-trigger langsung dari perintah
# def handle_jalankan_command(update: Update, context: CallbackContext):
#     pass # Akan dipanggil secara internal oleh handle_text_message

# Handle fungsi ini sekarang akan menjadi internal, tidak lagi di-trigger langsung dari perintah
# def handle_buatprogram_command(update: Update, context: CallbackContext):
#     pass # Akan dipanggil secara internal oleh handle_text_message

def handle_listfiles_command(update: Update, context: CallbackContext):
    """Menangani perintah /listfiles untuk menampilkan daftar file yang dihasilkan."""
    chat_id = update.effective_chat.id

    if str(chat_id) != TELEGRAM_CHAT_ID:
        kirim_ke_telegram(chat_id, context, f"*{COLOR_RED}Anda tidak diizinkan untuk menggunakan fitur ini. Hubungi admin bot.{COLOR_RESET}*")
        logger.warning(f"[AUTH ERROR] Percobaan akses tidak sah /listfiles dari {chat_id}.")
        return

    files = [f for f in os.listdir('.') if os.path.isfile(f) and f.endswith('.py') and f != os.path.basename(__file__)]
    
    if files:
        file_list_msg = "*Daftar file Python yang tersedia:*\n" + "\n".join([f"- `{f}`" for f in files])
    else:
        file_list_msg = "*Tidak ada file Python yang dihasilkan.*"
    
    kirim_ke_telegram(chat_id, context, file_list_msg)
    logger.info(f"[Telegram] Daftar file dikirim ke {chat_id}.")

def handle_deletefile_command(update: Update, context: CallbackContext):
    """Menangani perintah /deletefile untuk menghapus file tertentu."""
    chat_id = update.effective_chat.id
    filename_to_delete = " ".join(context.args).strip()

    if str(chat_id) != TELEGRAM_CHAT_ID:
        kirim_ke_telegram(chat_id, context, f"*{COLOR_RED}Anda tidak diizinkan untuk menggunakan fitur ini. Hubungi admin bot.{COLOR_RESET}*")
        logger.warning(f"[AUTH ERROR] Percobaan akses tidak sah /deletefile dari {chat_id}.")
        return

    if not filename_to_delete:
        kirim_ke_telegram(chat_id, context, f"*{COLOR_YELLOW}Mohon berikan nama file yang ingin dihapus. Contoh: /deletefile nama_program_anda.py{COLOR_RESET}*")
        return

    # Pastikan file yang akan dihapus adalah file Python dan bukan file bot itu sendiri
    if not filename_to_delete.endswith(".py") or filename_to_delete == os.path.basename(__file__):
        kirim_ke_telegram(chat_id, context, f"*{COLOR_RED}Hanya file Python yang dihasilkan yang bisa dihapus. Anda tidak bisa menghapus file bot itu sendiri.{COLOR_RESET}*")
        logger.warning(f"[SECURITY VIOLATION] Percobaan menghapus file tidak valid: {filename_to_delete} dari {chat_id}")
        return

    try:
        if os.path.exists(filename_to_delete) and os.path.isfile(filename_to_delete):
            os.remove(filename_to_delete)
            kirim_ke_telegram(chat_id, context, f"*{COLOR_GREEN}File `{filename_to_delete}` berhasil dihapus.{COLOR_RESET}*")
            logger.info(f"[Telegram] File {filename_to_delete} dihapus oleh {chat_id}.")
        else:
            kirim_ke_telegram(chat_id, context, f"*{COLOR_YELLOW}File `{filename_to_delete}` tidak ditemukan.{COLOR_RESET}*")
    except Exception as e:
        kirim_ke_telegram(chat_id, context, f"*{COLOR_RED}Gagal menghapus file `{filename_to_delete}`: {e}{COLOR_RESET}*")
        logger.error(f"[ERROR] Gagal menghapus file {filename_to_delete}: {e}")

def handle_clear_chat_command(update: Update, context: CallbackContext):
    """Menangani perintah /clear_chat untuk menghapus riwayat percakapan."""
    chat_id = update.effective_chat.id
    if str(chat_id) != TELEGRAM_CHAT_ID:
        kirim_ke_telegram(chat_id, context, f"*{COLOR_RED}Anda tidak diizinkan untuk menggunakan fitur ini. Hubungi admin bot.{COLOR_RESET}*")
        logger.warning(f"[AUTH ERROR] Percobaan akses tidak sah /clear_chat dari {chat_id}.")
        return

    if chat_id in chat_histories:
        del chat_histories[chat_id]
        kirim_ke_telegram(chat_id, context, f"*{COLOR_GREEN}Riwayat percakapan Anda telah dihapus.{COLOR_RESET}*")
        logger.info(f"[Telegram] Riwayat chat untuk {chat_id} dihapus.")
    else:
        kirim_ke_telegram(chat_id, context, f"*{COLOR_YELLOW}Tidak ada riwayat percakapan untuk dihapus.{COLOR_RESET}*")


def handle_text_message(update: Update, context: CallbackContext):
    """
    Menangani semua pesan teks non-perintah dari Telegram.
    Akan mendeteksi niat pengguna dan memanggil fungsi yang sesuai.
    """
    chat_id = update.effective_chat.id
    user_message = update.message.text.strip()
    user_context = get_user_context(chat_id)
    
    # Verifikasi bahwa hanya pengguna yang diizinkan (berdasarkan CHAT_ID) yang dapat berinteraksi
    if str(chat_id) != TELEGRAM_CHAT_ID:
        kirim_ke_telegram(chat_id, context, f"*{COLOR_RED}Anda tidak diizinkan untuk berinteraksi dengan bot ini. Hubungi admin bot.{COLOR_RESET}*")
        logger.warning(f"[AUTH ERROR] Percobaan akses tidak sah dari {chat_id}: {user_message}")
        return

    context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING) # Indikator mengetik
    
    # Deteksi niat pengguna
    niat = deteksi_niat_pengguna(user_message)
    user_context["last_user_message_intent"] = niat # Simpan niat pesan terakhir
    logger.info(f"[Niat Terdeteksi] Pengguna {chat_id} -> Niat: {niat}")

    if niat == "shell":
        kirim_ke_telegram(chat_id, context, f"*{COLOR_BLUE}Niat terdeteksi: Perintah Shell. Menerjemahkan instruksi:* `{user_message}`*{COLOR_RESET}*")
        success_konversi, perintah_shell = konversi_ke_perintah_shell(user_message, chat_id)
        perintah_shell = perintah_shell.strip()

        if not success_konversi:
            kirim_ke_telegram(chat_id, context, f"*{COLOR_RED}Terjadi masalah saat mengonversi perintah:{COLOR_RESET}\n{perintah_shell}*")
            logger.error(f"[Error Konversi] {perintah_shell}")
            user_context["last_ai_response_type"] = None
            user_context["last_generated_code"] = None # Reset jika gagal
            return
        elif perintah_shell == "TIDAK_DAPAT_DIKONVERSI":
            kirim_ke_telegram(chat_id, context, f"*{COLOR_YELLOW}Maaf, saya tidak dapat mengonversi instruksi tersebut menjadi perintah shell yang jelas. Harap berikan instruksi yang lebih spesifik.{COLOR_RESET}*")
            user_context["last_ai_response_type"] = None
            user_context["last_generated_code"] = None # Reset jika gagal
            return

        if is_command_dangerous(perintah_shell):
            kirim_ke_telegram(chat_id, context, f"*{COLOR_RED}Perintah ini tidak diizinkan untuk dieksekusi: `{perintah_shell}`. Harap gunakan perintah lain.{COLOR_RESET}*")
            logger.warning(f"[SECURITY VIOLATION] Percobaan menjalankan perintah berbahaya: {perintah_shell} dari {chat_id}")
            user_context["last_ai_response_type"] = None
            user_context["last_generated_code"] = None # Reset jika gagal
            return
        
        kirim_ke_telegram(chat_id, context, f"*{COLOR_GREEN}Menerjemahkan ke perintah shell: `{perintah_shell}`{COLOR_RESET}*")
        user_context["last_ai_response_type"] = "shell"
        user_context["last_generated_code"] = None # Reset kode yang dihasilkan sebelumnya
        run_shell_observer_telegram(perintah_shell, update, context)

    elif niat == "program":
        kirim_ke_telegram(chat_id, context, f"*{COLOR_BLUE}Niat terdeteksi: Pembuatan Program. Mulai menghasilkan kode untuk:* `{user_message}`*{COLOR_RESET}*")
        generated_file_name = generate_filename(user_message)
        success_code, kode_tergenerasi = minta_kode(user_message, chat_id=chat_id)

        if not success_code:
            kirim_ke_telegram(chat_id, context, f"*{COLOR_RED}Terjadi masalah saat menghasilkan kode:{COLOR_RESET}\n{kode_tergenerasi}*")
            logger.error(f"[Error Gen Code] {kode_tergenerasi}")
            user_context["last_ai_response_type"] = None
            user_context["last_generated_code"] = None # Reset jika gagal
            return

        simpan_ok = simpan_ke_file(generated_file_name, kode_tergenerasi)

        if simpan_ok:
            user_context["last_generated_code"] = kode_tergenerasi # Simpan kode yang berhasil digenerasi
            user_context["last_ai_response_type"] = "program"
            user_context["last_command_run"] = None # Reset perintah terakhir
            kirim_ke_telegram(chat_id, context, f"*{COLOR_GREEN}Kode Python berhasil dihasilkan dan disimpan ke {generated_file_name}.{COLOR_RESET}*")
            kirim_ke_telegram(chat_id, context, f"Anda bisa membukanya di Termux dengan: `nano {generated_file_name}`")
            kirim_ke_telegram(chat_id, context, f"Dan menjalankannya dengan: `python {generated_file_name}`")
            kirim_ke_telegram(chat_id, context, f"*{COLOR_PURPLE}Kode Tergenerasi:*{COLOR_RESET}\n```python\n{kode_tergenerasi}\n```")
        else:
            kirim_ke_telegram(chat_id, context, f"*{COLOR_RED}Gagal menyimpan kode yang dihasilkan ke file.{COLOR_RESET}*")
            user_context["last_ai_response_type"] = None
            user_context["last_generated_code"] = None # Reset jika gagal

    else: # niat == "konversasi"
        kirim_ke_telegram(chat_id, context, f"*{COLOR_BLUE}Niat terdeteksi: Percakapan Umum. Meminta jawaban AI...{COLOR_RESET}*")
        success_response, jawaban_llm = minta_jawaban_konversasi(chat_id, user_message)
        user_context["last_ai_response_type"] = "konversasi"
        user_context["last_command_run"] = None # Reset perintah terakhir
        user_context["last_generated_code"] = None # Reset kode yang dihasilkan
        
        if not success_response:
            kirim_ke_telegram(chat_id, context, f"*{COLOR_RED}Terjadi masalah saat memproses percakapan:{COLOR_RESET}\n{jawaban_llm}*")
            logger.error(f"[Error Konversasi] {jawaban_llm}")
        else:
            # Escape the LLM's conversation response before sending to Telegram
            escaped_jawaban_llm = escape_telegram_markdown_v1(jawaban_llm)
            kirim_ke_telegram(chat_id, context, f"*{COLOR_PURPLE}AI Menjawab:*{COLOR_RESET}\n{escaped_jawaban_llm}")


def handle_unknown_command(update: Update, context: CallbackContext):
    """Menanggapi perintah yang tidak dikenal (misalnya /foo bar)."""
    chat_id = update.effective_chat.id
    kirim_ke_telegram(chat_id, context, f"*{COLOR_YELLOW}Perintah tidak dikenal. Silakan gunakan /start untuk melihat perintah yang tersedia.{COLOR_RESET}*")
    logger.warning(f"[WARNING] Perintah tidak dikenal dari {chat_id}: {update.message.text}")

# === Debugging Conversation Handlers ===
def ask_for_debug_response(update: Update, context: CallbackContext):
    """Meminta pengguna untuk respon Ya/Tidak untuk debugging."""
    chat_id = update.effective_chat.id
    user_context = get_user_context(chat_id)
    
    if user_context["awaiting_debug_response"]:
        user_response = update.message.text.strip().lower()
        if user_response in ["ya", "yes"]:
            context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING) # Indikator mengetik
            kirim_ke_telegram(chat_id, context, f"*{COLOR_BLUE}Memulai sesi debugging...{COLOR_RESET}*")
            logger.info(f"{COLOR_BLUE}[Debug] Memulai debugging untuk {chat_id}{COLOR_RESET}")
            
            # Panggil LLM untuk debugging menggunakan full_error_output
            error_log = user_context["last_error_log"] # Ini sudah full error output
            last_command = user_context["last_command_run"]

            if error_log:
                kirim_ke_telegram(chat_id, context, f"*{COLOR_BLUE}Meminta LLM untuk menganalisis error dan memberikan perbaikan/kode baru...{COLOR_RESET}*")
                success_debug, debug_saran = minta_kode(prompt="", error_context=error_log, chat_id=chat_id) # Minta kode dengan konteks error
                
                if not success_debug:
                    kirim_ke_telegram(chat_id, context, f"*{COLOR_RED}Terjadi masalah saat melakukan debugging:{COLOR_RESET}\n{debug_saran}*")
                    logger.error(f"[Debug Error] {debug_saran}")
                else:
                    # Coba deteksi nama file dari perintah terakhir yang dijalankan
                    match = re.search(r"^(python|sh|bash|\./)\s+(\S+\.(py|sh|js|rb|pl|php))", last_command, re.IGNORECASE)
                    if match:
                        debug_file_name = match.group(2) # Ambil nama file dari perintah terakhir
                    else:
                        debug_file_name = "debug_code.py" # Fallback

                    simpan_ok = simpan_ke_file(debug_file_name, debug_saran)
                    if simpan_ok:
                        user_context["last_generated_code"] = debug_saran # Simpan kode yang berhasil digenerasi dari debug
                        user_context["last_ai_response_type"] = "program" # Tipe respons ini adalah program
                        kirim_ke_telegram(chat_id, context, f"*{COLOR_GREEN}AI telah menghasilkan perbaikan/kode baru ke {debug_file_name}.{COLOR_RESET}*")
                        kirim_ke_telegram(chat_id, context, f"Silakan tinjau dan coba jalankan kembali dengan: `{last_command.split()[0]} {debug_file_name}`\n```python\n{debug_saran}\n```")
                    else:
                        kirim_ke_telegram(chat_id, context, f"*{COLOR_RED}Gagal menyimpan kode perbaikan yang dihasilkan ke file.{COLOR_RESET}*")
                        user_context["last_generated_code"] = None # Reset jika gagal
                        user_context["last_ai_response_type"] = None
            else:
                kirim_ke_telegram(chat_id, context, f"*{COLOR_YELLOW}Tidak ada log error yang tersedia untuk debugging.{COLOR_RESET}*")
            
            # Reset konteks debugging
            user_context["last_error_log"] = None
            user_context["last_command_run"] = None
            user_context["awaiting_debug_response"] = False
            user_context["full_error_output"] = []
            return ConversationHandler.END # Akhiri conversation
        elif user_response in ["tidak", "no"]:
            kirim_ke_telegram(chat_id, context, f"*{COLOR_GREEN}Debugging dibatalkan.{COLOR_RESET}*")
            logger.info(f"{COLOR_GREEN}[Debug] Debugging dibatalkan oleh {chat_id}{COLOR_RESET}")
            # Reset konteks debugging
            user_context["last_error_log"] = None
            user_context["last_command_run"] = None
            user_context["awaiting_debug_response"] = False
            user_context["full_error_output"] = []
            return ConversationHandler.END # Akhiri conversation
        else:
            kirim_ke_telegram(chat_id, context, f"*{COLOR_YELLOW}Mohon jawab 'Ya' atau 'Tidak'.{COLOR_RESET}*")
            return DEBUGGING_STATE # Tetap dalam state ini sampai ada jawaban valid
    else:
        # Jika bukan dalam mode menunggu respon debug, lanjutkan ke percakapan biasa
        return handle_text_message(update, context)


def main():
    """Fungsi utama untuk memulai bot Telegram."""
    if not TELEGRAM_BOT_TOKEN:
        logger.error(f"ERROR: TELEGRAM_BOT_TOKEN tidak diatur. Harap setel variabel lingkungan atau masukkan langsung.")
        return
    if not TELEGRAM_CHAT_ID:
        logger.error(f"ERROR: TELEGRAM_CHAT_ID tidak diatur. Harap setel variabel lingkungan atau masukkan langsung.")
        return
    if not OPENROUTER_API_KEY:
        logger.error(f"ERROR: OPENROUTER_API_KEY tidak diatur. Harap setel variabel lingkungan atau masukkan langsung.")
        return

    logger.info(f"{COLOR_GREEN}Memulai Bot Telegram...{COLOR_RESET}")
    logger.info(f"Menggunakan TOKEN: {'*' * (len(TELEGRAM_BOT_TOKEN) - 5) + TELEGRAM_BOT_TOKEN[-5:] if len(TELEGRAM_BOT_TOKEN) > 5 else TELEGRAM_BOT_TOKEN}") # Sensor token
    logger.info(f"Chat ID yang diizinkan: {TELEGRAM_CHAT_ID}")

    updater = Updater(TELEGRAM_BOT_TOKEN)
    dispatcher = updater.dispatcher

    # Handler untuk perintah spesifik (yang tidak akan dideteksi niatnya secara otomatis)
    dispatcher.add_handler(CommandHandler("start", start_command))
    dispatcher.add_handler(CommandHandler("listfiles", handle_listfiles_command))
    dispatcher.add_handler(CommandHandler("deletefile", handle_deletefile_command))
    dispatcher.add_handler(CommandHandler("clear_chat", handle_clear_chat_command))
    
    # Conversation Handler untuk debugging (tetap ada karena ada alur Ya/Tidak)
    conv_handler = ConversationHandler(
        entry_points=[MessageHandler(Filters.text & ~Filters.command, ask_for_debug_response)],
        states={
            DEBUGGING_STATE: [MessageHandler(Filters.text & ~Filters.command, ask_for_debug_response)],
        },
        fallbacks=[CommandHandler('cancel', lambda update, context: ConversationHandler.END)]
    )
    dispatcher.add_handler(conv_handler)
    
    # Handler utama untuk semua pesan teks non-perintah (akan mendeteksi niat)
    dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_text_message))
    
    # Handler untuk perintah yang tidak dikenal
    dispatcher.add_handler(MessageHandler(Filters.command, handle_unknown_command))

    # Mulai bot
    updater.start_polling()
    logger.info(f"{COLOR_GREEN}Bot sedang berjalan. Tekan Ctrl+C untuk menghentikan.{COLOR_RESET}")
    updater.idle() # Menunggu sinyal berhenti (Ctrl+C)

if __name__ == "__main__":
    main()

