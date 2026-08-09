from gtts import gTTS
from langdetect import detect
import os
import shutil
import time

# 语言代码 -> gTTS 语言参数
# gTTS 每种语言只有一个默认声音（不像 edge-tts 可选具体音色）
LANG_MAP = {
    "en": "en",       # 英文
    "zh": "zh-CN",    # 中文（简体）
    "zh-cn": "zh-CN",
    "zh-tw": "zh-TW", # 繁体
    "ja": "ja",       # 日文
    "ko": "ko",       # 韩文
    "ms": "ms",       # 马来文
    "id": "id",       # 印尼文
}


def clear_folder(folder_path):
    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
        print(f"文件夹 '{folder_path}' 不存在，已创建")
        return

    items = os.listdir(folder_path)
    if not items:
        print(f"文件夹 '{folder_path}' 已经为空")
        return

    for item in items:
        item_path = os.path.join(folder_path, item)
        if os.path.isfile(item_path):
            os.remove(item_path)
            print(f"删除文件: {item_path}")
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)
            print(f"删除文件夹: {item_path}")

    print(f"文件夹 '{folder_path}' 已清空")


# --- 播放音频（本地调试用；Streamlit Cloud 上不需要）---
def play_audio(file_path):
    try:
        import pygame
        pygame.mixer.init()
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(1)
        print("播放完成！")
    except Exception as e:
        print(f"播放失败: {e}")
    finally:
        try:
            pygame.mixer.quit()
        except Exception:
            pass


def generate_tts(text, speaker_wav=None):
    """
    自动侦测语言 -> 用 gTTS 生成 mp3。
    返回 (file_path, language)；失败时返回 (None, language_or_None)。
    """
    # 自動偵測語言代碼（如 zh, en, fr）
    try:
        language = detect(text)
    except Exception as e:
        print(f"Language detection failed for text: '{text[:50]}...'. Error: {e}")
        return None, None

    folder_path = "./out_answer/"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
        print(f"Folder '{folder_path}' was created.")

    timestamp = int(time.time())
    tts_file_path = os.path.join(folder_path, f"sft_{timestamp}.mp3")

    # 找 gTTS 对应的语言代码（先精确匹配，再退化到前缀匹配）
    lang_key = language.lower()
    gtts_lang = LANG_MAP.get(lang_key)
    if gtts_lang is None:
        # zh-cn / zh-tw 之类的前缀匹配
        for prefix, code in LANG_MAP.items():
            if lang_key.startswith(prefix):
                gtts_lang = code
                break

    if gtts_lang is None:
        print(f"Unsupported language: '{language}' for text: '{text[:50]}...'. TTS not generated.")
        return None, language

    try:
        print(f"Generating TTS [{gtts_lang}] for: '{text[:50]}...'")
        tts = gTTS(text=text, lang=gtts_lang)
        tts.save(tts_file_path)
        return tts_file_path, language
    except Exception as e:
        print(f"TTS generation failed: {e}")
        return None, language
