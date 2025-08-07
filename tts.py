import edge_tts
from langdetect import detect
import os
import shutil
import asyncio
import pygame
import time

def clear_folder(folder_path):
    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
        print(f"文件夹 '{folder_path}' 不存在，已创建")
        return
    
    # 获取文件夹中的所有文件和子文件夹
    items = os.listdir(folder_path)
    
    # 如果文件夹为空，直接返回
    if not items:
        print(f"文件夹 '{folder_path}' 已经为空")
        return
    
    # 遍历文件和文件夹并删除
    for item in items:
        item_path = os.path.join(folder_path, item)
        
        # 判断是否是文件夹或文件
        if os.path.isfile(item_path):
            os.remove(item_path)  # 删除文件
            print(f"删除文件: {item_path}")
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)  # 删除文件夹及其内容
            print(f"删除文件夹: {item_path}")
    
    print(f"文件夹 '{folder_path}' 已清空")

async def amain(TEXT, VOICE, OUTPUT_FILE) -> None:
    """Main function"""
    communicate = edge_tts.Communicate(TEXT, VOICE)
    await communicate.save(OUTPUT_FILE)

# --- 播放音频 -
def play_audio(file_path):
    try:
        pygame.mixer.init()
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(1)  # 等待音频播放结束
        print("播放完成！")
    except Exception as e:
        print(f"播放失败: {e}")
    finally:
        pygame.mixer.quit()

def generate_tts(text, speaker_wav=None):
    # 自動偵測語言代碼（如 zh, en, fr）
    try:
        language = detect(text)
    except Exception as e:
        print(f"Language detection failed for text: '{text[:50]}...'. Error: {e}")
        return None, None # Or handle as appropriate

    folder_path = "./out_answer/"
    
    # Ensure the output folder exists
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
        print(f"Folder '{folder_path}' was created.")
        
    # Define the output file path
    #tts_file_path = os.path.join(folder_path, "sft_0.mp3") # Using a consistent filename
    timestamp = int(time.time())
    tts_file_path = os.path.join(folder_path, f"sft_{timestamp}.mp3")
    if language == "en":
        print(f"Generating English TTS for: '{text[:50]}...'")
        asyncio.run(amain(text, "en-US-JennyNeural", tts_file_path))
        return tts_file_path, language
    elif language.startswith("zh"):  # Handles "zh-cn", "zh-tw", etc.
        print(f"Generating Chinese TTS for: '{text[:50]}...'")
        asyncio.run(amain(text, "zh-CN-XiaoyiNeural", tts_file_path))
        return tts_file_path, language
    elif language.startswith("ko"):  # Handles "zh-cn", "zh-tw", etc.
        print(f"Generating Chinese TTS for: '{text[:50]}...'")
        asyncio.run(amain(text, "zh-CN-XiaoyiNeural", tts_file_path))
        return tts_file_path, language
    elif language.startswith("ja"):  # Handles "zh-cn", "zh-tw", etc.
        print(f"Generating Chinese TTS for: '{text[:50]}...'")
        asyncio.run(amain(text, "ja-JP-NanamiNeural", tts_file_path))
        return tts_file_path, language
    elif language.startswith("ms"):  # Handles "zh-cn", "zh-tw", etc.
        print(f"Generating Malay TTS for: '{text[:50]}...'")
        asyncio.run(amain(text, "ms-MY-OsmanNeural", tts_file_path))
        return tts_file_path, language

    elif language.startswith("id"):  # Handles "zh-cn", "zh-tw", etc.
        print(f"Generating Indonesia TTS for: '{text[:50]}...'")
        asyncio.run(amain(text, "id-ID-GadisNeural", tts_file_path))
        return tts_file_path, language
    else:
        # Handle unsupported language
        print(f"Unsupported language: '{language}' for text: '{text[:50]}...'. TTS not generated.")
        return None, language # Return None for path to indicate no file was generated