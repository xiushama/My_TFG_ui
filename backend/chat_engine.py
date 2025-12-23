import os
import speech_recognition as sr
from zhipuai import ZhipuAI
from pydub import AudioSegment
from gtts import gTTS


def chat_response(data):
    """
    模拟实时对话系统视频生成逻辑。
    """
    print("[backend.chat_engine] 收到数据：")
    for k, v in data.items():
        print(f"  {k}: {v}")

    # 语音转文字
    # input_audio = "./static/audios/input.wav"
    input_audio = "./static/audios/input.wav"
    input_text = "./static/text/input.txt"

    os.makedirs("./static/text", exist_ok=True)

    text = audio_to_text(input_audio, input_text)
    if not text:
        return os.path.join("static", "videos", "chat_response.mp4")

    # 大模型回答
    output_text = "./static/text/output.txt"

    # set to "31af4e1567ad48f49b6d7b914b4145fb.MDVLvMiePGYLRJ7M"
    api_key = os.getenv("ZHIPU_API_KEY")
    model = "glm-4-plus"

    get_ai_response(input_text, output_text, api_key, model)

    # Text to speech
    audio_output_path = "./static/audios/output.wav"
    text_to_speech(output_text, audio_output_path)

    
    video_path = os.path.join("static", "videos", "chat_response.mp4")
    print(f"[backend.chat_engine] 生成视频路径：{video_path}")
    return video_path

def audio_to_text(input_audio, input_text):

    # Convert audio to PCM WAV format if needed
    try:
        audio = AudioSegment.from_file(input_audio)
        audio = audio.set_channels(1).set_frame_rate(16000)
        audio.export(input_audio, format="wav")
    except Exception as e:
        print(f"音频格式转换失败: {e}")
        return None

    try:
        # 初始化识别器
        recognizer = sr.Recognizer()
        
        # 加载音频文件
        with sr.AudioFile(input_audio) as source:
            # 调整环境噪声
            recognizer.adjust_for_ambient_noise(source)
            # 读取音频数据
            audio_data = recognizer.record(source)
            
            print("正在识别语音...")
            
            # 使用Google语音识别
            text = recognizer.recognize_google(audio_data, language='zh-CN')
            
            # 将结果写入文件
            with open(input_text, 'w', encoding='utf-8') as f:
                f.write(text)
                
            print(f"语音识别完成！结果已保存到: {input_text}")
            print(f"识别结果: {text}")
            
            return text
            
    except sr.UnknownValueError:
        print("无法识别音频内容")
    except sr.RequestError as e:
        print(f"语音识别服务错误: {e}")
    except FileNotFoundError:
        print(f"音频文件不存在: {input_audio}")
    except Exception as e:
        print(f"发生错误: {e}")

def get_ai_response(input_text, output_text, api_key, model):
    client = ZhipuAI(api_key = api_key)
    with open(input_text, 'r', encoding='utf-8') as file:
        content = file.read().strip()

        # protection from null input
        if not content:
            output = "对不起，我没有听清楚，可以再说一遍吗？"
            with open(output_text, 'w', encoding='utf-8') as file:
                file.write(output)
                audio_output_path = "./static/audios/output.wav"
                text_to_speech(output_text, audio_output_path)
            return output

        # some logs
        print("[LLM] User input:")
        print(content)

    # if unable to connect the model because there are no tokens left
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}]
        )
        output = response.choices[0].message.content
    except Exception as e:
        print("[LLM] Error:", e)
        output = "我现在暂时无法连接大模型，但我已经收到你的话了。"

    with open(output_text, 'w', encoding='utf-8') as file:
        file.write(output)

        # some logs again
        print("[LLM] Model output:")
        print(output)

    print(f"答复已保存到: {output_text}")
    return output

def text_to_speech(input_text_path, output_audio_path):
    # Read text
    with open(input_text_path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    if not text:
        print("[TTS] Empty text, skip TTS")
        return

    try:
        tts = gTTS(text=text, lang="zh")
        tts.save(output_audio_path)
        print(f"[TTS] Audio saved to: {output_audio_path}")
    except Exception as e:
        print("[TTS] Error:", e)


