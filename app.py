from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
from backend.video_generator import generate_video
from backend.model_trainer import train_model
from backend.chat_engine import chat_response

app = Flask(__name__)

# 简单的GPU参数映射（前端到脚本）
def _map_gpu_choice(gpu_choice: str) -> str:
    if not gpu_choice:
        return "GPU0"
    val = str(gpu_choice).upper()
    if val in ("GPU0", "GPU1", "CPU"):
        return val
    # 前端的 auto / multi 统一回退到 GPU0
    return "GPU0"

# 首页
@app.route('/')
def index():
    return render_template('index.html')

# 视频生成界面
@app.route('/video_generation', methods=['GET', 'POST'])
def video_generation():
    if request.method == 'POST':
        # 处理音频上传（字段名：audio_file）
        os.makedirs('./static/audios', exist_ok=True)
        saved_audio_path = None
        if 'audio_file' in request.files:
            audio_file = request.files['audio_file']
            if audio_file and audio_file.filename:
                filename = secure_filename(audio_file.filename)
                # 避免重名，加入时间戳
                name, ext = os.path.splitext(filename)
                filename = f"{int(__import__('time').time())}_{name}{ext}"
                save_path = os.path.join('static', 'audios', filename)
                audio_file.save(save_path)
                saved_audio_path = save_path

        data = {
            "model_name": request.form.get('model_name'),
            "model_param": request.form.get('model_param'),
            # 优先使用实际保存的音频路径
            "ref_audio": saved_audio_path or request.form.get('ref_audio'),
            "gpu_choice": _map_gpu_choice(request.form.get('gpu_choice')),
            "target_text": request.form.get('target_text'),
        }

        video_path = generate_video(data)
        # 前端使用的路径统一加前缀"/"并规范分隔符
        if isinstance(video_path, str):
            video_path = "/" + video_path.replace("\\", "/")
        return jsonify({'status': 'success', 'video_path': video_path})

    return render_template('video_generation.html')


# 模型训练界面
@app.route('/model_training', methods=['GET', 'POST'])
def model_training():
    if request.method == 'POST':
        # 是否生成训练日志
        generate_log = bool(request.form.get('generate_log'))

        data = {
            "model_choice": request.form.get('model_choice'),
            "ref_video": request.form.get('ref_video'),
            "gpu_choice": _map_gpu_choice(request.form.get('gpu_choice')),
            "epoch": request.form.get('epoch'),
            "custom_params": request.form.get('custom_params'),
            "generate_log": generate_log,
        }

        video_path = train_model(data)
        video_path = "/" + video_path.replace("\\", "/")

        return jsonify({'status': 'success', 'video_path': video_path})

    return render_template('model_training.html')


# 实时对话系统界面
@app.route('/chat_system', methods=['GET', 'POST'])
def chat_system():
    if request.method == 'POST':
        data = {
            "model_name": request.form.get('model_name'),
            "model_param": request.form.get('model_param'),
            "voice_clone": request.form.get('voice_clone'),
            "api_choice": request.form.get('api_choice'),
        }

        video_path = chat_response(data)
        video_path = "/" + video_path.replace("\\", "/")

        return jsonify({'status': 'success', 'video_path': video_path})

    return render_template('chat_system.html')

@app.route('/save_audio', methods=['POST'])
def save_audio():
    if 'audio' not in request.files:
        return jsonify({'status': 'error', 'message': '没有音频文件'})
    
    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({'status': 'error', 'message': '没有选择文件'})
    
    # 确保目录存在
    os.makedirs('./static/audios', exist_ok=True)
    
    # 保存文件
    audio_file.save('./static/audios/input.wav')
    
    return jsonify({'status': 'success', 'message': '音频保存成功'})


if __name__ == '__main__':
    app.run(debug=True, port = 5001)
