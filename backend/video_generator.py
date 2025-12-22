import os
import time
import subprocess
import shutil
from glob import glob


def _resolve_model_dir(model_param):
    """Handle dir/file/glob inputs and return the model directory containing checkpoints."""
    if not model_param:
        return None

    def _from_checkpoint(path):
        parent = os.path.dirname(path)
        return os.path.dirname(parent) if os.path.basename(parent) == "checkpoints" else parent

    expanded = os.path.expanduser(model_param)

    # Wildcard paths (e.g., SyncTalk/model/trial_may/checkpoints/*.pth)
    if any(ch in expanded for ch in ("*", "?", "[")):
        for candidate in glob(expanded):
            if os.path.isfile(candidate):
                return _from_checkpoint(candidate)
            if os.path.isdir(candidate):
                if os.path.basename(candidate.rstrip(os.sep)) == "checkpoints":
                    return os.path.dirname(candidate)
                checkpoints_dir = os.path.join(candidate, "checkpoints")
                if os.path.isdir(checkpoints_dir):
                    return candidate
        return None

    # Direct file or directory
    if os.path.isfile(expanded):
        return _from_checkpoint(expanded)

    if os.path.isdir(expanded):
        return os.path.dirname(expanded) if os.path.basename(expanded.rstrip(os.sep)) == "checkpoints" else expanded

    return None

def generate_video(data):
    """
    模拟视频生成逻辑：接收来自前端的参数，并返回一个视频路径。
    """
    print("[backend.video_generator] 收到数据：")
    for k, v in data.items():
        print(f"  {k}: {v}")

    if data.get('model_name') == "SyncTalk":
        try:
            # 基础校验
            model_param_raw = data.get('model_param')
            model_param = _resolve_model_dir(model_param_raw)
            ref_audio = data.get('ref_audio')
            gpu_choice = data.get('gpu_choice') or "GPU0"

            if not model_param or not os.path.isdir(model_param):
                raise ValueError(f"模型目录无效: {model_param_raw}")

            checkpoints_dir = os.path.join(model_param, "checkpoints")
            if not os.path.isdir(checkpoints_dir):
                raise ValueError(f"未找到checkpoints目录: {checkpoints_dir}")

            if not ref_audio or not os.path.isfile(ref_audio):
                raise ValueError(f"音频文件不存在: {ref_audio}")

            # 确保输出目录存在
            os.makedirs(os.path.join("static", "videos"), exist_ok=True)

            # 构建命令
            cmd = [
                './SyncTalk/run_synctalk.sh', 'infer',
                '--model_dir', model_param,
                '--audio_path', ref_audio,
                '--gpu', gpu_choice
            ]

            print(f"[backend.video_generator] 解析模型目录: {model_param}")
            print(f"[backend.video_generator] 执行命令: {' '.join(cmd)}")

            # 执行命令
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True
                # check=True
            )
            
            print("命令标准输出:", result.stdout)
            if result.stderr:
                print("命令标准错误:", result.stderr)
            
            # 根据脚本重命名规则定位输出文件
            model_dir_name = os.path.basename(model_param)
            audio_name = os.path.splitext(os.path.basename(ref_audio))[0]
            results_dir = os.path.join("SyncTalk", "model", model_dir_name, "results")
            expected_output = os.path.join(results_dir, f"{model_dir_name}_{audio_name}.mp4")
            destination_path = os.path.join("static", "videos", f"{model_dir_name}_{audio_name}.mp4")

            if os.path.exists(expected_output):
                shutil.copy(expected_output, destination_path)
                print(f"[backend.video_generator] 视频生成完成，路径：{destination_path}")
                return destination_path

            # 兼容旧的默认输出文件名
            default_output = os.path.join(results_dir, "test_audio.mp4")
            if os.path.exists(default_output):
                shutil.copy(default_output, destination_path)
                print(f"[backend.video_generator] 使用默认输出文件，路径：{destination_path}")
                return destination_path

            # 兜底：选取 results 下最新的 mp4 文件
            if os.path.exists(results_dir):
                mp4_files = [f for f in os.listdir(results_dir) if f.endswith('.mp4')]
                if mp4_files:
                    latest_file = max(mp4_files, key=lambda f: os.path.getctime(os.path.join(results_dir, f)))
                    source_path = os.path.join(results_dir, latest_file)
                    shutil.copy(source_path, destination_path)
                    print(f"[backend.video_generator] 找到最新视频文件: {destination_path}")
                    return destination_path

            print(f"[backend.video_generator] 未找到输出视频，返回占位文件")
            return os.path.join("static", "videos", "out.mp4")
            
        except subprocess.CalledProcessError as e:
            print(f"[backend.video_generator] 命令执行失败: {e}")
            print("错误输出:", e.stderr)
            return os.path.join("static", "videos", "out.mp4")
        except Exception as e:
            print(f"[backend.video_generator] 其他错误: {e}")
            return os.path.join("static", "videos", "out.mp4")
    
    video_path = os.path.join("static", "videos", "out.mp4")
    print(f"[backend.video_generator] 视频生成完成，路径：{video_path}")
    return video_path
