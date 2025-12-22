import subprocess
import os
import time
from pathlib import Path

def _map_gpu_choice(gpu_choice: str) -> str:
    if not gpu_choice:
        return "GPU0"
    val = str(gpu_choice).upper()
    if val in ("GPU0", "GPU1", "CPU"):
        return val
    return "GPU0"

def train_model(data):
    """运行 SyncTalk 训练流程，并在成功后生成一个简短推理预览视频。"""
    print("[backend.model_trainer] 收到数据：")
    for k, v in data.items():
        print(f"  {k}: {v}")

    ref_video = data.get('ref_video')
    gpu_choice = _map_gpu_choice(data.get('gpu_choice'))
    epochs = str(data.get('epoch') or '140')
    generate_log = bool(data.get('generate_log'))

    # 基础校验
    if not ref_video or not os.path.isfile(ref_video):
        print(f"[backend.model_trainer] 错误: 训练输入视频不存在: {ref_video}")
        return ref_video or os.path.join("static", "videos", "sample.mp4")

    # 日志文件准备
    log_path = None
    if generate_log:
        os.makedirs(os.path.join("static", "logs"), exist_ok=True)
        ts = int(time.time())
        log_path = os.path.join("static", "logs", f"train_{ts}.log")

    print("[backend.model_trainer] 模型训练中...")

    if data.get('model_choice') == "SyncTalk":
        try:
            cmd = [
                "./SyncTalk/run_synctalk.sh", "train",
                "--video_path", ref_video,
                "--gpu", gpu_choice,
                "--epochs", epochs,
            ]

            print(f"[backend.model_trainer] 执行命令: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
            )

            # 打印并保存日志
            print("[backend.model_trainer] 训练输出:\n", result.stdout)
            if result.stderr:
                print("[backend.model_trainer] 错误输出:\n", result.stderr)
            if log_path:
                with open(log_path, 'w', encoding='utf-8') as f:
                    f.write("# CMD\n" + " ".join(cmd) + "\n\n")
                    f.write("# STDOUT\n" + (result.stdout or "") + "\n\n")
                    if result.stderr:
                        f.write("# STDERR\n" + result.stderr + "\n")

        except subprocess.CalledProcessError as e:
            print(f"[backend.model_trainer] 训练失败，退出码: {e.returncode}")
            if log_path:
                with open(log_path, 'a', encoding='utf-8') as f:
                    f.write(f"# ERROR\nReturn code: {e.returncode}\n{e.stderr or ''}\n")
            return ref_video
        except FileNotFoundError:
            print("[backend.model_trainer] 错误: 找不到训练脚本")
            return ref_video
        except Exception as e:
            print(f"[backend.model_trainer] 训练过程中发生未知错误: {e}")
            return ref_video

        # 训练完成后尝试生成一个预览视频（使用默认音频 aud.wav）
        try:
            video_name = Path(ref_video).stem
            model_dir_name = f"{video_name}_ep{epochs}"
            model_dir = os.path.join("SyncTalk", "model", model_dir_name)
            preview_audio = os.path.join("SyncTalk", "audio", "aud.wav")

            os.makedirs(os.path.join("static", "videos"), exist_ok=True)

            if os.path.isdir(model_dir) and os.path.isfile(preview_audio):
                infer_cmd = [
                    "./SyncTalk/run_synctalk.sh", "infer",
                    "--model_dir", model_dir,
                    "--audio_path", preview_audio,
                    "--gpu", gpu_choice,
                ]
                print(f"[backend.model_trainer] 训练后预览推理: {' '.join(infer_cmd)}")
                infer_res = subprocess.run(
                    infer_cmd,
                    capture_output=True,
                    text=True,
                )
                if log_path:
                    with open(log_path, 'a', encoding='utf-8') as f:
                        f.write("\n# INFER STDOUT\n" + (infer_res.stdout or "") + "\n")
                        if infer_res.stderr:
                            f.write("# INFER STDERR\n" + infer_res.stderr + "\n")

                # 复制输出视频到静态目录
                audio_name = Path(preview_audio).stem
                results_dir = os.path.join(model_dir, "results")
                expected_output = os.path.join(results_dir, f"{model_dir_name}_{audio_name}.mp4")
                if os.path.exists(expected_output):
                    dest = os.path.join("static", "videos", f"{model_dir_name}_{audio_name}.mp4")
                    try:
                        import shutil
                        shutil.copy(expected_output, dest)
                        print(f"[backend.model_trainer] 预览视频生成完成: {dest}")
                        return dest
                    except Exception as copy_err:
                        print(f"[backend.model_trainer] 复制预览视频失败: {copy_err}")

            else:
                print("[backend.model_trainer] 缺少模型目录或预览音频，跳过预览推理")

        except Exception as e:
            print(f"[backend.model_trainer] 预览推理失败: {e}")

    print("[backend.model_trainer] 训练完成")
    # 默认返回原始训练视频路径（作为页面占位或预览）
    return ref_video
