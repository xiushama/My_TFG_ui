#!/bin/bash
# run_synctalk.sh

set -e  # 遇到错误立即退出

# 基础变量
IMAGE_NAME="synctalk:latest"
WORKSPACE="/SyncTalk"
SYNCTALK_DIR="./SyncTalk"  # 宿主机SyncTalk目录

# Torch 缓存目录
TORCH_CACHE_DIR="/data/torch_cache"
mkdir -p "$TORCH_CACHE_DIR"

# 测试模式开关
TEST_MODE=${TEST_MODE:-0}

# 模拟Docker命令的函数
mock_docker() {
    if [ "$TEST_MODE" -eq 1 ]; then
        echo "[MOCK DOCKER] 模拟执行: $*"
        echo "  工作目录: $(pwd)"
        echo "  挂载点: $WORKSPACE"
        return 0
    else
        # 真实执行Docker命令
        docker "$@"
    fi
}

# 获取视频名称
get_basename() {
    local file_path="$1"
    local basename=$(basename "$file_path")
    echo "${basename%.*}"
}

# 智能解析GPU参数
parse_gpu_arg() {
    local gpu_arg="$1"
    
    local upper_arg=$(echo "$gpu_arg" | tr '[:lower:]' '[:upper:]')
    
    if [[ "$upper_arg" == "CPU" ]]; then
        echo ""
        return 0
    fi
    
    if [[ "$upper_arg" =~ ^GPU[0-9]+$ ]]; then
        local gpu_num=$(echo "$upper_arg" | sed 's/GPU//')
        echo "--gpus device=$gpu_num"
    else
        echo "--gpus device=0"
    fi
}

# 生成模型目录名（videoName_epN）
generate_model_dir_name() {
    local video_name="$1"
    local epochs="$2"
    echo "${video_name}_ep${epochs}"
}

# 确保SyncTalk目录结构存在
ensure_synctalk_dirs() {
    mkdir -p "$SYNCTALK_DIR/data"
    mkdir -p "$SYNCTALK_DIR/model"
    mkdir -p "$SYNCTALK_DIR/audio"
    echo "SyncTalk目录结构已创建: $SYNCTALK_DIR"
}

# 选择首选的checkpoint文件（在宿主机上完成查找，再传入容器）
select_checkpoint() {
    local model_dir="$1"
    local checkpoints_dir="$model_dir/checkpoints"

    if [ ! -d "$checkpoints_dir" ]; then
        echo "";
        return
    fi

    if [ -f "$checkpoints_dir/ngp.pth" ]; then
        basename "$checkpoints_dir/ngp.pth"
        return
    fi

    local candidate=$(ls "$checkpoints_dir"/ngp_ep*.pth 2>/dev/null | head -1)
    if [ -z "$candidate" ]; then
        candidate=$(ls "$checkpoints_dir"/*.pth 2>/dev/null | head -1)
    fi

    if [ -n "$candidate" ]; then
        basename "$candidate"
        return
    fi

    echo ""
}

# 数据预处理函数
preprocess_only() {
    local video_path=""
    local gpu_arg="GPU0"
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --video_path)
                video_path="$2"
                shift 2
                ;;
            --gpu)
                gpu_arg="$2"
                shift 2
                ;;
            *)
                echo "未知参数: $1"
                usage
                exit 1
                ;;
        esac
    done
    
    if [ -z "$video_path" ]; then
        echo "错误: 必须指定视频路径(--video_path)"
        usage
        exit 1
    fi
    
    if [ ! -f "$video_path" ]; then
        echo "错误: 视频文件不存在: $video_path"
        exit 1
    fi

    ensure_synctalk_dirs

    local video_name=$(get_basename "$video_path")
    local data_dir="$SYNCTALK_DIR/data/${video_name}"
    local gpu_param=$(parse_gpu_arg "$gpu_arg")
    
    echo "开始数据预处理..."
    echo "  - 输入视频: $video_path"
    echo "  - 数据目录: $data_dir"
    echo "  - GPU设置: $gpu_arg -> $gpu_param"
    
    mkdir -p "$data_dir"
    rsync -u "$video_path" "$data_dir/"
    
    local synctalk_abs=$(realpath "$SYNCTALK_DIR")
    
    mock_docker run --rm $gpu_param \
        -v "$synctalk_abs/data:$WORKSPACE/data" \
        -v "$TORCH_CACHE_DIR:$TORCH_CACHE_DIR" \
        -e TORCH_HOME="$TORCH_CACHE_DIR" \
        $IMAGE_NAME \
        python data_utils/process.py \
        "data/${video_name}/$(basename "$video_path")" \
        --asr ave \
        --task -1
    
    echo "预处理完成!"
    echo "数据保存在: $data_dir"
}

# 仅训练函数
train_only() {
    local video_path=""
    local gpu_arg="GPU0"
    local epochs="140"
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --video_path)
                video_path="$2"
                shift 2
                ;;
            --gpu)
                gpu_arg="$2"
                shift 2
                ;;
            --epochs)
                epochs="$2"
                shift 2
                ;;
            *)
                echo "未知参数: $1"
                usage
                exit 1
                ;;
        esac
    done
    
    if [ -z "$video_path" ]; then
        echo "错误: 必须指定视频路径(--video_path)"
        usage
        exit 1
    fi

    ensure_synctalk_dirs

    local video_name=$(get_basename "$video_path")
    local data_dir="$SYNCTALK_DIR/data/${video_name}"
    
    if [ ! -d "$data_dir" ]; then
        echo "错误: 数据目录不存在: $data_dir"
        echo "请先运行预处理步骤"
        exit 1
    fi
    
    local model_dir_name=$(generate_model_dir_name "$video_name" "$epochs")
    local model_dir="$SYNCTALK_DIR/model/${model_dir_name}"
    mkdir -p "$model_dir"
    
    local gpu_param=$(parse_gpu_arg "$gpu_arg")
    
    echo "SyncTalk开始模型训练..."
    echo "  - 视频名称: $video_name"
    echo "  - 数据目录: $data_dir"
    echo "  - 模型目录: $model_dir"
    echo "  - GPU设置: $gpu_arg -> $gpu_param"
    echo "  - 训练轮数: $epochs"
    
    local max_epoch_first=$((epochs * 7 / 10))
    local asr_model="ave"
    local iters_1=60000
    local iters_2=100000
    local learning_rate=0.01
    local learning_rate_net=0.001
    
    local synctalk_abs=$(realpath "$SYNCTALK_DIR")
    
    echo "第一阶段训练..."
    mock_docker run --rm $gpu_param \
        -v "$synctalk_abs/data:$WORKSPACE/data" \
        -v "$synctalk_abs/model:$WORKSPACE/model" \
        -v "$TORCH_CACHE_DIR:$TORCH_CACHE_DIR" \
        -e TORCH_HOME="$TORCH_CACHE_DIR" \
        $IMAGE_NAME \
        python main.py "$WORKSPACE/data/${video_name}" \
        --workspace "$WORKSPACE/model/${model_dir_name}" \
        -O \
        --iters $iters_1 \
        --asr_model $asr_model \
        --preload 1 \
        --max_epoch $max_epoch_first \
        --lr $learning_rate \
        --lr_net $learning_rate_net
    
    echo "第二阶段微调..."
    mock_docker run --rm $gpu_param \
        -v "$synctalk_abs/data:$WORKSPACE/data" \
        -v "$synctalk_abs/model:$WORKSPACE/model" \
        -v "$TORCH_CACHE_DIR:$TORCH_CACHE_DIR" \
        -e TORCH_HOME="$TORCH_CACHE_DIR" \
        $IMAGE_NAME \
        python main.py "$WORKSPACE/data/${video_name}" \
        --workspace "$WORKSPACE/model/${model_dir_name}" \
        -O \
        --iters $iters_2 \
        --finetune_lips \
        --patch_size 64 \
        --asr_model $asr_model \
        --preload 1 \
        --max_epoch $epochs
    
    echo "SyncTalk训练完成!"
    echo "模型保存在: $model_dir"
}

# 完整训练流程
train() {
    local video_path=""
    local gpu_arg="GPU0"
    local epochs="140"
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --video_path)
                video_path="$2"
                shift 2
                ;;
            --gpu)
                gpu_arg="$2"
                shift 2
                ;;
            --epochs)
                epochs="$2"
                shift 2
                ;;
            *)
                echo "未知参数: $1"
                usage
                exit 1
                ;;
        esac
    done
    
    if [ -z "$video_path" ]; then
        echo "错误: 必须指定视频路径(--video_path)"
        usage
        exit 1
    fi
    
    echo "开始完整训练流程..."
    
    echo "步骤1: 数据预处理"
    preprocess_only --video_path "$video_path" --gpu "$gpu_arg"
    
    echo "步骤2: 模型训练"
    train_only --video_path "$video_path" --gpu "$gpu_arg" --epochs "$epochs"
    
    echo "完整训练流程完成!"
    echo "SyncTalk目录: $SYNCTALK_DIR"
}

# 推理函数
infer() {
    local model_dir=""
    local audio_path=""
    local gpu_arg="GPU0"
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --model_dir)
                model_dir="$2"
                shift 2
                ;;
            --audio_path)
                audio_path="$2"
                shift 2
                ;;
            --gpu)
                gpu_arg="$2"
                shift 2
                ;;
            *)
                echo "未知参数: $1"
                usage
                exit 1
                ;;
        esac
    done
    
    if [ -z "$model_dir" ] || [ -z "$audio_path" ]; then
        echo "错误: 必须指定模型目录和音频路径"
        usage
        exit 1
    fi
    
    if [ ! -f "$audio_path" ]; then
        echo "错误: 音频文件不存在: $audio_path"
        exit 1
    fi

    ensure_synctalk_dirs
    local gpu_param=$(parse_gpu_arg "$gpu_arg")
    local audio_name=$(get_basename "$audio_path")
    local model_dir_name=$(get_basename "$model_dir")
    local output_filename="${model_dir_name}_${audio_name}.mp4"
    local results_dir="$model_dir/results"
    local video_name=$(echo "$model_dir_name" | sed 's/_ep[0-9]*$//')
    local data_dir="$SYNCTALK_DIR/data/${video_name}"
    local audio_dir="$SYNCTALK_DIR/audio"
    
    if [ ! -d "$data_dir" ] || [ ! -d "$model_dir" ]; then
        echo "错误: 数据或模型目录不存在"
        exit 1
    fi
    
    mkdir -p "$results_dir"
    rsync -u "$audio_path" "$audio_dir/"
    local audio_filename=$(basename "$audio_path")

    # 在宿主机上选择checkpoint文件，然后传递给容器
    local ckpt_basename=$(select_checkpoint "$model_dir")
    if [ -z "$ckpt_basename" ]; then
        echo "错误: 未找到checkpoint文件: $model_dir/checkpoints"
        exit 1
    fi

    local synctalk_abs=$(realpath "$SYNCTALK_DIR")
    
    mock_docker run --rm $gpu_param \
        -v "$synctalk_abs/data:$WORKSPACE/data" \
        -v "$synctalk_abs/model:$WORKSPACE/model" \
        -v "$synctalk_abs/audio:$WORKSPACE/audio" \
        -v "$TORCH_CACHE_DIR:$TORCH_CACHE_DIR" \
        -e TORCH_HOME="$TORCH_CACHE_DIR" \
        $IMAGE_NAME \
        python main.py "$WORKSPACE/data/${video_name}" \
        --workspace "$WORKSPACE/model/${model_dir_name}" \
        -O \
        --test \
        --test_train \
        --asr_model ave \
        --aud "$WORKSPACE/audio/${audio_filename}" \
        --portrait \
        --ckpt "$WORKSPACE/model/${model_dir_name}/checkpoints/${ckpt_basename}"
    
    local default_output="$results_dir/test_audio.mp4"
    local final_output="$results_dir/$output_filename"
    
    if [ -f "$default_output" ]; then
        mv "$default_output" "$final_output"
        echo "推理完成! 输出视频: $final_output"
    else
        local found_output=$(find "$results_dir" -name "*.mp4" | head -1)
        if [ -n "$found_output" ]; then
            echo "警告: 未找到默认输出文件，但发现: $(basename "$found_output")"
            echo "输出视频: $found_output"
        else
            echo "错误: 推理完成但未生成输出视频"
            exit 1
        fi
    fi
}

usage() {
    echo "用法: $0 <任务类型> [参数]"
    echo "任务: train | preprocess_only | train_only | infer"
    echo "示例: $0 train --video_path ./video.mp4 --gpu GPU0 --epochs 50"
}

main() {
    if [ $# -eq 0 ]; then
        usage
        exit 1
    fi
    
    case $1 in
        train) shift; train "$@" ;;
        preprocess_only) shift; preprocess_only "$@" ;;
        train_only) shift; train_only "$@" ;;
        infer) shift; infer "$@" ;;
        -h|--help) usage ;;
        *) echo "未知任务 $1"; usage; exit 1 ;;
    esac
}

main "$@"
