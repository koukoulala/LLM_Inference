CUDA_VISIBLE_DEVICES=0 vllm serve /data/xiaoyukou/ckpts/Qwen3-VL-8B-Instruct \
    --async-scheduling \
    --limit-mm-per-prompt.video 0 \
    --media-io-kwargs '{"video": {"num_frames": -1}}' \
    --gpu-memory-utilization 0.95 \
    --max-model-len 20000 \
    --host 0.0.0.0 \
    --port 22002