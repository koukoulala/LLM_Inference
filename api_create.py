import os


class API_Create:

    def __init__(self, args):
        pass

    @staticmethod
    def run(self, args):
        cmd = 'python -m vllm.entrypoints.api_server --port {} --model {} --tensor-parallel-size {} --gpu-memory-utilization {}'.format(
            args.port, args.model, args.tensor_parallel_size, args.gpu_memory_utilization)
        if args.enable_lora:
            cmd += ' --enable-lora --lora-modules sql-lora={}'.format(args.lora_modules)
        os.system(cmd)
