import json
import pandas as pd
from vllm import LLM, SamplingParams, EngineArgs, LLMEngine
from vllm.lora.request import LoRARequest
from transformers import AutoTokenizer
import time
import os

class Offline_Inference:

    def __init__(self, args):
        print("Initializing Offline Inference with VLLM.")
        print("args:", args)
        cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if cuda_visible_devices:
            tensor_parallel_size = len(cuda_visible_devices.split(','))
        else:
            tensor_parallel_size = 1
        print(f"Number of GPUs specified by CUDA_VISIBLE_DEVICES: {tensor_parallel_size}")

        self.llm = LLM(model=args.model, quantization=args.quantization,
                       tensor_parallel_size=tensor_parallel_size, gpu_memory_utilization=args.gpu_memory_utilization,
                       dtype=args.dtype, enable_lora=args.enable_lora)
        self.sampling_params = SamplingParams(temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens, repetition_penalty=args.repetition_penalty)

    def batch_inference(self, args, prompt_list, RowId_list, fw, time_token_results):
        print("\nprompt_list: ", prompt_list)
        try:
            t0 = time.perf_counter()
            if args.enable_lora:
                outputs = self.llm.generate(prompt_list, self.sampling_params, lora_request=LoRARequest('lora_adapter', 1, args.lora_modules))
            else:
                outputs = self.llm.generate(prompt_list, self.sampling_params)
            t1 = time.perf_counter()
        except Exception as e:
            print(f"Error: {e}")
            return time_token_results
        for out_idx, output in enumerate(outputs):
            RowId = RowId_list[out_idx]
            generated_text = output.outputs[0].text
            fw.write(json.dumps({'RowId': RowId, 'response': generated_text}, ensure_ascii=False) + '\n')

        time_token_results.append(
            {"time": t1 - t0}
        )

        return time_token_results
    
    def run(self, args):
        t0_all = time.perf_counter()
        os.makedirs(args.output_dir, exist_ok=True)
        fw = open(os.path.join(args.output_dir, args.output_file_name), 'w', encoding='utf8')

        # Read all data into memory
        with open(args.input_file, 'r', encoding='utf8') as f:
            data = [json.loads(line.strip()) for line in f]

        total_line = len(data)
        print(f"Total number of texts: {total_line}")

        time_token_results = []
        RowId_list = []
        prompt_list = []
        for idx, prompt_data in enumerate(data):
            if idx % 100 == 0:
                print(f"Processing {idx}th text")
            RowId = prompt_data["RowId"]
            prompt_text = "[INST] " + prompt_data["prompt"] + " [/INST]"
            #prompt_text = prompt_data["prompt"]
            RowId_list.append(RowId)
            prompt_list.append(prompt_text)

            if ((idx + 1) % args.batch_size) == 0:
                time_token_results = self.batch_inference(args, prompt_list, RowId_list, fw, time_token_results)
                RowId_list = []
                prompt_list = []
        
        if len(prompt_list) > 0:
            time_token_results = self.batch_inference(args, prompt_list, RowId_list, fw, time_token_results)

        fw.close()
        print(f"Total number of texts: {total_line}")
        df = pd.DataFrame(time_token_results)
        print(f"Average time to complete texts: {df.time.sum() / total_line: .3f}")

        print("Inference completed")
        t1_all = time.perf_counter()
        print(f"Total time for processing all data: {t1_all - t0_all: .3f}")
            
