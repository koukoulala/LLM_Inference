import json
import pandas as pd
from vllm import LLM, SamplingParams
import time
import os
import argparse

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
                       dtype=args.dtype)
        self.sampling_params = SamplingParams(temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens, repetition_penalty=args.repetition_penalty)

    def batch_inference(self, args, prompt_list, RowId_list, fw, time_token_results):
        try:
            t0 = time.perf_counter()
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
        fw = open(os.path.join(args.output_file), 'w', encoding='utf8')

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
            prompt_text = "<s>[INST] " + prompt_data["prompt"] + " [/INST]"
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



if __name__=='__main__':
    parser = argparse.ArgumentParser(description='LLM inference')

    parser.add_argument('--model', type=str, required=True, help='LLM base model path')
    parser.add_argument('--input_file', type=str, required=True, help='input data path')
    parser.add_argument('--output_file', type=str, required=True, help='output data path')
    parser.add_argument('--dtype', choices=['auto', 'float32', 'float16', 'bfloat16'], default='float16', help='data type for the model weights and activations')
    parser.add_argument('--use_flash_attention_2', type=bool, default=False, help='whether to use flash attention 2')
    parser.add_argument('--temperature', type=float, default=0.95, help='LLM inference sampling params')
    parser.add_argument('--top_p', type=float, default=0.7, help='LLM inference sampling params')
    parser.add_argument('--batch_size', type=int, default=16, help='offline inference batch size')
    parser.add_argument('--max_tokens', type=int, default=500, help='max length of generated text')
    parser.add_argument('--repetition_penalty', type=float, default=1.0, help='To control the repetition of tokens in the generated text, the higher the value, the less repetition')

    '''params using in vllm inference procedure'''
    parser.add_argument('--quantization', type=str, default=None, help='The method used to load quantized the model. If use quantized model, please set quantization method, e.g. AWQ, gptq etc.')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.9, help='GPU memory utilization')
    

    args = parser.parse_args()
    offline_infer = Offline_Inference(args)
    offline_infer.run(args)
            
