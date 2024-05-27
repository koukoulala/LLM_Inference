import json
import pandas as pd
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
import time
import os

class Offline_Inference:

    def __init__(self, args):
        tensor_parallel_size = int(os.environ.get("DEVICES", "1"))
        self.llm = LLM(model=args.model,
                       tensor_parallel_size=tensor_parallel_size, gpu_memory_utilization=args.gpu_memory_utilization,
                       dtype=args.dtype, enable_lora=args.enable_lora)
        self.sampling_params = SamplingParams(temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens)

    def batch_inference(self, args, prompt_list, RowId_list, fw, time_token_results, t0):
        if args.enable_lora:
            outputs = self.llm.generate(prompt_list, self.sampling_params, lora_request=LoRARequest('lora_adapter', 1, args.lora_modules))
        else:
            outputs = self.llm.generate(prompt_list, self.sampling_params)
        t1 = time.perf_counter()
        #print("Inference time: ", t1 - t0)
        time_token_results.append(
            {"time": t1 - t0, "tokens_generated": len(outputs[0].outputs[0].token_ids)}
        )
        for out_idx, output in enumerate(outputs):
            RowId = RowId_list[out_idx]
            prompt = output.prompt
            generated_text = output.outputs[0].text
            #print("RowId: ", RowId)
            #print("prompt: ", prompt)
            #print("generated_text: ", generated_text)
            fw.write(json.dumps({'RowId': RowId, 'response': generated_text}, ensure_ascii=False) + '\n')

        return time_token_results
    
    def run(self, args):
        os.makedirs(args.output_dir, exist_ok=True)
        fw = open(os.path.join(args.output_dir, "generate_response.tsv"), 'w', encoding='utf8')
        time_token_results = []
        RowId_list = []
        prompt_list = []
        for idx, line in enumerate(open(args.input_file, 'r', encoding='utf8')):
            if idx % 100 == 0:
                print(f"Processing {idx}th text")
            prompt_data = json.loads(line.strip())
            RowId = prompt_data["RowId"]
            prompt_text = prompt_data["prompt"]
            RowId_list.append(RowId)
            prompt_list.append(prompt_text)

            if ((idx + 1) % args.batch_size) == 0:
                t0 = time.perf_counter()
                time_token_results = self.batch_inference(args, prompt_list, RowId_list, fw, time_token_results, t0)
                RowId_list = []
                prompt_list = []
        
        if len(prompt_list) > 0:
            t0 = time.perf_counter()
            time_token_results = self.batch_inference(args, prompt_list, RowId_list, fw, time_token_results, t0)

        fw.close()
        df = pd.DataFrame(time_token_results)
        print(f"Average time to complete texts: {df.time.mean(): .3f}")
        df["tokens_per_sec"] = df.tokens_generated / df.time
        print(f"Average tokens/sec: {df.tokens_per_sec.mean(): .3f}")

        print("Inference completed")
            
