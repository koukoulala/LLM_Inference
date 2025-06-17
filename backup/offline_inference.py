import json
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline, GenerationConfig
from peft import PeftModel
import time
import os

class Offline_Inference:

    def __init__(self, args):
        print("Initializing Offline Inference with Transformers pipeline.")
        print("args:", args)
        # specify how to quantize the model
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=args.dtype,
        )
        model_kwargs = dict(
            use_flash_attention_2=args.use_flash_attention_2,
            torch_dtype="auto",
            quantization_config=quantization_config,
        )
        self.llm = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
        if args.enable_lora:
            self.llm = PeftModel.from_pretrained(self.llm, args.lora_modules)
        self.tokenizer = AutoTokenizer.from_pretrained(args.model)
        
        # Create a pipeline for text generation
        self.llm_pipeline = pipeline("text-generation", model=self.llm, tokenizer=self.tokenizer, batch_size = args.batch_size)
        self.llm_pipeline.tokenizer.pad_token_id = self.llm.config.eos_token_id

    def batch_inference(self, args, prompt_list, RowId_list, fw, time_token_results):
        try:
            t0 = time.perf_counter()
            outputs = self.llm_pipeline(prompt_list, do_sample=True, max_new_tokens=args.max_tokens, temperature=args.temperature, top_p=args.top_p)
            t1 = time.perf_counter()
        except Exception as e:
            print(f"Error: {e}")
            return time_token_results
        for out_idx, output in enumerate(outputs):
            RowId = RowId_list[out_idx]
            #print(RowId, output)
            generated_text = output[0]['generated_text'][-1]["content"]
            #tokens_generated += len(self.tokenizer(generated_text))
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
            total_line += 1
            RowId = prompt_data["RowId"]
            prompt_text = prompt_data["prompt"]
            if args.add_system_role:
                message = [{"role": "system", "content": ""}, {"role": "user", "content": prompt_text}]
            else:
                message = [{"role": "user", "content": prompt_text}]
            RowId_list.append(RowId)
            prompt_list.append(message)

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
            
