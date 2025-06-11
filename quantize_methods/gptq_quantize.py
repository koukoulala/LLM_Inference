import os
max_cpu_threads = "14"
os.environ["OMP_NUM_THREADS"] = max_cpu_threads
os.environ["OPENBLAS_NUM_THREADS"] = max_cpu_threads
os.environ["MKL_NUM_THREADS"] = max_cpu_threads
os.environ["VECLIB_MAXIMUM_THREADS"] = max_cpu_threads
os.environ["NUMEXPR_NUM_THREADS"] = max_cpu_threads
os.environ["NUMEXPR_MAX_THREADS"] = max_cpu_threads

from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from transformers import AutoTokenizer
import torch
import logging
import json
import argparse

def quantize(model_path, quant_path, test_data):
    quantize_config = BaseQuantizeConfig(
        bits=4, # 4 or 8
        group_size=128,
        damp_percent=0.01,
        desc_act=False,  # set to False can significantly speed up inference but the perplexity may slightly bad
        static_groups=False,
        sym=True,
        true_sequential=True,
        model_name_or_path=None,
        model_file_base_name="model"
    )
    max_len = 1024

    # Load model
    model = AutoGPTQForCausalLM.from_pretrained(model_path, quantize_config)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    '''
    messages = [
        [
            {"role": "system", "content": ""},
            {"role": "user", "content": "Please generate 12 Ad Headline in English language, based on the following information:\n\nFinalUrl: https://www.sfinsurancequotesflorida.com/auto/?cmpid=MLBM005-1R \nDomain: sfinsurancequotesflorida.com \nCategory: Financial Services & Insurance -- Insurance \nLandingPage:  . Jonathan Gibbs - State Farm Insurance Agent in Ponte Vedra, FL;  . STATE FARM ® INSURANCE AGENT . Jonathan Gibbs . Email: Email Agent . Office: Get Directions;  . Phone: 904-834-7312 . 340 Town Plaza Avenue Suite 250 . Ponte Vedra, FL 32081 . Agent's Credentials and Licenses: . FL-P214287 . Hours ( EST ): . weekends & evenings by . appointment \nCharacterLimit: between 10 to 30 characters. \nInsight: Ensure diversity by highlighting various selling pionts in each headline. \n"},
            {"role": "assistant", "content": "Ad:Auto Insurance Estimate\nAd:State Farm® Insurance Agent\nAd:Affordable Auto Insurance Rate\nAd:Call Your State Farm® Agent\nAd:State Farm® Auto Insurance\nAd:Auto Insurance Quote\nAd:Affordable Auto Insurance\nAd:Contact Your State Farm® Agent\nAd:Get a Quote\nAd:State Farm® Auto Quote\nAd:Get State Farm® Quote\nAd:Protect Your Belongings\n"}
        ]
    ]
    '''
    #test_data = "/data/xiaoyukou/LLM_Inference/data/small_test.json"
    messages = []
    with open(test_data, 'r', encoding='utf8') as fr:
        test_data = json.load(fr)

    for RowId, data in enumerate(test_data):
        prompt = data["instruction"] + "\n" + data["input"]
        tmp_message = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": data["output"]}
        ]
        messages.append(tmp_message)
    print("count of messages: ", len(messages))
    print("messages examples: ", messages[:2])

    data = []
    for msg in messages:
        text = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
        model_inputs = tokenizer([text])
        input_ids = torch.tensor(model_inputs.input_ids[:max_len], dtype=torch.int)
        data.append(dict(input_ids=input_ids, attention_mask=input_ids.ne(tokenizer.pad_token_id)))

    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Quantize
    #model.quantize(data, cache_examples_on_gpu=False)
    model.quantize(data, cache_examples_on_gpu=20, batch_size=20, use_triton=True)

    # Save quantized model
    model.save_quantized(quant_path, use_safetensors=True)
    tokenizer.save_pretrained(quant_path)
    print("Quantized model saved at: ", quant_path)

# write a main function with arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/data/xiaoyukou/LLM_Inference/output/Mistral-7B-sft-add-copilot-2/", help="The path of the original model.")
    parser.add_argument("--quant_path", type=str, default="/data/xiaoyukou/LLM_Inference/output/Mistral-7B-sft-add-copilot-gptq-2/", help="The path to save the quantized model.")
    parser.add_argument("--test_data", type=str, default="/data/xiaoyukou/LLM_Inference/data/small_test.json", help="The path of the test data.")
    args = parser.parse_args()

    quantize(args.model_path, args.quant_path, args.test_data)
    