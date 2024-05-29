import argparse
import offline_inference, api_create, api_inference, offline_inference_vllm


def init_parse():
    parser = argparse.ArgumentParser(description='LLM inference with vllm utils')

    '''LLM model path, set enable-lora True and provide lora modules path if using lora module'''
    parser.add_argument('--model', type=str, required=True, help='LLM base model path')
    parser.add_argument('--infer_backend', type=str, default="transformers", choices=['transformers', 'vllm'], help="Inference backend.")
    parser.add_argument('--enable_lora', action='store_true', help='Enable lora module')
    parser.add_argument('--lora_modules', type=str, help='LLM lora module path')

    '''data path, each prompt in a line in the input file, and each line of json: {prompt, response} in a line in the output file'''
    parser.add_argument('--input_file', type=str, required=True, help='input data path')
    parser.add_argument('--output_dir', type=str, required=True, help='output data dir path')
    parser.add_argument('--output_file_name', type=str, default="generate_response.tsv", help='output data file name')

    '''choose a mode'''
    parser.add_argument('--infer_type', choices=['offline', 'api_create', 'api_infer'], default='offline', help='please set a mode in [offline], [api_create] and [api_inference]')
    parser.add_argument('--port', type=int, default=7890, help='api port')

    '''params using in LLM inference procedure'''
    parser.add_argument('--dtype', choices=['auto', 'float32', 'float16', 'bfloat16'], default='float16', help='data type for the model weights and activations')
    parser.add_argument('--quantization', type=str, default=None, help='The method used to load quantized the model. If use quantized model, please set quantization method, e.g. AWQ, gptq etc.')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.9, help='GPU memory utilization')
    parser.add_argument('--temperature', type=float, default=0.95, help='LLM inference sampling params')
    parser.add_argument('--top_p', type=float, default=0.7, help='LLM inference sampling params')

    parser.add_argument('--batch_size', type=int, default=1, help='offline inference batch size')
    parser.add_argument('--max_tokens', type=int, default=500, help='max length of generated text')

    args = parser.parse_args()
    return args


def main():
    args = init_parse()
    if args.infer_type == 'offline':
        if args.infer_backend == 'transformers':
            llm_infer = offline_inference.Offline_Inference(args)
            llm_infer.run(args)
        elif args.infer_backend == 'vllm':
            llm_infer = offline_inference_vllm.Offline_Inference(args)
            llm_infer.run(args)
    elif args.infer_type == 'api_create':
        api_create.API_Create.run(args)
    elif args.infer_type == 'api_infer':
        api_inference.API_Inference.run(args)


if __name__=='__main__':
    main()
    pass
