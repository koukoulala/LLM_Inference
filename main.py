import argparse
import offline_inference, api_create, api_inference


def init_parse():
    parser = argparse.ArgumentParser(description='LLM inference with vllm utils')

    '''LLM model path, set enable-lora True and provide lora modules path if using lora module'''
    parser.add_argument('--model', type=str, required=True, help='LLM base model path')
    parser.add_argument('--enable_lora', action='store_true', help='Enable lora module')
    parser.add_argument('--lora_modules', type=str, help='LLM lora module path')

    '''data path, each prompt in a line in the input file, and each line of json: {prompt, response} in a line in the output file'''
    parser.add_argument('--input_file', type=str, required=True, help='input data path')
    parser.add_argument('--output_dir', type=str, required=True, help='output data dir path')

    '''choose a mode'''
    parser.add_argument('--infer_type', choices=['offline', 'api_create', 'api_infer'], default='offline', help='please set a mode in [offline], [api_create] and [api_inference]')
    parser.add_argument('--port', type=int, default=7890, help='api port')

    '''params using in LLM inference procedure'''
    parser.add_argument('--dtype', choices=['auto', 'float32', 'float16', 'bfloat16'], default='auto', help='data type for the model weights and activations')
    #parser.add_argument('--quantization', type=str, default='AWQ', help='The method used to load quantized the model.')
    #parser.add_argument('--tensor_parallel_size', type=int, default=1, help='use multiple GPUs')
    parser.add_argument('--gpu_memory_utilization', type=float, default=1.0, help='GPU memory utilization')
    parser.add_argument('--temperature', type=float, default=0.9, help='LLM inference sampling params')
    parser.add_argument('--top_p', type=float, default=0.7, help='LLM inference sampling params')

    parser.add_argument('--batch_size', type=int, default=1, help='offline inference batch size')
    parser.add_argument('--max_length', type=int, default=500, help='max length of generated text')

    args = parser.parse_args()
    return args


def main():
    args = init_parse()
    if args.infer_type == 'offline':
        llm_infer = offline_inference.Offline_Inference(args)
        llm_infer.run(args)
    elif args.infer_type == 'api_create':
        api_create.API_Create.run(args)
    elif args.infer_type == 'api_infer':
        api_inference.API_Inference.run(args)


if __name__=='__main__':
    main()
    pass
