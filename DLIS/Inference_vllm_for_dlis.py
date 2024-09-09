import json
import pandas as pd
from vllm import LLM, SamplingParams
import time
import os
import argparse

class ModelImp:

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

    def EvalBatch(self, prompt_list):

        prompt_list_add_tag = ["<s>[INST] " + prompt_text + " [/INST]" for prompt_text in prompt_list]

        outputs = self.llm.generate(prompt_list_add_tag, self.sampling_params)

        res_list = [output.outputs[0].text for output in outputs]

        return res_list
    
    def Eval(self, data):
        print('Eval...')
        res = self.EvalBatch([data])
        return res[0]


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='LLM inference')

    parser.add_argument('--model', type=str, required=True, help='LLM base model path')
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

    test_data = ["Please generate 8 Ad Headline in English language, based on the following information:\n\nFinalUrl: https://join.dallasnews.com/semhd \nDomain: dallasnews.com \nCategory: Entertainment -- Sports & Recreation \nLandingPage:  . Subscribe - The Dallas Morning News | The Dallas Morning News;  . Subscribe today!;  . Special offers: . Print+Digital . 8 weeks . 13 weeks;  . 7-Days . Subscribe Learn more . Convenience of print delivery . Award-winning coverage of DFW . Money-saving coupons delivered each Sunday . Health, business, arts, entertainment + more . 24/7 access to our news & sports sites . ePaper Edition . Exclusi \nCharacterLimit: between 10 to 30 characters. \nInsight: Ensure diversity by highlighting various selling pionts in each headline. \n"
                "Please generate 3 Ad Headline in German language, based on the following information:\n\nFinalUrl: https://www.e-hoi.de/?fuseaction=search.doSearch&preis=1000-1999&preis=500-999&preis=1-499&cruisecompanyid=7&cruisecompanyid=5&cruisecompanyid=178&datum_von=01.01.2025&datum_bis=31.12.2025&usersearch=1&cruisingareatyp=0&header=1,2,3,4,5,6&partnerID=111553000000 \nDomain: e-hoi.de \nCategory: Travel & Transportation -- Cruise \nLandingPage:  . Kreuzfahrten suchen & buchen │ e-hoi;  . AIDA Cruises Kreuzfahrten;  . Schiffsreisen mit AIDA Cruises online buchen . 855 Kreuzfahrt-Ergebnisse . Schiff . Weitere Suchfilter . Kanarische Inseln Kreuzfahrt ab/bis Las Palmas - 92523 . 8 Tage mit der AIDAcosma . Nordland Kreuzfahrt ab/bis Kiel - 86051 . 8 Tage mit der MSC Euribia . Routenkarte . Kanarische Inseln Kreuzfahrt ab/bis Santa Cruz de Te \nCharacterLimit: between 10 to 30 characters. \nInsight: Promote Trust and Brand Identity \n"
                "Please generate 1 Ad Description in English language, based on the following information:\n\nFinalUrl: https://airsculpt.elitebodysculpture.com/lp/vs-traditional-liposuction/?utm_source=bing&utm_medium=cpc&utm_campaign=Bing_National_Men_bodycontouring_BR&utm_term={keyword} \nDomain: elitebodysculpture.com \nCategory: Health & Wellness -- Health Services \nCharacterLimit: between 88 to 90 characters. \nInsight: Emphasize Product Variety or Personalization \n"
                "Please generate 2 Ad Description in Spanish language, based on the following information:\n\nFinalUrl: https://www.latamairlines.com/cl/es/destinos/australia/vuelos-a-sydney \nDomain: latamairlines.com \nCategory: Travel & Transportation \nLandingPage:  . Pasajes y vuelos baratos a Sydney | LATAM Airlines;  . Explora las mejores ofertas de vuelos a Sydney. Reserva tu viaje con los pasajes aéreos más baratos y vive la experiencia LATAM. \nCharacterLimit: between 71 to 90 characters. \nInsight: Ensure diversity by highlighting various selling pionts in each description. \n"
                "Please generate 5 Ad Headline in Simplified Chinese language, based on the following information:\n\nFinalUrl: https://adby.wayboo8.com/page/4/50273/50273-B-230620-16731-v2.html \nDomain: wayboo8.com \nCharacterLimit: between 10 to 30 characters. \nInsight: Promote Trust and Brand Identity \n"
                ]

    test_data = [json.dumps(data) for data in test_data]
    # print(test.EvalBatch(test_data))
    
    test = ModelImp()
    for data in test_data:
        # print('Input: '+data)
        start_time = time.time()
        print('Output: '+ test.Eval(data))
        end_time = time.time()
        print('inference time: '+ str(end_time - start_time))
        print('\n')

            
