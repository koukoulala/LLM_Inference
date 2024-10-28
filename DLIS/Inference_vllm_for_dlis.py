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
                       tensor_parallel_size=tensor_parallel_size, gpu_memory_utilization=args.gpu_memory_utilization, dtype=args.dtype)
        self.sampling_params = SamplingParams(temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens, repetition_penalty=args.repetition_penalty)
        self.args = args
    
    def EvalBatch(self, prompt_list):

        #prompt_list = ["<s>[INST] " + prompt_text + " [/INST]" for prompt_text in prompt_list]
        mistral_template = '[INST] {} [/INST]'
        llama3_template = '<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n' 
        if self.args.model_name == 'mistral':
            prompt_list = [mistral_template.format(prompt_text) for prompt_text in prompt_list]
        elif self.args.model_name == 'llama3':
            prompt_list = [llama3_template.format(prompt_text) for prompt_text in prompt_list]

        print('prompt_list:', prompt_list)

        outputs = self.llm.generate(prompt_list, self.sampling_params)

        res_list = [output.outputs[0].text for output in outputs]
        #print('res_list:', res_list)

        return res_list
    
    def Eval(self, single_prompt):
        print('Eval...')
        res = self.EvalBatch([single_prompt])
        return res[0]


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='LLM inference')

    parser.add_argument('--model', type=str, required=True, help='LLM base model path')
    parser.add_argument('--model_name', type=str, default="mistral", help='LLM base model name')
    parser.add_argument('--dtype', choices=['auto', 'float32', 'float16', 'bfloat16'], default='auto', help='data type for the model weights and activations')
    parser.add_argument('--use_flash_attention_2', type=bool, default=True, help='whether to use flash attention 2')
    parser.add_argument('--temperature', type=float, default=0.95, help='LLM inference sampling params')
    parser.add_argument('--top_p', type=float, default=0.7, help='LLM inference sampling params')
    parser.add_argument('--max_tokens', type=int, default=500, help='max length of generated text')
    parser.add_argument('--repetition_penalty', type=float, default=1.0, help='To control the repetition of tokens in the generated text, the higher the value, the less repetition')

    '''params using in vllm inference procedure'''
    parser.add_argument('--quantization', type=str, default=None, help='The method used to load quantized the model. If use quantized model, please set quantization method, e.g. AWQ, gptq etc.')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.9, help='GPU memory utilization')
    
    args = parser.parse_args()
    '''
    test_data = ["Please generate 8 Ad Headline in English language, based on the following information:\n\nFinalUrl: https://join.dallasnews.com/semhd \nDomain: dallasnews.com \nCategory: Entertainment -- Sports & Recreation \nLandingPage:  . Subscribe - The Dallas Morning News | The Dallas Morning News;  . Subscribe today!;  . Special offers: . Print+Digital . 8 weeks . 13 weeks;  . 7-Days . Subscribe Learn more . Convenience of print delivery . Award-winning coverage of DFW . Money-saving coupons delivered each Sunday . Health, business, arts, entertainment + more . 24/7 access to our news & sports sites . ePaper Edition . Exclusi \nCharacterLimit: between 10 to 30 characters. \nInsight: Ensure diversity by highlighting various selling pionts in each headline. \n",
                "Please generate 3 Ad Headline in German language, based on the following information:\n\nFinalUrl: https://www.e-hoi.de/?fuseaction=search.doSearch&preis=1000-1999&preis=500-999&preis=1-499&cruisecompanyid=7&cruisecompanyid=5&cruisecompanyid=178&datum_von=01.01.2025&datum_bis=31.12.2025&usersearch=1&cruisingareatyp=0&header=1,2,3,4,5,6&partnerID=111553000000 \nDomain: e-hoi.de \nCategory: Travel & Transportation -- Cruise \nLandingPage:  . Kreuzfahrten suchen & buchen │ e-hoi;  . AIDA Cruises Kreuzfahrten;  . Schiffsreisen mit AIDA Cruises online buchen . 855 Kreuzfahrt-Ergebnisse . Schiff . Weitere Suchfilter . Kanarische Inseln Kreuzfahrt ab/bis Las Palmas - 92523 . 8 Tage mit der AIDAcosma . Nordland Kreuzfahrt ab/bis Kiel - 86051 . 8 Tage mit der MSC Euribia . Routenkarte . Kanarische Inseln Kreuzfahrt ab/bis Santa Cruz de Te \nCharacterLimit: between 10 to 30 characters. \nInsight: Incorporate dynamic keyword insertion to make your ad more relevant to query. \n",
                "Please generate 1 Ad Description in English language, based on the following information:\n\nFinalUrl: https://airsculpt.elitebodysculpture.com/lp/vs-traditional-liposuction/?utm_source=bing&utm_medium=cpc&utm_campaign=Bing_National_Men_bodycontouring_BR&utm_term={keyword} \nDomain: elitebodysculpture.com \nCategory: Health & Wellness -- Health Services \nCharacterLimit: between 88 to 90 characters. \nInsight: Emphasize Product Variety or Personalization \n",
                "Please generate 2 Ad Description in Spanish language, based on the following information:\n\nFinalUrl: https://www.latamairlines.com/cl/es/destinos/australia/vuelos-a-sydney \nDomain: latamairlines.com \nCategory: Travel & Transportation \nLandingPage:  . Pasajes y vuelos baratos a Sydney | LATAM Airlines;  . Explora las mejores ofertas de vuelos a Sydney. Reserva tu viaje con los pasajes aéreos más baratos y vive la experiencia LATAM. \nCharacterLimit: between 71 to 90 characters. \nInsight: Ensure diversity by highlighting various selling pionts in each description. \n",
                "Please generate 3 Ad Headline in English language, based on the following information:\n\nFinalUrl: https://join.dallasnews.com/semhd \nLandingPage:  . Subscribe - The Dallas Morning News | The Dallas Morning News;  . Subscribe today!;  . Special offers: . Print+Digital . 8 weeks . 13 weeks;  . 7-Days . Subscribe Learn more . Convenience of print delivery . Award-winning coverage of DFW . Money-saving coupons delivered each Sunday . Health, business, arts, entertainment + more . 24/7 access to our news & sports sites . ePaper Edition . Exclusi \nCharacterLimit: between 10 to 30 characters. \nInsight: Incorporate dynamic keyword insertion to make your ad more relevant to query. \n",
                "Please generate 4 Ad Description in English language, based on the following information:\n\nFinalUrl: https://www.ticketpurchase.com/venues/ravinia-pavilion \nLandingPage:  . Ravinia Pavilion Tickets;  . Ravinia Pavilion Event Schedule & Ticket Availability;  . Date . Performer . Average Ticket Price . Jun 28, 2024 . Ben Platt . Jun 29, 2024 . Roger Daltrey & KT Tunstall . Jun 30, 2024 . Clint Black & Lyle Lovett and His Large Band . Jul 5, 2024 . Arrival From Sweden: The Music of Abba . Jul 6, 2024 . Daryl Hall & Elvis Costello and The Imposters . Jul 7, 2024 . The \nCharacterLimit: between 63 to 90 characters. \nKeywords: ravina#ravinia schedule#ravinia festival 2024#ravinia schedule 2024#ravinia festival schedule#ravinia festival#ravinia park schedule#ravinia pavilion#ravinia 2024#ravinia concerts \nInsight: Ensure relevance by including reasonable keywords in each description.",
                "Please generate 2 Ad Description in English language, based on the following information:\n\nFinalUrl: www.Temu.com \nDomain: Temu.com \nCategory: Sports & Entertainment -- Yoga & Pilates -- Yoga Mats \nDescriptionOfAdvertiser: Temu is a global platform owned by PDD Holdings that offers discounted goods. \nCharacterLimit: between 81 to 90 characters. \nInsight: Incorporate dynamic keyword insertion to make your ad more relevant to query. \n",
                "Please generate 5 Ad Headline in German language, based on the following information:\n\nFinalUrl: www.Alibaba.com \nDomain: Alibaba.com \nCategory: Sports & Entertainment -- Yoga & Pilates -- Yoga Mats \nDescriptionOfAdvertiser: Alibaba, a global B2B E-Commerce platform for wholesale sourcing, offers diverse products and services, connecting buyers and sellers worldwide. \nCharacterLimit: between 20 to 30 characters. \nInsight: Incorporate dynamic keyword insertion to make your ad more relevant to query. \n",
                ]
    
    test_data = ["Please generate 2 Ad Description in English language, based on the following information:\n\nFinalUrl: www.Temu.com \nDomain: Temu.com \nCategory: Sports & Entertainment -- Yoga & Pilates -- Yoga Mats \nDescriptionOfAdvertiser: Temu is a global platform owned by PDD Holdings that offers discounted goods. \nCharacterLimit: between 81 to 90 characters. \nInsight: Incorporate dynamic keyword insertion to make your ad more relevant to query. \n",
                "Please generate 5 Ad Headline in German language, based on the following information:\n\nFinalUrl: www.Alibaba.com \nDomain: Alibaba.com \nCategory: Sports & Entertainment -- Yoga & Pilates -- Yoga Mats \nDescriptionOfAdvertiser: Alibaba, a global B2B E-Commerce platform for wholesale sourcing, offers diverse products and services, connecting buyers and sellers worldwide. \nCharacterLimit: between 20 to 30 characters. \nInsight: Ensure diversity by highlighting various selling pionts in each Headline. \n",
                "Please generate 5 Ad Headline in German language, based on the following information:\n\nFinalUrl: www.Alibaba.com \nDomain: Alibaba.com \nCategory: Sports & Entertainment -- Yoga & Pilates -- Yoga Mats \nDescriptionOfAdvertiser: Alibaba, a global B2B E-Commerce platform for wholesale sourcing, offers diverse products and services, connecting buyers and sellers worldwide. \nCharacterLimit: between 20 to 30 characters. \nInsight: Incorporate dynamic keyword insertion to make your ad more relevant to query. \n",
                "Please generate 5 Ad Headline in German language, based on the following information:\n\nFinalUrl: www.Alibaba.com \nDomain: Alibaba.com \nCategory: Power Transmission -- Power Transmission Parts -- Power Transmission Parts \nDescriptionOfAdvertiser: Alibaba, a global B2B E-Commerce platform for wholesale sourcing, offers diverse products and services, connecting buyers and sellers worldwide. \nCharacterLimit: between 20 to 30 characters. \nInsight: Incorporate dynamic keyword insertion to make your ad more relevant to query. \n",
                "Please generate 5 Ad Headline in Simplified Chinese language, based on the following information:\n\nFinalUrl: www.Alibaba.com \nDomain: Alibaba.com \nCategory: Sports & Entertainment -- Amusement Park Products -- SensoryÂ TrainingÂ Equipment \nDescriptionOfAdvertiser: Alibaba, a global B2B E-Commerce platform for wholesale sourcing, offers diverse products and services, connecting buyers and sellers worldwide. \nCharacterLimit: between 20 to 30 characters. \nInsight: Ensure diversity by highlighting various selling pionts in each Headline. \n",
                "Please generate 5 Ad Headline in Simplified Chinese language, based on the following information:\n\nFinalUrl: www.Alibaba.com \nDomain: Alibaba.com \nCategory: Sports & Entertainment -- Amusement Park Products -- SensoryÂ TrainingÂ Equipment \nDescriptionOfAdvertiser: Alibaba, a global B2B E-Commerce platform for wholesale sourcing, offers diverse products and services, connecting buyers and sellers worldwide. \nCharacterLimit: between 20 to 30 characters. \nInsight: Incorporate dynamic keyword insertion to make your ad more relevant to query. \n",
                ]
    
    test_data = ["Please generate 8 Ad Headline in English language, based on the following information:\\n\\nFinalUrl: HTTP://www.autoeurope.co.uk/CAR-HIRE-PERPIGNAN-AIRPORT/?AFF=MSNUK&C=4024&PUCODE=FR&PUCITY=PERPIGNAN \\nLandingPage:   . Car Hire Perpignan Airporr – Save with Auto Europe; . CHEAP CAR HIRE AT PERPIGNAN AIRPORT; . Search and Compare Car Hire Rates . Car hire at Perpignan Airport - Gateway to the South of France . Airport Information . Where to pick up my car hire at Perpignan Airport . Terminals . Airport Facilities . How to get to and from Perpignan Airport . Parking Tips . Useful links; . Drop-off at a differen \\nCharacterLimit:  between 10 to 30 characters. \\nInsight: Promote the selling point: <24-hour support>.\\",
                 "Please generate 4 Ad Description in Chinese language, based on the following information:\\n\\nFinalUrl: http://2bridges.com.tw/ \\nLandingPage:   . 双橋日本留學-日本留學, 日本遊學, 日本留學代辦, 日本遊學代辦, 日本打工度假, 日本語言學校,日本專門學校, 日本大學申請, 日本研究所申請 , 大阪留學, 東京留學, 日本工作, 大阪遊學, 東京遊學,日本學生簽證申請;;;; . prev . next . 日本留學 Q&A . 日本打工度假 All in one 方案輕鬆成行去~ . 365天免費預約專線 TEL： (02)2311-7333 MAIL： two.bridges2@gmail.com . 日本留學, 日本遊學, 日本留學代辦, 日本遊學代辦, 日本打工度假, 日本語言學校,日本專門學校, 日本大學申請, 日本研究所申請 , 大阪留學, 東京留學, 日本工作, 大阪遊學, 東京遊學,日本學生簽證申請 \\nCharacterLimit:  between 60 to 90 characters. \\nInsight: Rewrite the existing ad <想去日本唸書擔心學費嗎?日本學校手續簡便不綁約，利息輕鬆不沉重，為未來開啟夢想的藍圖。> in the required tone <Inspiring>\\n",
                 "Please generate 4 Ad Description in English language, based on the following information:\\n\\nFinalUrl: HTTP://squeegeeclean4u.com  \\nCharacterLimit:  between 60 to 90 characters. \\nInsight: Promote the selling point: <Exceed expectations>.\\n",
                 "Please generate 4 Ad Theme in English language, based on the following information:\\n\\nFinalUrl: HTTPS://SHOPMELHORCOMPRAONLINE.COM  \\nCharacterLimit:  between 30 to 50 characters. \\nInsight: Generate diverse and appealing selling points.\\n",
                 "Please generate 8 Ad Headline in English language, based on the following information:\\n\\nFinalUrl: HTTPS://www.thesalvageyardsd.com  \\nCharacterLimit:  between 10 to 30 characters. \\nInsight: Promote the selling point: <Natural movement>.\\n",
                 "Please generate 4 Ad Description in Simplified Chinese language, based on the following information:\\n\\nFinalUrl: http://adsl.hcqixin.com.cn/ \\nLandingPage:  ;;;; . Sanlovalve . Product . Case . News . Contact Us . factories . LEARN MORE → . Copyright © 2024 上海上龙供水设备有限公司 沪ICP备18023868号-1 . 上海上龙供水设备有限公司是一家集合生产研发于一体的创新型中小企业，本部位于上海市普陀区，在上海、南通设有工厂，是供水行业控制阀的专业生产企业，拥有阀类技术多项，新型技术多项，是上海市技术成果转化企业和**试点企业，公司研发生产的“低阻力倒流防止器”获上海世博会中国馆供货商“詹天佑奖”，这是中国土木工程设立的奖项。公司还是中国城镇供水排水协会和全国建筑给排水协会的成员单位，先后参与编制给排水领域的国家标准和行业标准15项，其控制类阀门的技术水平跻身同行业 \\nCharacterLimit:  between 60 to 90 characters. \\nInsight: Generate similar ad for the existing ad <{KeyWord:倒流防止器}上龙供水，供水行业控制阀的专业生产企业，拥有阀类发明技术多项，实用新型技术多项>.\\n",
                 "Please generate 4 Ad Description in Simplified Chinese language, based on the following information:\\n\\nFinalUrl:http://api.bjweizhifu.com/apply/dyxcx/ \\nLandingPage: 微智服采购网，抖音小程序， 抖音B2B小程序 \\nCharacterLimit:  between 60 to 90 characters. \\nInsight: Promote the selling point: <限时优惠>.\\n",
                 "Please generate 8 Ad Headline in English language, based on the following information:\\n\\nFinalUrl: http://alpersteinanddiener.mydex.com/wOah6yUkFuY \\nLandingPage:   . Dex Media Reverse Proxy;;; . INVALID Reverse Proxy URL ACCESS! . (Failed to decryptSymmetric click id!); \\nCharacterLimit:  between 10 to 30 characters. \\nInsight: Generate similar ad for the existing ad <We'll Protect Your Rights>.\\n",
                 "Please generate 4 Ad Description in Dutch language, based on the following information:\\n\\nFinalUrl: http://apenheul.nl/apenheul30 \\nLandingPage:   . Apenheul tickets - Apenheul; . Kinderkaartjes online €15; . Winkelmandje; . Welkom in onze webshop! Bestel hier je tickets voor het leukste dagje uit tussen de apen. Online krijg je per ticket tot wel €7,50 korting op de kassaprijs. En deze zomervakantie bestel je jouw kinderkaartjes voor maar €15! Per bestelling wordt €0,65 administratiekosten in rekening gebracht. . Maak een keuze voor een da \\nCharacterLimit:  between 60 to 90 characters. \\nInsight: Generate similar ad for the existing ad <De nieuwe gorillaleider Banjoko ontmoeten. Nu tickets met 30% korting online.>.\\n",
                 "Please generate 5 Ad Headline in English language, based on the following information:\\n\\nFinalUrl: http://WowPartyRental.com \\nCharacterLimit:  between 10 to 30 characters. \\nInsight: Incorporate dynamic keyword insertion to make your ad more relevant to query.\\n", 
    ]
    
    test_data = ["Please generate 5 Ad Description in English language, based on the following information:\nFinalUrl: https://example.com/ \nCharacterLimit: between 71 to 90 characters. \nInsight: Rewrite the existing ad <Try Advanced Digital Hearing Aids Made for You. Start Your Journey Today.> based on the requirement <sasdasd>. \n",
                 "Please generate 3 Ad Description in Spanish language, based on the following information:\nFinalUrl: https://www.homedepot.com.mx/ \nCharacterLimit: between 64 to 90 characters. \nInsight: Rewrite the existing ad <Envío gratis desde $499 en línea - Redescubre tus espacios con las mejores marcas> based on the requirement <productos para todo el hogar>. \n",
                 "Please generate 8 Ad Headline in English language, based on the following information:\nFinalUrl: https://www.birakit.com/services/hometelephones.php \nCharacterLimit: between 10 to 30 characters. \nInsight: Rewrite the existing ad <Get VOIP Home> based on the requirement <providing voip telephone services for home use>. \n",
                 "Please generate 4 Ad Description in English language, based on the following information:\nFinalUrl: https://www.publicdatacheck.com/people/index \nCharacterLimit: between 61 to 90 characters. \nInsight: Rewrite the existing ad <Reduce Tenant Risks: Secure Your Property with an Eviction History Check> based on the requirement <View Your Work History>. \n",
                 "Please generate 4 Ad Description in English language, based on the following information:\nFinalUrl: https://healthexpertspro.com/different-types-of-coffees/ \nCharacterLimit: between 51 to 90 characters. \nInsight: Rewrite the existing ad <Coffee bridges gaps across the globe with its unity> in the required tone <Persuasive> \n",
                 "Please generate 8 Ad Headline in German language, based on the following information:\nFinalUrl: https://www.benomik.com/de-industries/bauindustrie \nCharacterLimit: between 10 to 30 characters. \nInsight: Rewrite the existing ad <Beratung auf Augehöhe für B2B> in the required tone <Friendly> \n",
                 "Please generate 8 Ad Headline in English language, based on the following information:\nFinalUrl: https://www.computeroisd.com \nCharacterLimit: between 10 to 30 characters. \nInsight: Promote the selling point: <Boost PC performance> and reflect the highlight information: <Printer and Driver update tool>. \n",
                 "Please generate 5 Ad Description in English language, based on the following information:\nFinalUrl: https://shirtmellow.com/ \nCharacterLimit: between 56 to 90 characters. \nInsight: Promote the selling point: <Comfy T-shirts> and reflect the highlight information: <t shirt sale>. \n",
                 "Please generate 3 Ad Description in Simplified Chinese language, based on the following information:\nFinalUrl: https://www.fasoft.cn \nLandingPage:  . 金甲U盘文件加密;  . 金甲企业数据加密软件，保护企业数据安全，对Windows平台下任意格式的文件... . 金甲U盘加密软件把普通U盘改造成打开文件的的Key，>达到控制授权文档的阅读...;  . 武汉市风奥科技股份有限公司—中国数据防泄漏领域的专家。文件加密,加密软件,文档加密,数据加密,视频加密、文件夹加密软件、企业加密软件首选品牌，风奥科技是国内最早的文件加密软件厂商,拥有遍>布全国和部分海外地区的企业级用户,迄今每天有数百万台终端使用风奥的软件产品进行安全控制.全国热线：027-87409509 \nCharacterLimit: between 26 to 90 characters. \nInsight: Promote the selling point: <申请试用> and reflect the highlight information: <加密软件排名>. \n"
    ]
    '''
    test_data = ["who are you?"

    ]
    
    test_data = [data.strip() for data in test_data]

    test = ModelImp(args)
    #print(test.EvalBatch(test_data))
    
    for data in test_data:
        #print('Input: '+ data)
        start_time = time.time()
        print('Output: '+ test.Eval(data))
        end_time = time.time()
        print('inference time: '+ str(end_time - start_time))
        print('\n')
    
            
