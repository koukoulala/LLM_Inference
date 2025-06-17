import json
import pandas as pd
from vllm import LLM, SamplingParams
from typing import Dict
import time
import os
import argparse
import numpy as np
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO)

import random
random.seed()
seed = random.randint(0,10000)
print(seed)
# print(random.random())

# seed=np.random.seed(42)
# print(seed)


try:
	# noinspection PyUnresolvedReferences
	import triton_python_backend_utils as pb_utils
except ImportError:
	pass  # triton_python_backend_utils exists only inside Triton Python backend.


class ModelImpl:
	"""Implement the model specific logics here, including data preprocess,
	model inference, data postproces, etc.
	"""
	def __init__(self):
		
		logger.info("Initializing Offline Inference with VLLM.")
		#print("args:", args)
		cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
		if cuda_visible_devices:
			tensor_parallel_size = len(cuda_visible_devices.split(','))
		else:
			tensor_parallel_size = 1
		logger.info(f"Number of GPUs specified by CUDA_VISIBLE_DEVICES: {tensor_parallel_size}")
		
		self.data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output/Mistral-7B-sft-add-copilot-gptq/")
		
		print(self.data_dir)
		print(os.listdir(self.data_dir))
		model_path = self.data_dir
		quantization = None 
		gpu_memory_utilization = 0.9
		dtype = 'float16'
		temperature = 0.95
		top_p = 0.7
		max_tokens = 500
		repetition_penalty = 1.0		

		self.llm = LLM(model= model_path, quantization=quantization,
					   tensor_parallel_size=tensor_parallel_size, gpu_memory_utilization=gpu_memory_utilization, dtype=dtype, seed=seed)
		
		self.sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens, repetition_penalty=repetition_penalty)
		print("===================sampling===============")
		print(self.sampling_params.sampling_type)


	def execute(self, requests):
		outputs = self.llm.generate(requests, self.sampling_params)
		responses = [output.outputs[0].text for output in outputs]
		return responses

	def finalize(self):
		logger.info("Cleaning up...")


class TritonPythonModel:
	"""Do not change this class name. Every model must have this class name, so
	the server python backend could load this class as the model entrance.
	"""

	def initialize(self, args: Dict[str, str]) -> None:
		"""`initialize` is called only once when the model is being loaded.
		Implementing `initialize` function is optional. This function allows
		the model to initialize any state associated with this model.

		Parameters
		----------
		args : dict
		  Both keys and values are strings. The dictionary keys and values are:
		  * model_config: A JSON string containing the model configuration
		  * model_instance_kind: A string containing model instance kind
		  * model_instance_device_id: A string containing model instance device ID
		  * model_repository: Model repository path
		  * model_version: Model version
		  * model_name: Model name
		"""
		print("Initializing...")
		self.model = ModelImpl()

	def execute(self, requests):
		"""`execute` must be implemented in every Python model. `execute`
		function receives a list of pb_utils.InferenceRequest as the only
		argument. This function is called when an inference is requested
		for this model.

		Parameters
		----------
		requests : list
		  A list of pb_utils.InferenceRequest

		Returns
		-------
		list
		  A list of pb_utils.InferenceResponse. The length of this list must
		  be the same as `requests`
		"""
		try:
			queries = []
			for request in requests:
				queries.append(
					pb_utils.get_input_tensor_by_name(request, "raw_input")
					.as_numpy()
					.tolist()[0][0].decode("utf-8") # [0][0] if dynamic batch else [0]
				)
			model_outputs = self.model.execute(queries)
		except Exception as e:
			model_outputs = [str(e)] * len(requests)

		if len(model_outputs) != len(requests):
			model_outputs = ["Bad implementation!"] * len(requests)

		responses = []
		for model_output in model_outputs:
			output = pb_utils.Tensor("output", np.array(model_output, dtype=object))
			responses.append(pb_utils.InferenceResponse([output]))
		return responses
	
	def finalize(self):
		"""`finalize` is called only once when the model is being unloaded.
		Implementing `finalize` function is optional. This function allows
		the model to perform any necessary clean ups before exit.
		"""
		print("Cleaning up...")

if __name__=='__main__':
	
			
	prompt1 = "[INST] Please generate 4 Ad Headline in Swedish language, based on the following information:\\n\\nFinalUrl: https://www.ogonlasern.se/behandlingar/ \nLandingPage:   . Våra ögonbehandlingar och ögonoperationer | Ögonlasern; . Våra behandlingar och ögonoperationer; . Pris för ögonbehandlingar och ögonoperationer . Ögonoperationer i Stockholm sedan 1990 . Våra mest populära synkorrigeringar . Ålder vid ögonoperationer och ögonlaserbehandlingar . LASEK - Ögonlaserbehandling . RLE – Linsbyte vid ålderssynthet . ICL - Linsimplantat . Gråstarrsoperation . Vitrektom \\nCharacterLimit: between 10 to 30 characters. \\nInsight: Promote the selling point: <LASIK - synkorrigering>.\\n [/INST]"
	prompt2 = "[INST] Please generate 4 Ad Headline in Danish language, based on the following information:\\n\\nFinalUrl: https://topdanskecasinos.com/slots/ #N#LandingPage:   . Bedste online casinoer;; . Dyk ned i det ultimative casino-eventyr: Oplev uovertrufne velkomsttilbud, ekspertanmeldelser og hurtige udbetalinger i Danmarks bedste casinoer! . Sidste opdatering: september 2024 . Top casino sider i 2024 . Din Pålidelige Casino Guide . Hvorfor Du Kan Stole på Os . Sådan Evaluerer Vi Casinoer . Sådan Vælger Du et Casino . Sikkerhed . Ofte stillede spørgsmål . Hvord \\nCharacterLimit: between 10 to 30 characters. \\nInsight: Promote the selling point: <Sikkerhed og fairness>.\\n [/INST]"
	test_data = []
	test_data.append(prompt1)
	test_data.append(prompt2)
	test = ModelImpl()
	responses = test.execute(test_data)
	print("=================== responses ======================")
	for idx, response in enumerate(responses):
		print("\nprompt:", test_data[idx])
		print("response:", response)