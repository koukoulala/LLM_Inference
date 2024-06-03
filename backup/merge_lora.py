#### COMMENT IN TO MERGE PEFT AND BASE MODEL ####
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
import torch
 
# Load PEFT model on CPU
checkpoint = "/data/xiaoyukou/LLaMA-Factory/saves/mistral/fsdp_qlora_sft/"
model = AutoPeftModelForCausalLM.from_pretrained(
    checkpoint,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)
# Merge LoRA and base model and save
merged_model = model.merge_and_unload()
output_dir = "./output/Mistral-7B-sft-fsdp-qlora-2"
merged_model.save_pretrained(output_dir, safe_serialization=True, max_shard_size="2GB")

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenizer.save_pretrained(output_dir)
