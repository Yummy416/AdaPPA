import warnings
warnings.simplefilter("ignore")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, RobertaTokenizer, RobertaModel, BertTokenizer, BertModel
from transformers import MarianMTModel, MarianTokenizer

def load_model_baichuan2_13b():
    model_dir_baichuan2_13b = '/LLM/Baichuan2-13B-Chat'
    tokenizer_baichuan2_13b = AutoTokenizer.from_pretrained(model_dir_baichuan2_13b, trust_remote_code=True, use_fast=False)
    model_baichuan2_13b = AutoModelForCausalLM.from_pretrained(model_dir_baichuan2_13b, device_map="cuda:0", trust_remote_code=True, torch_dtype=torch.float16)
    model_baichuan2_13b.generation_config = GenerationConfig.from_pretrained(model_dir_baichuan2_13b)
    return tokenizer_baichuan2_13b, model_baichuan2_13b

def load_model_baichuan2_7b():
    model_dir_baichuan2_13b = '/LLM/Baichuan2-7B-Chat/models--baichuan-inc--Baichuan2-7B-Chat/snapshots/ea66ced17780ca3db39bc9f8aa601d8463db3da5'
    tokenizer_baichuan2_13b = AutoTokenizer.from_pretrained(model_dir_baichuan2_13b, trust_remote_code=True, use_fast=False)
    model_baichuan2_13b = AutoModelForCausalLM.from_pretrained(model_dir_baichuan2_13b, device_map="cuda:0", trust_remote_code=True, torch_dtype=torch.float16)
    model_baichuan2_13b.generation_config = GenerationConfig.from_pretrained(model_dir_baichuan2_13b)
    return tokenizer_baichuan2_13b, model_baichuan2_13b

# Vicuna-13B
def load_model_vicuna_13b():
    model_dir_vicuna = '/LLM/vicuna-13b-v1.5/models--lmsys--vicuna-13b-v1.5/snapshots/c8327bf999adbd2efe2e75f6509fa01436100dc2'
    tokenizer_vicuna = AutoTokenizer.from_pretrained(model_dir_vicuna, trust_remote_code=True, use_fast=False)
    model_vicuna = AutoModelForCausalLM.from_pretrained(model_dir_vicuna, device_map="cuda:0", trust_remote_code=True, torch_dtype=torch.float16)
    model_vicuna.generation_config = GenerationConfig.from_pretrained(model_dir_vicuna)
    return tokenizer_vicuna, model_vicuna

# Vicuna-7B
def load_model_vicuna_7b():
    model_dir_vicuna = '/LLM/vicuna-7b-v1.5/models--lmsys--vicuna-7b-v1.5/snapshots/3321f76e3f527bd14065daf69dad9344000a201d'
    tokenizer_vicuna = AutoTokenizer.from_pretrained(model_dir_vicuna, trust_remote_code=True, use_fast=False)
    model_vicuna = AutoModelForCausalLM.from_pretrained(model_dir_vicuna, device_map="cuda:0", trust_remote_code=True, torch_dtype=torch.float16)
    model_vicuna.generation_config = GenerationConfig.from_pretrained(model_dir_vicuna)
    return tokenizer_vicuna, model_vicuna

# Llama2-13B
def load_model_llama2_13b():
    model_dir_llama = '/LLM/Llama-2-13b-chat-hf/models--meta-llama--Llama-2-13b-chat-hf/snapshots/a2cb7a712bb6e5e736ca7f8cd98167f81a0b5bd8'
    tokenizer_llama = AutoTokenizer.from_pretrained(model_dir_llama, trust_remote_code=True, use_fast=False)
    model_llama = AutoModelForCausalLM.from_pretrained(model_dir_llama, device_map="cuda:0", trust_remote_code=True, torch_dtype=torch.float16)
    model_llama.generation_config = GenerationConfig.from_pretrained(model_dir_llama)
    return tokenizer_llama, model_llama

# Llama2-7B
def load_model_llama2_7b():
    model_dir_llama = '/LLM/Llama-2-7b-chat-hf/models--meta-llama--Llama-2-7b-chat-hf/snapshots/f5db02db724555f92da89c216ac04704f23d4590'
    tokenizer_llama = AutoTokenizer.from_pretrained(model_dir_llama, trust_remote_code=True, use_fast=False)
    model_llama = AutoModelForCausalLM.from_pretrained(model_dir_llama, device_map="cuda:0", trust_remote_code=True, torch_dtype=torch.float16)
    model_llama.generation_config = GenerationConfig.from_pretrained(model_dir_llama)
    return tokenizer_llama, model_llama

# ChatGLM3-6B
def load_model_chatglm3_6b():
    model_dir_chatglm = '/LLM/chatglm3-6b'
    tokenizer_chatglm = AutoTokenizer.from_pretrained("THUDM/chatglm3-6b", cache_dir=model_dir_chatglm, trust_remote_code=True, use_fast=False)
    model_chatglm = AutoModelForCausalLM.from_pretrained("THUDM/chatglm3-6b", cache_dir=model_dir_chatglm, device_map="cuda:0", trust_remote_code=True, torch_dtype=torch.float16)
    return tokenizer_chatglm, model_chatglm

def load_model_similarity_en():
    model_dir_similarity = '/LLM/sup-simcse-roberta-large/models--princeton-nlp--sup-simcse-roberta-large/snapshots/96d164d9950b72f4ce179cb1eb3414de0910953f'
    tokenizer_similarity = RobertaTokenizer.from_pretrained(model_dir_similarity)
    model_similarity = RobertaModel.from_pretrained(model_dir_similarity)
    return tokenizer_similarity, model_similarity

def load_model_llama_guard2():
    model_dir = '/LLM/Meta-Llama-Guard-2-8B/models--meta-llama--Meta-Llama-Guard-2-8B/snapshots/7d257f3c1a0ec6ed99b2cb715027149dfb9784ef'
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16, device_map='cuda:0')
    return tokenizer, model

def load_model_llama_guard3():
    model_dir = '/LLM/Llama-Guard-3-8B/models--meta-llama--Llama-Guard-3-8B/snapshots/66d920a7564151b86b039c197f3cf4394707bdf2'
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16, device_map='cuda:0')
    return tokenizer, model

def load_model_llama3_8b():
    model_dir_llama = '/LLM/Meta-Llama-3-8B-Instruct/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/e1945c40cd546c78e41f1151f4db032b271faeaa'
    tokenizer_llama = AutoTokenizer.from_pretrained(model_dir_llama, trust_remote_code=True, use_fast=False)
    model_llama = AutoModelForCausalLM.from_pretrained(model_dir_llama, device_map="cuda:0", trust_remote_code=True, torch_dtype=torch.float16)
    model_llama.generation_config = GenerationConfig.from_pretrained(model_dir_llama)
    return tokenizer_llama, model_llama
