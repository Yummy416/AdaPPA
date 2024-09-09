import warnings
warnings.filterwarnings('ignore')
import transformers
transformers.logging.set_verbosity_error()
import sys
sys.path.append('/llj/red_team')
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

from tqdm import tqdm
from collections import defaultdict

from py_file.small_module import read_jsonl_file, write_jsonl_file
from py_file.model_chat import llama_guard2_toxigen, llama2_13b_kaiyuan

if __name__ == '__main__':
    datas = read_jsonl_file('/llj/red_team/data/train.jsonl')
    
    results = []
    num = defaultdict(int)
    for data in tqdm(datas):
        prompt = data['prompt']
        category = None
        for scene in data['category'].keys():
            if data['category'][scene] == True:
                category = scene
                break
        if category is None or num[category] > 1000:
            continue
        response = llama2_13b_kaiyuan(prompt)
        score = llama_guard2_toxigen(prompt, response)
        if score == 0:
            results.append({"prompt": prompt, "response": response, "category": category})
            write_jsonl_file('/llj/red_team/data/safe_response.jsonl', results)
            num[category] += 1