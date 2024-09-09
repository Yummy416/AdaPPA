import warnings
warnings.filterwarnings('ignore')
import transformers
transformers.logging.set_verbosity_error()
import sys
sys.path.append('/llj/red_team')

from tqdm import tqdm
from py_file.small_module import read_jsonl_file, write_jsonl_file

if __name__ == '__main__':
    datas_1 = read_jsonl_file('/llj/red_team/data/safe_response.jsonl')
    datas_2 = read_jsonl_file('/llj/red_team/data/harmful_response.jsonl')

    results = []
    for data_1 in tqdm(datas_1):
        for data_2 in datas_2:
            if data_2['prompt'] == data_1['prompt'] and data_2['category'] == data_1['category']:
                results.append({'prompt': data_1['prompt'], 'safe_response': data_1['response'], 'harmful_response': data_2['response'], 'category': data_1['category']})
                break

    write_jsonl_file('/llj/red_team/data/safe_harmful_response.jsonl', results)
                
    
    