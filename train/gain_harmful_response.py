import warnings
warnings.filterwarnings('ignore')
import transformers
transformers.logging.set_verbosity_error()
import sys
sys.path.append('/llj/red_team')

from tqdm import tqdm
from py_file.small_module import read_jsonl_file, write_jsonl_file

if __name__ == '__main__':
    datas_1 = read_jsonl_file('/llj/red_team/data/train.jsonl')
    datas_2 = read_jsonl_file('/llj/red_team/data/safe_response.jsonl')

    results = []
    for data_2 in tqdm(datas_2):
        for data_1 in datas_1:
            if data_2['prompt'] == data_1['prompt'] and data_1['category'][data_2['category']] == True:
                results.append({'prompt': data_1['prompt'], 'response': data_1['response'], 'category': data_2['category']})
                break

    write_jsonl_file('/llj/red_team/data/harmful_response.jsonl', results)
                
    
    