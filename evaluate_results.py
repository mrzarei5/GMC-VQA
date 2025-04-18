import os
import json
from utils import evaluate_answer, complete_yn_question
import argparse

import openai

def evaluate(save_path):
    print('started')

    path = os.path.join(save_path, 'results.json')
    with open(path, 'r') as f:
        export_dic = json.load(f)
    correct = 0
    overall = 0
    for id, data_dic in export_dic.items():
        matched = data_dic.get('answer_matched',None)
        if matched:
            if matched == 'yes':
                correct+=1
            overall+=1
            continue
        answer = data_dic['answer_label']
        if answer in ['yes','no']:
            answer = complete_yn_question(data_dic['question'], answer=answer, model = 'gpt-4o-mini', temp_gpt= 0)
        matched, verification = evaluate_answer(data_dic['question'], answer, data_dic['predict_answer'], model = 'gpt-4o-mini', temp_gpt= 0)
        if matched:
            correct+= 1
            data_dic['answer_matched'] = 'yes'
        else:
            data_dic['answer_matched'] = 'no'
        data_dic['verification'] = verification
        overall += 1
        with open(path, "w") as outfile:
            json.dump(export_dic, outfile, indent = 2) 
    print('model:',save_path, ' correct prediction: ', correct, ' overall number:', overall, ' acc:', correct/overall)
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation Args.')
    parser.add_argument('--exp_tag', type=str, default='experiment1')
    parser.add_argument('--save_root', type=str, default='./exp_result/')
    parser.add_argument('--dataset', type=str, default='vqav2')
    parser.add_argument('--openai_key', type=str,  default='', 
                        help='OpenAI Key for GPT-3.5/GPT4-o Mini API')
    
    args = parser.parse_args()
    save_path = os.path.join(args.save_root, f'{args.dataset}_{args.exp_tag}', 'result')
    
    OPENAI_API_KEY = args.openai_key
    openai.api_key = OPENAI_API_KEY
    
    evaluate(save_path)