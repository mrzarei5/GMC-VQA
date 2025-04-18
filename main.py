import os
import yaml
import argparse
import torch

import openai
from tqdm import tqdm
import pdb


from data import VQAV2Sampler
from data import OKVQASampler
from data import STVQASampler

from chat import OKVQAConversationTwoAgent
from chat import STVQAConversationTwoAgent

from chat import VQAV2ConversationTwoAgent
import random

from utils import write_output, is_done

def GmcVqa(vqa_model, dataset, data_ids, model, save_path='', max_n_rounds=5, print_mode='no', prompt_setting='v1a', temp_gpt=0.0, description_extend = False, objects_prompt_type = 'question'):
    """
    Conduct IdealGPT conversation

    Args:
        vqa_model : vqa model.
        dataset: the dataset used to caption
        data_ids (list): a list of sample ids in the dataset
        model (str): the model name used to ask question. Valid values are 'chatgpt', 'gpt4', 'gpt4o_mini'.
        save_path (str): the path to save caption results. If it is empty, results are not being saved.
        max_n_rounds (int): the max number of chat rounds
        print_mode (str): print mode. 'chat' for printing everything. 'bar' for printing everything but the chat process. 'no' for no printing
    """
    if model == 'chatgpt':
        model = 'gpt-3.5-turbo'
    elif model =='gpt4':
        model = 'gpt-4'
    elif model == 'gpt4o_mini':
        model = 'gpt-4o-mini'

    result_path = os.path.join(save_path, 'result')
    for data_id in tqdm(data_ids, disable=print_mode!='no'):
        if print_mode != 'no':
            print('Data ID {}'.format(data_id))

        #check if the sample has already been processed
        if is_done(data_id,result_path): 
            continue
        
        dataset_types = [VQAV2Sampler,OKVQASampler, STVQASampler]

     
        if type(dataset) in dataset_types:
            image_path, qa = dataset.fetch_data(data_id)
            info = {'setting':
                        {
                        'id': data_id,
                        'question': qa['question'].strip(),
                        'answer_label': str(qa['answer_label']) if 'answer_label' in qa else None,
                        'max_n_rounds': max_n_rounds,
                        'img_path': qa['img_path'] if 'img_path' in qa else None
                        }
                }
            caption = None

        results = {}
        # Initialize VQA Instance.
        if type(dataset) == VQAV2Sampler:
            chat = VQAV2ConversationTwoAgent(img=image_path,
                                vqa_model=vqa_model,
                                model=model,
                                question=info['setting']['question'],
                                label= str(info['setting']['answer_label']),
                                data_id=data_id,
                                prompt_setting=prompt_setting,
                                caption=caption,
                                temp_gpt=temp_gpt,
                                description_extend = description_extend,
                                objects_prompt_type = objects_prompt_type, temp_name = result_path)

        elif type(dataset) == OKVQASampler:
            chat = OKVQAConversationTwoAgent(img=image_path,
                                vqa_model=vqa_model,
                                model=model,
                                question=info['setting']['question'],
                                label= str(info['setting']['answer_label']),
                                data_id=data_id,
                                prompt_setting=prompt_setting,
                                caption=caption,
                                temp_gpt=temp_gpt,
                                description_extend = description_extend,
                                objects_prompt_type = objects_prompt_type, temp_name = result_path)
        elif type(dataset) == STVQASampler:
            chat = STVQAConversationTwoAgent(img=image_path,
                                vqa_model=vqa_model,
                                model=model,
                                question=info['setting']['question'],
                                label= str(info['setting']['answer_label']),
                                data_id=data_id,
                                prompt_setting=prompt_setting,
                                caption=caption,
                                temp_gpt=temp_gpt,
                                description_extend = description_extend,
                                objects_prompt_type = objects_prompt_type, temp_name = result_path)


        used_round = chat.chatting(max_n_rounds, print_mode=print_mode)
        
        results['question'] = chat.question
        results['answer_label'] = str(info['setting']['answer_label'])
        results['predict_answer'] = chat.answer_predict
        results['used_round'] = used_round
        results['sub_images_data'] = chat.sub_images_data
        results['analysis'] = chat.analysis
        
        if hasattr(chat,'objects'):
            results['objects'] = chat.objects

                
        write_output(data_id, results, chat.sub_images_data, result_path)

    return


def parse():
    parser = argparse.ArgumentParser(description='GMC-VQA Args.')
    parser.add_argument('--data_root', type=str, default='/home/student/storage/datasets/vqav2', 
                        help='root path to the dataset')
    parser.add_argument('--save_root', type=str, default='./exp_result/', 
                        help='root path for saving results')
    parser.add_argument('--sample_number', type=int, default=2000,
                        help='Number of images to sample in experiments')
    parser.add_argument('--exp_tag', type=str, required=True, 
                        help='tag for this experiment. caption results will be saved in save_root/exp_tag')
    parser.add_argument('--dataset', type=str, default='vqav2',
                        help='Names of the dataset to use in the experiment. Valid datasets include vqav2, okvqa and stvqa. Default is vqav2')
    parser.add_argument('--max_n_rounds', type=int, default=4,
                        help='Max Number of QA rounds between LLM and VLM. Default is 4.')
    parser.add_argument('--model', type=str, default='gpt4o_mini', choices=['chatgpt', 'gpt4', 'gpt4o_mini'],
                        help='model used to ask question. can be gpt3, chatgpt, or its concrete tags in openai system')
    parser.add_argument('--vqa_model', type=str, default='blip2_t5_xl', choices=['blip2_t5_xl', 'llava', 'instruct_blip'],
                        help='model as Answerer.')
    parser.add_argument('--device_id', type=int, default=0, 
                        help='Which GPU to use.')
    parser.add_argument('--prompt_setting', type=str,  default='v1a', 
                        help='Prompt Setting Version')
    parser.add_argument('--extend_description', action='store_true', help='Flag to set for extending descriptions')
    parser.add_argument('--objects_prompt_type', type=str, default= 'question', help='Prompt to use for object detection.', choices=['question','objects'])
    parser.add_argument('--openai_key', type=str,  default='', 
                        help='OpenAI Key for GPT-3.5/GPT4-o Mini API')
    parser.add_argument('--temp_gpt', type=float,  default=0.0, 
                        help='Temperature for GPT')
    parser.add_argument('--temp_vqa', type=float,  default=0.001, 
                        help='Temperature for VQA model, must be positive')
    parser.add_argument('--seed', type=int, default=3, help='random seed')
    args = parser.parse_args()
    return args
    
    
def main(args):
    # Set OpenAI
    OPENAI_API_KEY = args.openai_key
    openai.api_key = OPENAI_API_KEY

    random.seed(args.seed)

    # load the dataset
    if 'vqav2' in args.dataset:
        if 'train' in args.dataset:
            dataset = VQAV2Sampler(args.data_root, dataSubType= 'train', data_num = args.sample_number)
        else:
            dataset = VQAV2Sampler(args.data_root, dataSubType= 'val', data_num = args.sample_number)

    elif 'okvqa' in args.dataset:
        if 'train' in args.dataset:
            dataset = OKVQASampler(args.data_root, dataSubType= 'train', data_num = args.sample_number)
        else:
            dataset = OKVQASampler(args.data_root, dataSubType= 'val', data_num = args.sample_number)

    elif 'stvqa' in args.dataset:
        if 'test' in args.dataset:
            dataset = STVQASampler(args.data_root, dataSubType= 'test', data_num = args.sample_number)
        else:
            dataset = STVQASampler(args.data_root, dataSubType= 'train', data_num = args.sample_number)
    print('Finish loading data')

    print('Start loading VQA model')
    if 'blip2' in args.vqa_model:
        from lib.blip2_lib import Blip2Lavis
        vqa_model = Blip2Lavis(name="blip2_t5", model_type="pretrain_flant5xl", device=torch.device("cuda:{}".format(args.device_id)))
    elif 'llava' in args.vqa_model:
        from lib.llava_lib import LLAVA
        vqa_model = LLAVA(temperature=args.temp_vqa)
    elif 'instruct_blip' in args.vqa_model:
        from lib.instructblip_lib import INSTRUCT_BLIP
        vqa_model = INSTRUCT_BLIP(device=torch.device("cuda:{}".format(args.device_id)))
    else:
        raise NotImplemented(f'{args.vqa_model} not supported')
    print('Finish loading VQA model {}'.format(args.vqa_model))
    
    question_model = args.model

    # preparing the folder to save results
    save_path = os.path.join(args.save_root, f'{args.dataset}_{args.exp_tag}')
    if not os.path.exists(save_path):
        os.makedirs(os.path.join(save_path, 'result'))
    with open(os.path.join(save_path, 'args.yaml'), 'w') as f:
        yaml.dump(vars(args), f)

    # start Conversation
    GmcVqa(vqa_model,
                dataset, 
                dataset.ids, 
                save_path=save_path, 
                max_n_rounds=args.max_n_rounds, 
                model=question_model,
                print_mode='no',
                prompt_setting=args.prompt_setting,
                temp_gpt=args.temp_gpt,
                description_extend=args.extend_description,
                objects_prompt_type = args.objects_prompt_type)
    

if __name__ == '__main__':
    args = parse()
    main(args)
