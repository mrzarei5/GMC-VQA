import json
import os
import re
import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff



def is_done(id,save_path):
    path = os.path.join(save_path, 'results.json')
    if os.path.exists(path):
        with open(path, 'r') as f:
            export_dic = json.load(f)
        data = export_dic.get(str(id),None)
        if data:
            return True
        return False
    else:
        return False

def write_output(id, data, sub_images_data, save_path):
    export_dic = {}
    path = os.path.join(save_path, 'results.json')
    if os.path.exists(path):
        with open(path, 'r') as f:
            export_dic = json.load(f)
    
    sample_directory = os.path.join(save_path, str(id))
    if not os.path.exists(sample_directory):
        os.makedirs(sample_directory)
    image_counter = 1

    for sub_image_data in sub_images_data:
        image = sub_image_data['img']
        image_directory = os.path.join(sample_directory, str(image_counter)+'.jpg')
        image.save(image_directory)
        image_counter += 1
        sub_image_data['img'] = image_directory
        
        sub_questions = sub_image_data['sub_questions']
        sub_answers = sub_image_data['sub_answers']

        q_ans = []

        flat_sub_questions = []
        for sub_questions_i in sub_questions:
            flat_sub_questions.extend(sub_questions_i)
        flat_sub_answers = []
        for sub_answers_i in sub_answers:
            flat_sub_answers.extend(sub_answers_i)
        
        for i,q in enumerate(flat_sub_questions):
            q_ans.append(q+ ' '+ flat_sub_answers[i])
        sub_image_data.pop('sub_questions')
        sub_image_data.pop('sub_answers')
        sub_image_data['supp_questions'] = q_ans
    
    data['sub_images_data'] = sub_images_data
    
    export_dic[id] = data
    

    with open(path, "w") as outfile:
        json.dump(export_dic, outfile, indent = 2) 

@retry(wait=wait_random_exponential(min=0.1, max=0.2), stop=stop_after_attempt(10))
def call_gpt(chatgpt_messages, model="gpt-3.5-turbo", temp_gpt=0.0):
    response = openai.ChatCompletion.create(model=model, messages=chatgpt_messages, temperature=temp_gpt, max_tokens=512)
    reply = response['choices'][0]['message']['content']
    total_tokens = response['usage']['total_tokens']
    return reply, total_tokens


def parse_final_answer(gpt_response):
    # ===== Parse the paragraph starting with verification. =====
    verification_result = re.search('Verification:(.*)\n', gpt_response)
    if verification_result:
        verification_string = verification_result.group(1).strip()
    else:
        print(f'Can not parse verification from {gpt_response}')
        raise ValueError

    
    substring = "Final decision on verification:"

    pattern = f"{re.escape(substring)}(.*?)(?=\n|$)"
    matches = re.findall(pattern, gpt_response)
    if matches:
        answer_string = matches[-1].strip()
        # In middle rounds, detect 'not sure' at first.
        if 'yes' in answer_string.lower():
            answer_string = 'yes'
        elif 'no' in answer_string.lower():
            answer_string = 'no'
        else:
            answer_string = None
    else:
        print(f'Can not parse final decision on verification from {gpt_response}')
        raise ValueError
    return verification_string, answer_string

def parse_final_answer_rerun(gpt_response):
    try:
        verification_string, answer_string = parse_final_answer(gpt_response=gpt_response)
        need_rerun = False
        return verification_string, answer_string, need_rerun
    except:
        need_rerun = True
        return None, None, need_rerun


def evaluate_answer(question, ground_truth, predicted_answer, model, temp_gpt=0.0):
    prompt = '''You will be given a question, an answer predicted by a LLM to that question and a ground truth answer for the question. The answer predicted by LLM may differ in language and grammatical structure compared to the ground truth answer. It may also be accompanied by additional explanations.
        Your goal is:
        You need to verify whether the ground truth answer to the main question can be inferred from the predicted answer and matches it or not. You have to consider that the predicted answer may be different but with the same meaning. It may also have some information that is not presented in the ground truth answer. In that case, the predicted answer is acceptable.  
        
        Here are the rules you should follow in your response:
        1. At first, demonstrate your verifying process in one or two sentences. Start with the format of "Verification:".
        2. From the verification process, Tell me whether the the predicted answer matches with the ground truth answer or not by providing only one word, either ‘yes’ or ‘no’. Respond with ‘yes’ if the predicted answer matches with the ground truth answer and return ‘no’ if it does not match with the ground truth answer. Respond in the format of "Final decision on verification: yes/no" 


        Response Format:

        Verification: xxxxxx.

        Final decision on verification: yes/no.
        '''
    input_prompt = 'Question: {}\n Predicted answer: {}\n Ground truth answer:{}\n'.format(question,predicted_answer,ground_truth)
    user_prompt = '''{} 
    Please follow the above-mentioned instructions to decide on whether the predicted answer matches with the ground truth answer or not.'''.format(input_prompt)
    

    request_prompts = [{"role": "system", "content": prompt},
                    {'role': 'user', 'content': user_prompt}]
   
    try_num = 0
    max_try = 15
    # Parse predicted answer from GPT output if any.
    while try_num < max_try:
        try_num += 1
        if try_num > max_try/2:
            cur_temp_gpt = temp_gpt + 0.1 * ((2.0*try_num/max_try)-1)
        else:
            cur_temp_gpt = temp_gpt
        gpt_response, n_tokens = call_gpt(request_prompts, model=model, temp_gpt=cur_temp_gpt)
        cur_verification, gpt_answer, need_rerun = parse_final_answer_rerun(gpt_response)
        if not need_rerun:
            break
        else:
            if try_num == max_try:
                raise ValueError('Rerun too many times, still failed in parsing.')
            else:
                print(f'Parsing failed, Time {try_num} of Rerun GPT Decision.')

    if 'yes' in gpt_answer.lower():
        return True, cur_verification
    elif 'no' in gpt_answer.lower():
        return False, cur_verification
    
def complete_yn_question(question,answer, model, temp_gpt=0.0):
    prompt = '''You will be given a question and a one word answer, either 'yes' or 'no'
    Your goal: 
    Turn the answer to a complete sentence using the text of the question and return the complete sentence answer.

    Example 1:
    Question: Is the dog being aggressive?
    Answer: yes

    Expected output: yes, the dog is being aggressive.

    Example 2:
    Question: Are they in a bathtub?
    Answer: no

    Expected output: no, they are not in a bathtub.
    '''
    input_prompt = 'Question: {}\n Answer: {}\n'.format(question,answer)
    user_prompt = '''{} 
    Please follow the above-mentioned instructions to provide the complete answer.'''.format(input_prompt)
    

    request_prompts = [{"role": "system", "content": prompt},
                    {'role': 'user', 'content': user_prompt}]
    
    gpt_response, n_tokens = call_gpt(request_prompts, model=model, temp_gpt=temp_gpt)
    return gpt_response