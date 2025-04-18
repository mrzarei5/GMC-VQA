
import re
from copy import deepcopy
from utils import call_gpt
from prompts.decrtiption_prompt import *
import json
import os
from PIL import Image

class ExtendDescriptionTwoAgent():
    def __init__(self,vqa_model, model, vqa_prompt,  n_rounds = 2, temp_gpt=0):
        self.vqa_model = vqa_model
        self.model = model
        self.temp_gpt = temp_gpt
        self.vqa_prompt = vqa_prompt
        self.n_rounds = n_rounds

    def prepare_init_asker_user_message(self, prompt, caption):
        input_prompt = 'Initial caption for the image: {}\n'.format(caption)
        
        input_prompt = prompt.replace('[placeholder]', input_prompt)
        
        messages = {'role': 'user', 'content': input_prompt}
        
        return messages

    def prepare_more_asker_user_message(self, prompt, caption, sub_questions, sub_answers):
        sub_answer_prompt = ''
        flat_sub_questions = []
        for sub_questions_i in sub_questions:
            flat_sub_questions.extend(sub_questions_i)
        flat_sub_answers = []
        for sub_answers_i in sub_answers:
            flat_sub_answers.extend(sub_answers_i)
        assert len(flat_sub_questions) == len(flat_sub_answers)
        for ans_id, ans_str in enumerate(flat_sub_answers):
            sub_answer_prompt = sub_answer_prompt + 'Sub-question: {} Answer: {}\n'.format(flat_sub_questions[ans_id], ans_str)

        input_prompt = 'Initial imperfect caption for the image: {}\n Sub-questions and answers: \n{}'.format(
                    caption, sub_answer_prompt)
        

        input_prompt = prompt.replace('[placeholder]', input_prompt)
        
        messages = {'role': 'user', 'content': input_prompt}
        
        return messages

    def prepare_extender_message(self, prompt, caption, sub_questions, sub_answers):


        sub_answer_prompt = ''
        flat_sub_questions = []
        for sub_questions_i in sub_questions:
            flat_sub_questions.extend(sub_questions_i)
        flat_sub_answers = []
        for sub_answers_i in sub_answers:
            flat_sub_answers.extend(sub_answers_i)

        assert len(flat_sub_questions) == len(flat_sub_answers)
        for ans_id, ans_str in enumerate(flat_sub_answers):
            sub_answer_prompt = sub_answer_prompt + 'Sub-question: {} Answer: {}\n'.format(flat_sub_questions[ans_id], ans_str)
            
        input_prompt = 'Imperfect initial caption: {}\nSub-questions and answers: \n{}'.format(
                caption, sub_answer_prompt)
       

        input_prompt = prompt.replace('[placeholder]',input_prompt)
        
        messages = {'role': 'user', 'content': input_prompt}
        
        return messages


    def parse_subquestion(self, gpt_response):
        gpt_response = gpt_response + '\n'
        sub_questions = []
        while True:
            result = re.search('Sub-question.{0,3}:(.*)\n', gpt_response)
            if result is None:
                break
            else:
                sub_questions.append(result.group(1).strip())
                gpt_response = gpt_response.split(result.group(1))[1]

        return sub_questions


    def answer_question_with_image(self, image, cur_sub_questions):
        # prepare the context for blip2
        sub_answers = []
        for sub_question_i in cur_sub_questions:
            vqa_prompt = self.vqa_prompt.replace('placeholder', sub_question_i)
            # Feed into VQA model.
            if 'llava' in self.vqa_model.model_type or 'minigpt4' in self.vqa_model.model_type:
                answer = self.vqa_model.ask(self.img, vqa_prompt)
            elif 't5' in self.vqa_model.model_type or 'opt' in self.vqa_model.model_type:
                answer = self.vqa_model.ask(image, vqa_prompt, length_penalty=-1, max_length=10)
            else:
                raise NotImplementedError(f'Not support VQA of {self.vqa_model.model_type}.')

            answer = self.answer_trim(answer)
            sub_answers.append(answer)
        return sub_answers


    def extend_description(self,caption, img):
        sub_questions = []
        sub_answers = []
        for round_i in range(self.n_rounds):
            if round_i == 0:
                # Prepare initial gpt input 

                chat_asker_prompt = [{"role": "system", "content": INIT_ASKER_SYSTEM_PROMPT}]
                
                gpt_input = self.prepare_init_asker_user_message(INIT_ASKER_USER_PROMPT, caption)
                
                chat_asker_prompt.append(gpt_input)
                #print('initial asker input:',chat_asker_prompt)
                # Run GPT and update chat_history.
                gpt_response, n_tokens = call_gpt(chat_asker_prompt, model=self.model, temp_gpt=self.temp_gpt)
                #print('initial asker response:',gpt_response)
            else:
                #let GPT ask additional questions.
                chat_asker_prompt = [{"role": "system", "content": MORE_ASKER_SYSTEM_PROMPT}]
                gpt_input = self.prepare_more_asker_user_message(prompt= MORE_ASKER_USER_PROMPT, caption=caption, 
                                                            sub_questions= sub_questions, sub_answers=sub_answers)
                chat_asker_prompt.append(gpt_input)
                #print('more asker prompt:',chat_asker_prompt)
                # Run GPT.
                gpt_response, n_tokens = call_gpt(chat_asker_prompt, model=self.model, temp_gpt=self.temp_gpt)
                #print('more asker response:',gpt_response)
                
            
            #  Post process GPT response to get sub-questions.
            cur_sub_questions = self.parse_subquestion(gpt_response)
            sub_questions.append(cur_sub_questions)
            # if len(cur_sub_questions) != 0:
            

            # Use VQA model to answer sub-questions.
            cur_sub_answers = self.answer_question_with_image(img,cur_sub_questions)
            sub_answers.append(cur_sub_answers) 

            # Input sub-questions and sub-answers into a reasoner GPT.
        
        chat_asker_prompt = [{"role": "system", "content": EXTENDER_SYSTEM_PROMPT}]

        gpt_input = self.prepare_extender_message(prompt=EXTENDER_USER_PROMPT, caption=caption, sub_questions= sub_questions, sub_answers=sub_answers)
        
        chat_asker_prompt.append(gpt_input)
        #print('extender prompt:',chat_asker_prompt)

        gpt_response, n_tokens = call_gpt(chat_asker_prompt, model=self.model, temp_gpt=self.temp_gpt)
        
        #print('final caption:',gpt_response)
        
        return gpt_response

        