from tqdm import tqdm
import re
from copy import deepcopy
from utils import call_gpt
from prompts.okvqa_prompt import *
import json
import os
from PIL import Image
from .extend_description import ExtendDescriptionTwoAgent

from .grounding_dino import GroundingDino

class OKVQAConversationTwoAgent(ExtendDescriptionTwoAgent):
    def __init__(self, img, vqa_model, model, question, label, data_id, prompt_setting='v1a', caption=None, temp_gpt=0, description_extend = False, objects_prompt_type = 'question', temp_name = 'temp'):
        if type(self) == OKVQAConversationTwoAgent:
            assert prompt_setting in ['v1a']
        
        self.temp_name = temp_name

        self.img = img
        self.vqa_model = vqa_model
        self.model = model

        self.question = question
     
        self.answer_predict = None
        self.label = label

        self.caption = caption
        self.sub_questions = []
        self.sub_answers = []
        self.chat_history = []

        self.total_tokens = 0
        self.temp_gpt=temp_gpt

        self.prompt_setting = prompt_setting
        self.use_caption = True
        self.description_extend = description_extend
        
        self.objects_prompt_type = objects_prompt_type

        self.data_id = data_id
        self.chat_history = {}
        self.chat_history_init_asker = []
        self.chat_history_more_asker = []
        self.chat_history_reasoner = []

        if prompt_setting == 'v1a':
            self.INIT_ASKER_SYSTEM_PROMPT = INIT_ASKER_SYSTEM_PROMPT_V1
            self.INIT_ASKER_FIRST_QUESTION = INIT_ASKER_FIRST_QUESTION_V1

            self.MORE_ASKER_SYSTEM_PROMPT = MORE_ASKER_SYSTEM_PROMPT_V1
            self.MORE_ASKER_FIRST_QUESTION = MORE_ASKER_FIRST_QUESTION_V1

            self.REASONER_SYSTEM_PROMPT = REASONER_SYSTEM_PROMPT_V1A
            self.REASONER_FIRST_QUESTION = REASONER_FIRST_QUESTION_V1A
            self.FINAL_REASONER_SYSTEM_PROMPT = FINAL_REASONER_SYSTEM_PROMPT_V1A

        blip2_QA_prompt = 'Question: placeholder Answer:'
        llava_QA_prompt = 'placeholder Reply in short.'

        if 'llava' in self.vqa_model.model_type:
            self.vqa_prompt =  llava_QA_prompt
        elif 't5' in self.vqa_model.model_type or 'opt' in self.vqa_model.model_type:
            self.vqa_prompt =  blip2_QA_prompt
        elif 'instruct_blip' in self.vqa_model.model_type:
            self.vqa_prompt = llava_QA_prompt
        else:
            raise NotImplementedError(f'Could not find vqa prompt for {self.vqa_model.model_type}.')
        super().__init__(vqa_model, model, self.vqa_prompt,  temp_gpt = self.temp_gpt)

    def prepare_init_asker_message(self, prompt, caption, question, answer_choices):
        answer_prompt = ''
        if answer_choices:
            for ans_id, ans_str in enumerate(answer_choices):
                answer_prompt = answer_prompt + 'Answer {}: {}\n'.format(str(ans_id+1), ans_str)


        if self.prompt_setting in ['v1a']:
            if answer_prompt:
                input_prompt = 'Imperfect Caption for the patch: {}\nMain Question: {}\nFour choices:\n{}'.format(caption, question, answer_prompt)
            else:
                input_prompt = 'Imperfect Caption for the patch: {}\nMain Question: {}\n'.format(caption, question)
        else:
            raise NotImplementedError(f'{self.prompt_setting} Not supported')

        input_prompt = prompt.replace('[placeholder]', input_prompt)
        
        messages = {'role': 'user', 'content': input_prompt}
        
        return messages


    def prepare_more_asker_message(self, prompt, caption, question, answer_choices, sub_questions, sub_answers):
        answer_prompt = ''
        if answer_choices:
            for ans_id, ans_str in enumerate(answer_choices):
                answer_prompt = answer_prompt + 'Answer {}: {}\n'.format(str(ans_id+1), ans_str)

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
            
        if self.prompt_setting in ['v1a']:
            if answer_prompt:
                input_prompt = 'Imperfect Caption for this patch: {}\nMain Question: {}\nFour choices: \n{} Sub-questions and answers: \n{}'.format(
                    caption, question, answer_prompt, sub_answer_prompt)
            else:
                input_prompt = 'Imperfect Caption for this patch: {}\nMain Question: {}\n Sub-questions and answers: \n{}'.format(
                    caption, question, sub_answer_prompt)
        else:
            raise NotImplementedError(f'{self.prompt_setting} Not supported')

        input_prompt = prompt.replace('[placeholder]', input_prompt)
        
        messages = {'role': 'user', 'content': input_prompt}
        
        return messages


    def prepare_reasoner_message(self, prompt, captions, question, answer_choices, sub_questions, sub_answers):
        answer_prompt = ''
        if answer_choices:
            for ans_id, ans_str in enumerate(answer_choices):
                answer_prompt = answer_prompt + 'Answer {}: {}\n'.format(str(ans_id+1), ans_str)

        if self.prompt_setting in ['v1a']:
            if answer_prompt:
                input_prompt = 'Main Question: {}\nFour choices: \n{}'.format(
                    question, answer_prompt)
            else:
                input_prompt = 'Main Question: {}'.format(question)
        else:
            raise NotImplementedError(f'{self.prompt_setting} Not supported')

        for caption_patch, sub_questions_patch, sub_answers_patch in zip(captions, sub_questions, sub_answers):
            sub_answer_prompt = ''
            flat_sub_questions = []
            for sub_questions_i in sub_questions_patch:
                flat_sub_questions.extend(sub_questions_i)
            flat_sub_answers = []
            for sub_answers_i in sub_answers_patch:
                flat_sub_answers.extend(sub_answers_i)

            assert len(flat_sub_questions) == len(flat_sub_answers)
            for ans_id, ans_str in enumerate(flat_sub_answers):
                sub_answer_prompt = sub_answer_prompt + 'Sub-question: {} Answer: {}\n'.format(flat_sub_questions[ans_id], ans_str)
            

            input_prompt += '\n\nImperfect Caption for this patch of image: {}\n Existing Sub-questions and answers for this patch of image: \n{}'.format(
                caption_patch, sub_answer_prompt
            )

        input_prompt = prompt.replace('[placeholder]', input_prompt)
        
        messages = {'role': 'user', 'content': input_prompt}
        
        return messages
    

    def answer_question(self, cur_sub_questions):
        # prepare the context for blip2
        sub_answers = []
        for sub_question_i in cur_sub_questions:
            vqa_prompt = self.vqa_prompt.replace('placeholder', sub_question_i)
            # Feed into VQA model.
            if 'llava' in self.vqa_model.model_type in self.vqa_model.model_type:
                answer = self.vqa_model.ask(self.img, vqa_prompt)
            elif 't5' in self.vqa_model.model_type or 'opt' in self.vqa_model.model_type:
                answer = self.vqa_model.ask(self.img, vqa_prompt, length_penalty=-1, max_length=10)
            elif 'instruct_blip' in self.vqa_model.model_type:
                answer = self.vqa_model.ask(self.img, vqa_prompt, length_penalty=-1, max_length=10)
            else:
                raise NotImplementedError(f'Not support VQA of {self.vqa_model.model_type}.')

            answer = self.answer_trim(answer)
            sub_answers.append(answer)
        return sub_answers
    

    def answer_question_with_image(self, image, cur_sub_questions):
        # prepare the context for blip2
        sub_answers = []
        for sub_question_i in cur_sub_questions:
            vqa_prompt = self.vqa_prompt.replace('placeholder', sub_question_i)
            # Feed into VQA model.
            if 'llava' in self.vqa_model.model_type in self.vqa_model.model_type:
                answer = self.vqa_model.ask(self.img, vqa_prompt)
            elif 't5' in self.vqa_model.model_type or 'opt' in self.vqa_model.model_type:
                answer = self.vqa_model.ask(image, vqa_prompt, length_penalty=-1, max_length=10)
            elif 'instruct_blip' in self.vqa_model.model_type:
                answer = self.vqa_model.ask(self.img, vqa_prompt, length_penalty=-1, max_length=10)
            else:
                raise NotImplementedError(f'Not support VQA of {self.vqa_model.model_type}.')

            answer = self.answer_trim(answer)
            sub_answers.append(answer)
        return sub_answers


    def answer_trim(self, answer):
        answer = answer.split('Question:')[0].replace('\n', ' ').strip()
        return answer

    def parse_subquestion(self, gpt_response):
        gpt_response_original = gpt_response
        gpt_response = gpt_response + '\n'
        sub_questions = []
        while True:
            result = re.search('Sub-question.{0,3}:(.*)\n', gpt_response)
            if result is None:
                break
            else:
                sub_questions.append(result.group(1).strip())
                gpt_response = gpt_response.split(result.group(1))[1]
        if self.model == 'gpt-4o-mini':
            if len(sub_questions) == 1 or len(sub_questions) == 0:
                sub_questions = re.findall(r': (.*?)\?', gpt_response_original) 
                sub_questions = [q + '?' for q in sub_questions]
        return sub_questions

    def parse_final_answer(self, gpt_response, final_round_flag):
        # ===== Parse the paragraph starting with analysis. =====
        analysis_result = re.search('Analysis:(.*)\n', gpt_response)
        if analysis_result:
            analysis_string = analysis_result.group(1).strip()
        else:
            print(f'Can not parse analysis from {gpt_response}')
            raise ValueError

        if self.prompt_setting in ['v1a']:
            substring = "More Likely Answer:"

        pattern = f"{re.escape(substring)}(.*?)(?=\n|$)"
        matches = re.findall(pattern, gpt_response)
        if matches:
            answer_string = matches[-1].strip()
            # In middle rounds, detect 'not sure' at first.
            if not final_round_flag:
                if 'not sure' in answer_string.lower():
                    answer_string = None
        else:
            print(f'Can not parse Predicted Answer: from {gpt_response}')
            raise ValueError
        return analysis_string, answer_string
    
    
    def parse_final_answer_rerun(self, gpt_response, final_round_flag):
        try:
            analysis_string, answer_string = self.parse_final_answer(gpt_response=gpt_response, final_round_flag=final_round_flag)
            need_rerun = False
            return analysis_string, answer_string, need_rerun
        except:
            need_rerun = True
            return None, None, need_rerun


    def break_condition(self, gpt_answer):
        if gpt_answer:
            if 'not sure' not in gpt_answer.lower():
                return True
            return False
        else:
            return False

    def revise_objects_list(self,list):
        prompt = '''I have asked a VLM to name all the objects in a photo. You will be given its response. I want you to refine the response by removing nonsense or repetitive names. Return the unique names as a comma separated list.
        Example 1:
        Input: skis, skis, skis, skis, skis, skis, skis, skis, skis, skis,
        Output: skis

        Example 2:
        Input: people, crosswalk, buildings, trees, buildings, bo
        Output: people, crosswalk, buildings, trees
        '''
        user_prompt = 'Now provide the revised list for the following list: {}'.format(list)
        request_prompts = [{"role": "system", "content": prompt},
                           {'role': 'user', 'content': user_prompt}]
        return request_prompts


    def chatting(self, max_n_rounds, print_mode):
        
        image = Image.open(self.img).convert('RGB')
        image.save(os.path.join(self.temp_name,'temp.jpg'))

        #extract patches using grounding model.
        object_detector = GroundingDino()
        if self.objects_prompt_type == 'objects':
            detected_objects = self.vqa_model.detect_img(self.img)
            gpt_input = self.revise_objects_list(detected_objects)
            gpt_response, n_tokens = call_gpt(gpt_input, model=self.model, temp_gpt=self.temp_gpt)
            self.objects = gpt_response
            patches = object_detector.ground_image(self.img,self.objects)

            patches2 = object_detector.ground_image(os.path.join(self.temp_name,'temp.jpg'), self.objects)

            if len(patches2) > len(patches):
                patches = patches2 

        
        elif self.objects_prompt_type == 'question':
            patches = object_detector.ground_image(self.img, self.question)
            patches2 = object_detector.ground_image(os.path.join(self.temp_name,'temp.jpg'), self.question)
            if len(patches2) > len(patches):
                patches = patches2 

        sub_images_data = []
        
        self.caption = self.vqa_model.caption(image, prompt='Describe the image in details.')

        if self.description_extend:
            self.caption = self.extend_description(self.caption,image)
            
        sub_images_data.append({'img':image,'caption':self.caption, 'sub_questions':[], 'sub_answers':[]}) #add original image
        for patch in patches: #add sub-images
            patch = Image.fromarray(patch.astype('uint8'), mode='RGB')
            caption = self.vqa_model.caption(patch, prompt='Describe the image in details.')
            if self.description_extend:
                caption = self.extend_description(caption,patch)
            sub_images_data.append({'img':patch, 'caption':caption, 'sub_questions':[], 'sub_answers':[]})
        
        #print(sub_images_data)

        self.chat_history = {'init_asker':[],
                             'more_asker':[],
                             'reasoner':[]}
        for round_i in tqdm(range(max_n_rounds), desc='Chat Rounds', disable=print_mode != 'bar'):
            for sub_image_data in sub_images_data:
                if round_i == 0:
                    # Prepare initial LLM input for decomposing into sub-questions, and Update chat_history.

                    self.chat_history_init_asker = [{"role": "system", "content": self.INIT_ASKER_SYSTEM_PROMPT}]
                    gpt_input = self.prepare_init_asker_message(prompt=self.INIT_ASKER_FIRST_QUESTION, caption=sub_image_data['caption'], question=self.question, answer_choices=None)
                    self.chat_history_init_asker.append(gpt_input)

                    # Run LLM and update chat_history.
                    gpt_response, n_tokens = call_gpt(self.chat_history_init_asker, model=self.model, temp_gpt=self.temp_gpt)
                    self.chat_history_init_asker.append({'role': 'assistant', 'content': gpt_response})
                    self.total_tokens = self.total_tokens + n_tokens

                    # Save history
                    self.chat_history['init_asker'].append(self.chat_history_init_asker)

                else:
                    # LLM is not sure, let LLM ask additional questions, and update chat_history.
                    self.chat_history_more_asker = [{"role": "system", "content": self.MORE_ASKER_SYSTEM_PROMPT}]
                    gpt_input = self.prepare_more_asker_message(prompt=self.MORE_ASKER_FIRST_QUESTION, caption=self.caption, question=self.question, answer_choices=None, 
                                                                sub_questions=sub_image_data['sub_questions'], sub_answers=sub_image_data['sub_answers'])
                    self.chat_history_more_asker.append(gpt_input)

                    # Run LLM.
                    gpt_response, n_tokens = call_gpt(self.chat_history_more_asker, model=self.model, temp_gpt=self.temp_gpt)
                    self.chat_history_more_asker.append({'role': 'assistant', 'content': gpt_response})
                    self.total_tokens = self.total_tokens + n_tokens

                    # Save history
                    self.chat_history['more_asker'].append(self.chat_history_more_asker)


                #  Post process LLM response to get sub-questions.
                cur_sub_questions = self.parse_subquestion(gpt_response)
                # if len(cur_sub_questions) != 0:
                sub_image_data['sub_questions'].append(cur_sub_questions)

                # Use VQA model to answer sub-questions.
                cur_sub_answers = self.answer_question_with_image(sub_image_data['img'],cur_sub_questions)
                sub_image_data['sub_answers'].append(cur_sub_answers) 

            # Input sub-questions and sub-answers into a reasoner LLM.
            if round_i == max_n_rounds - 1:
                self.chat_history_reasoner = [{"role": "system", "content": self.FINAL_REASONER_SYSTEM_PROMPT}]
            else:
                self.chat_history_reasoner = [{"role": "system", "content": self.REASONER_SYSTEM_PROMPT}]
            sub_questions = []
            sub_answers = []
            captions = []
            for sub_image_data in sub_images_data:
                sub_questions.append(sub_image_data['sub_questions'])
                sub_answers.append(sub_image_data['sub_answers'])
                captions.append(sub_image_data['caption'])
            gpt_input = self.prepare_reasoner_message(prompt=self.REASONER_FIRST_QUESTION, captions=captions, question=self.question, answer_choices=None,
                                                      sub_questions=sub_questions, sub_answers=sub_answers)
            
            self.chat_history_reasoner.append(gpt_input)

            # Run LLM.
            try_num = 0
            max_try = 15
            # Parse predicted answer from LLM output if any.
            if round_i == max_n_rounds - 1:
                final_round_flag = True
            else:
                final_round_flag = False
            while try_num < max_try:
                try_num += 1
                if try_num > max_try/2:
                    cur_temp_gpt = self.temp_gpt + 0.1 * ((2.0*try_num/max_try)-1)
                else:
                    cur_temp_gpt = self.temp_gpt
                gpt_response, n_tokens = call_gpt(self.chat_history_reasoner, model=self.model, temp_gpt=cur_temp_gpt)
                self.total_tokens = self.total_tokens + n_tokens

                cur_analysis, gpt_answer, need_rerun = self.parse_final_answer_rerun(gpt_response, final_round_flag=final_round_flag)
                if not need_rerun:
                    break
                else:
                    if try_num == max_try:
                        raise ValueError('Rerun too many times, still failed in parsing.')
                    else:
                        print(f'Parsing failed, Time {try_num} of Rerun GPT Decision for data {self.data_id}.')

            # Save history
            self.chat_history_reasoner.append({'role': 'assistant', 'content': gpt_response})
            self.chat_history['reasoner'].append(self.chat_history_reasoner)

            self.answer_predict = gpt_answer
            self.analysis = cur_analysis

            # If LLM answer satisfies some condition. Finish current loop.
            if self.break_condition(gpt_answer=gpt_answer):
                break
        self.sub_images_data = sub_images_data
        return round_i+1