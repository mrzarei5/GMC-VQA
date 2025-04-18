import argparse
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
import torch
import os
import json
from tqdm import tqdm



from PIL import Image
import random
import math





class INSTRUCT_BLIP():
    def __init__(self, model_path="/home/student/Experiments/IdealGPT/models/instructblip-flan-t5-xl", device="cuda", temperature=0.001):

        self.model = InstructBlipForConditionalGeneration.from_pretrained(model_path).to(device)
        self.processor = InstructBlipProcessor.from_pretrained(model_path)

        self.temperature = temperature
        self.device = device
        self.model_type = 'instruct_blip'


    def ask(self, img_path, text, length_penalty=1.0, max_length=30):
        
        if isinstance(img_path,str):
            image = Image.open(img_path).convert('RGB')
        else:
            image = img_path
        inputs = self.processor(images=image, text=text, return_tensors="pt").to(self.device)

        outputs = self.model.generate(
        **inputs,
        do_sample=False,
        num_beams=1,
        max_length=max_length,
        min_length=1,
        length_penalty=length_penalty,
        temperature=self.temperature,
        )


        outputs = self.processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        return outputs

    def caption(self, img_path, prompt = "Describe the photo."):
        return self.ask(img_path=img_path, text=prompt)
    
    def detect_img(self, img, question = 'Name all the objects in the photo.'):
        return self.ask(img_path=img, text=question)

    
if __name__ == "__main__":
    instruct_blip = INSTRUCT_BLIP()
    print('successfully initialized instruct blip model')