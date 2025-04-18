import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig  
import torch
import os
import json
from tqdm import tqdm
import shortuuid

import sys
sys.path.append('/home/student/Experiments/LLaVA')


from llava.conversation import conv_templates
from llava.utils import disable_torch_init
from transformers import CLIPVisionModel, CLIPImageProcessor, StoppingCriteria

from PIL import Image
import random
import math

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model2

from llava.mm_utils import (
    get_model_name_from_path,
)


DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

def patch_config(config):
    patch_dict = {
        "use_mm_proj": True,
        "mm_vision_tower": "openai/clip-vit-large-patch14",
        "mm_hidden_size": 1024
    }

    cfg = AutoConfig.from_pretrained(config)
    if not hasattr(cfg, "mm_vision_tower"):
        print(f'`mm_vision_tower` not found in `{config}`, applying patch and save to disk.')
        for k, v in patch_dict.items():
            setattr(cfg, k, v)
        cfg.save_pretrained(config)


# new stopping implementation
class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.tokenizer = tokenizer
        self.start_len = None
        self.input_ids = input_ids

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if self.start_len is None:
            self.start_len = self.input_ids.shape[1]
        else:
            outputs = self.tokenizer.batch_decode(output_ids[:, self.start_len:], skip_special_tokens=True)[0]
            for keyword in self.keywords:
                if keyword in outputs:
                    return True
        return False

def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)

class LLAVA():
    def __init__(self, model_path="liuhaotian/llava-llama-2-7b-chat-lightning-lora-preview", mm_projector_setting=None,                           
                 vision_tower_setting=None, conv_mode=None, temperature=0.001):
        disable_torch_init()

        model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path=model_path,
            #model_base=None,
            model_base='meta-llama/LLaMA-2-7B-Chat-hf',
            model_name=model_name
        )   
        
        self.model_name = model_name
        
        self.conv_mode = conv_mode

        self.temperature = temperature
        self.model_type = 'llava'


    def ask(self, img_path, text):
        args = type('Args', (), {
            "model_name": self.model_name,
            "query": text,
            "conv_mode": self.conv_mode,
            "image_file": img_path,
            "sep": ",",
            "temperature": self.temperature,
            "top_p": None,
            "num_beams": 1,
            "max_new_tokens": 512,
            "tokenizer": self.tokenizer,
            "model": self.model,
            "image_processor": self.image_processor,
            "context_len": self.context_len
        })()
        outputs = eval_model2(args)
        return outputs

    def caption(self, img_path, prompt = None):
        return self.ask(img_path=img_path, text='Give a clear and concise summary of the image below in one paragraph.')
    
    def detect_img(self, img, question = 'Name all the objects in the photo.'):
        return self.ask(img_path=img, text=question)

    
if __name__ == "__main__":
    llava_model = LLAVA()
    print('successfully initialized llava model')