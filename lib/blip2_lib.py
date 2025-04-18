from lavis.models import load_model_and_preprocess
from PIL import Image

class Blip2Lavis():
    def __init__(self, name="blip2_opt", model_type="pretrain_opt6.7b", device="cuda"):
        self.model_type = model_type
        self.blip2, self.blip2_vis_processors, _ = load_model_and_preprocess(
            name=name, model_type=model_type, is_eval=True, device=device)
        # if 't5xl' in self.model_type:
        #     self.blip2 = self.blip2.float()
        self.device = device

    def ask(self, img_path, question, length_penalty=1.0, max_length=30):
        if isinstance(img_path,str):
            raw_image = Image.open(img_path).convert('RGB')
        else:
            raw_image = img_path
        image = self.blip2_vis_processors["eval"](raw_image).unsqueeze(0).to(self.device)
        if 't5' in self.model_type:
            answer = self.blip2.predict_answers({"image": image, "text_input": question}, length_penalty=length_penalty, max_length=max_length)
        else:
            answer = self.blip2.generate({"image": image, "prompt": question}, length_penalty=length_penalty, max_length=max_length)
        answer = answer[0]
        return answer

    def caption(self, img_path, prompt='a photo of'):
        if isinstance(img_path,str):
            raw_image = Image.open(img_path).convert('RGB')
        else:
            raw_image = img_path
        image = self.blip2_vis_processors["eval"](raw_image).unsqueeze(0).to(self.device)
        # caption = self.blip2.generate({"image": image})
        caption = self.blip2.generate({"image": image, "prompt": prompt})
        caption = caption[0].replace('\n', ' ').strip()  # trim caption
        return caption

    def detect_img(self, img, question = 'Name all the objects in the photo.'):
        raw_image = Image.open(img).convert('RGB')
        image = self.blip2_vis_processors["eval"](raw_image).unsqueeze(0).to(self.device)
        answer = self.blip2.generate({"image": image, "prompt": question})
        
        return answer[0]