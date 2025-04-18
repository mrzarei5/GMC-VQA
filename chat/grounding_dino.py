from groundingdino.util.inference import load_model, load_image, predict, annotate
import supervision as sv
import torch
import numpy as np
from torchvision.ops import box_convert

class GroundingDino():
    def __init__(self):
        
        self.weights_path ="./GroundingDINO/weights/groundingdino_swint_ogc.pth"
        self.config_path = "./GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        self.box_treshold = 0.35
        self.text_treshold = 0.25
        
        self.model = load_model(self.config_path, self.weights_path)
        
        self.expand_factor = 1.2


    def expand_bbox_cxcywh_within_image(self,bbox_cxcywh, image_width, image_height):
        # Unpack the coordinates
        center_x, center_y, width, height = bbox_cxcywh

        # Calculate the new width and height by multiplying with the expansion factor
        new_width = self.expand_factor * width
        new_height = self.expand_factor * height

        # Check if the expanded bounding box exceeds image boundaries
        xmin = max(0, center_x - new_width / 2)
        ymin = max(0, center_y - new_height / 2)
        xmax = min(image_width, center_x + new_width / 2)
        ymax = min(image_height, center_y + new_height / 2)

        # Return the adjusted bounding box coordinates
        return [xmin, ymin, xmax, ymax]

    def expand_bboxes_cxcywh_within_image(self,bboxes_cxcywh, image_width, image_height):
        expanded_bboxes = []

        for index in range(len(bboxes_cxcywh)):
            bbox_cxcywh = bboxes_cxcywh[index,:]
            expanded_bbox_cxcywh = self.expand_bbox_cxcywh_within_image(bbox_cxcywh, image_width, image_height)
            expanded_bboxes.append(expanded_bbox_cxcywh)

        return np.array(expanded_bboxes)

    def extract_patch_from_bbox(self, image, bbox):
        # Convert bounding box coordinates to integers
        bbox = np.round(bbox).astype(int)
        
        # Extract the patch from the original image
        patch = image[bbox[1]:bbox[3],bbox[0]:bbox[2],:]

        return patch

    


    def ground_image(self, img_path, prompt):

        image_source, image = load_image(img_path)

        boxes, logits, phrases = predict(
            model=self.model,
            image=image,
            caption=prompt,
            box_threshold=self.box_treshold,
            text_threshold=self.text_treshold
        )
        h, w, _ = image_source.shape
        boxes_ = boxes * torch.Tensor([w, h, w, h])

        xyxy = self.expand_bboxes_cxcywh_within_image(boxes_, w, h)

        #xyxy = box_convert(boxes=boxes__, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        patches = []

        for bbox in xyxy:
            patch = self.extract_patch_from_bbox(image_source, bbox)
            patches.append(patch)

        return patches