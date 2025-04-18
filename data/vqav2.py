from data.vqaTools.vqa import VQA
import random
from tqdm import tqdm
from copy import deepcopy
import random
class VQAV2Sampler():
    
    def __init__(self, dataDir , dataSubType ='val', versionType ='v2_', taskType ='OpenEnded', dataType ='mscoco', data_num = 100):
        random.seed(10)
        if dataSubType == 'train':
            self.data_sub_type = 'train2014'
        elif dataSubType == 'val':
            self.data_sub_type = 'val2014'
        self.data_dir = dataDir
        self.ann_file = '%s/Annotations/%s%s_%s_annotations.json'%(dataDir, versionType, dataType, self.data_sub_type)
        self.ques_file = '%s/Questions/%s%s_%s_%s_questions.json'%(dataDir, versionType, taskType, dataType, self.data_sub_type)
        self.img_dir = '%s/Images/%s/%s/' %(dataDir, dataType, self.data_sub_type)

        self.vqa=VQA(self.ann_file, self.ques_file)

        ids = self.vqa.getQuesIds()
        self._ids = random.sample(ids, data_num)
        self.ann = {}
        for annot_id in self._ids:
            annotation_this = {}
            annotation = self.vqa.loadQA(annot_id)[0]
            image_id = annotation['image_id']
            annotation_this['question_type'] = annotation['question_type']
            annotation_this['question'] = self.vqa.qqa[annot_id]['question']
            annotation_this['image_id'] = image_id
            annotation_this['img_path'] = self.img_dir + 'COCO_' + self.data_sub_type + '_'+ str(image_id).zfill(12) + '.jpg'
            annotation_this['answer_label'] = annotation['multiple_choice_answer']
            self.ann[annot_id] = annotation_this

    
    @property
    def ids(self):
        return deepcopy(self._ids)

    
    def fetch_data(self, id):
        ann = self.ann[id]
        img_path = ann['img_path']
        # raw_image = Image.open(img_path).convert('RGB')
        
        return img_path, ann

