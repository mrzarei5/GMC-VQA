from data.stvqaTools.stvqa import STVQA
import random
from tqdm import tqdm
from copy import deepcopy
import random
class STVQASampler():
    
    def __init__(self, dataDir , dataSubType ='train', taskNumber ='3', data_num = 100):
        random.seed(10)
       
        self.data_dir = dataDir
        self.ann_file = '%s/Annotations/%s_task_%s.json'%(dataDir, dataSubType, taskNumber)
       
        self.img_dir = '%s/Images/%s/' %(dataDir,dataSubType)

        self.vqa=STVQA(self.ann_file)

        ids = self.vqa.getQuesIds()
        
        ids_refined = []
        for annot_id in ids: #choose ids for questions with only one answer
            annotation = self.vqa.loadQA(annot_id)[0]
            answers = annotation['answers']
            if len(answers) == 1:
                ids_refined.append(annot_id)

        self._ids = random.sample(ids_refined, data_num)
        self.ann = {}
        for annot_id in self._ids:
            annotation_this = {}
            annotation = self.vqa.loadQA(annot_id)[0]
            image_id = annotation['file_name']
            annotation_this['question'] = annotation['question']
            annotation_this['image_id'] = image_id
            annotation_this['img_path'] = self.img_dir + annotation['file_path']
            annotation_this['answer_label'] = annotation['answers'][0]
            self.ann[annot_id] = annotation_this
    
    @property
    def ids(self):
        return deepcopy(self._ids)

    
    def fetch_data(self, id):
        ann = self.ann[id]
        img_path = ann['img_path']
        # raw_image = Image.open(img_path).convert('RGB')
        
        return img_path, ann

