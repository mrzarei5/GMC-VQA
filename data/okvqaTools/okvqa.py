

import json
import datetime
import copy

class OKVQA:
	def __init__(self, annotation_file=None, question_file=None):
		"""
       	Constructor of VQA helper class for reading and visualizing questions and answers.
        :param annotation_file (str): location of VQA annotation file
        :return:
		"""
        # load dataset
		self.dataset = {}
		self.questions = {}
		self.qa = {}
		self.qqa = {}
		self.imgToQA = {}
		self.answers = {}
		if not annotation_file == None and not question_file == None:
			print('loading VQA annotations and questions into memory...')
			time_t = datetime.datetime.utcnow()
			dataset = json.load(open(annotation_file, 'r'))
			questions = json.load(open(question_file, 'r'))
			print(datetime.datetime.utcnow() - time_t)
			self.dataset = dataset
			self.questions = questions
			self.createIndex()

	def createIndex(self):
        # create index
		print('creating index...')
		imgToQA = {ann['image_id']: [] for ann in self.dataset['annotations']}
		qa =  {ann['question_id']:       [] for ann in self.dataset['annotations']}
		qqa = {ann['question_id']:       [] for ann in self.dataset['annotations']}
		for ann in self.dataset['annotations']:
			imgToQA[ann['image_id']] += [ann]
			qa[ann['question_id']] = ann
		for ques in self.questions['questions']:
			qqa[ques['question_id']] = ques
		print('index created!')

 		# create class members
		self.qa = qa
		self.qqa = qqa
		self.imgToQA = imgToQA

	def info(self):
		"""
		Print information about the VQA annotation file.
		:return:
		"""
		for key, value in self.datset['info'].items():
			print('%s: %s'%(key, value))

	def getQuesIds(self, imgIds=[], quesTypes=[], ansTypes=[]):
		"""
		Get question ids that satisfy given filter conditions. default skips that filter
		:param 	imgIds    (int array)   : get question ids for given imgs
				quesTypes (str array)   : get question ids for given question types
				ansTypes  (str array)   : get question ids for given answer types
		:return:    ids   (int array)   : integer array of question ids
		"""
		imgIds 	  = imgIds    if type(imgIds)    == list else [imgIds]
		quesTypes = quesTypes if type(quesTypes) == list else [quesTypes]
		ansTypes  = ansTypes  if type(ansTypes)  == list else [ansTypes]

		if len(imgIds) == len(quesTypes) == len(ansTypes) == 0:
			anns = self.dataset['annotations']
		else:
			if not len(imgIds) == 0:
				anns = sum([self.imgToQA[imgId] for imgId in imgIds if imgId in self.imgToQA],[])
			else:
				anns = self.dataset['annotations']
			anns = anns if len(quesTypes) == 0 else [ann for ann in anns if ann['question_type'] in quesTypes]
			anns = anns if len(ansTypes)  == 0 else [ann for ann in anns if ann['answer_type'] in ansTypes]
		ids = [ann['question_id'] for ann in anns]
		return ids

	def getImgIds(self, quesIds=[], quesTypes=[], ansTypes=[]):
		"""
		Get image ids that satisfy given filter conditions. default skips that filter
		:param quesIds   (int array)   : get image ids for given question ids
               quesTypes (str array)   : get image ids for given question types
               ansTypes  (str array)   : get image ids for given answer types
		:return: ids     (int array)   : integer array of image ids
		"""
		quesIds   = quesIds   if type(quesIds)   == list else [quesIds]
		quesTypes = quesTypes if type(quesTypes) == list else [quesTypes]
		ansTypes  = ansTypes  if type(ansTypes)  == list else [ansTypes]

		if len(quesIds) == len(quesTypes) == len(ansTypes) == 0:
			anns = self.dataset['annotations']
		else:
			if not len(quesIds) == 0:
				anns = sum([self.qa[quesId] for quesId in quesIds if quesId in self.qa],[])
			else:
				anns = self.dataset['annotations']
			anns = anns if len(quesTypes) == 0 else [ann for ann in anns if ann['question_type'] in quesTypes]
			anns = anns if len(ansTypes)  == 0 else [ann for ann in anns if ann['answer_type'] in ansTypes]
		ids = [ann['image_id'] for ann in anns]
		return ids

	def loadQA(self, ids=[]):
		"""
		Load questions and answers with the specified question ids.
		:param ids (int array)       : integer ids specifying question ids
		:return: qa (object array)   : loaded qa objects
		"""
		if type(ids) == list:
			return [self.qa[id] for id in ids]
		elif type(ids) == int:
			return [self.qa[ids]]
		
	def detAns(self,ann):
		from collections import Counter

		def most_common_element(lst):
			# Count the occurrences of each element in the list
			counts = Counter(lst)
			
			# Find the element with the highest count
			most_common = counts.most_common(1)[0]
			
			return most_common

		answers = ann['answers']
		answers_list = []
		for answer_details in answers:
			answer = answer_details['answer']
			if answer_details['answer_confidence'] == 'yes':
				answers_list.append(answer)
		answer, count = most_common_element(answers_list)
		if count >= 4:
			self.answers[ann['question_id']] = answer
			return answer
		else:
			return None
	
	def getAns(self,q_id):
		return self.answers[q_id]

	def showQA(self, anns):
		"""
		Display the specified annotations.
		:param anns (array of object): annotations to display
		:return: None
		"""
		if len(anns) == 0:
			return 0
		for ann in anns:
			quesId = ann['question_id']
			print("Question: %s" %(self.qqa[quesId]['question']))
			for ans in ann['answers']:
				print("Answer %d: %s" %(ans['answer_id'], ans['answer']))
		
	