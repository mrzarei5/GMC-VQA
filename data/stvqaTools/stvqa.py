

import json
import datetime
import copy

class STVQA:
	def __init__(self, annotation_file=None):
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
		self.answers = {}
		if not annotation_file == None :
			print('loading VQA annotations and questions into memory...')
			time_t = datetime.datetime.utcnow()
			dataset = json.load(open(annotation_file, 'r'))
			print(datetime.datetime.utcnow() - time_t)
			self.dataset = dataset

			self.createIndex()
		
	def createIndex(self):
        # create index
		print('creating index...')
		

		qa =  {ann['question_id']:       [] for ann in self.dataset['data']}
		
		for ann in self.dataset['data']:
			qa[ann['question_id']] = ann
	
		print('index created!')

 		# create class members
		self.qa = qa
		
	def info(self):
		"""
		Print information about the VQA annotation file.
		:return:
		"""
		for key, value in self.datset['info'].items():
			print('%s: %s'%(key, value))

	def getQuesIds(self):
		"""
		Get question ids
		:return:    ids   (int array)   : integer array of question ids
		"""
		anns = self.dataset['data']
		
		ids = [ann['question_id'] for ann in anns]
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


	
		
	