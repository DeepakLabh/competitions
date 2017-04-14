import pandas as pd
import re

class bin_vecs(object):
    def __init__(self, vocab):
        self.vocab = vocab
	self.vocab_size = len(vocab)

    def vec(self, word):
	pass


class create_vocab(object):
    def __init__(self, filename, fields_name, file_type):
	self.filename = filename
	self.fields_name = fields_name
	self.file_type = file_type

    def vocab(self):
	if self.file_type == 'csv':
	    data = pd.read_csv(self.filename)
	    required_data = data[self.fields_name]	
	else: print 'only csv file supported as of now'
		
	return required_data
    
    def flatten(self, list_list):
	
