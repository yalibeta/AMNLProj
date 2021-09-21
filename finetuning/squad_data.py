import pickle
import os
import json


def get_token(id):
	return f"<extra_id_{id}>"


def create_item(p, q, a, sep='<sep>'):
	instance = p + ' ' + sep + ' ' + q + ' ' + get_token(0)
	label = get_token(0) + ' ' + a + ' ' + get_token(1)
	return instance, label


input_path = os.path.join('squad1.1', 'test.json')
output_path = 'test_data.pkl'
samples_n = 1000

data = json.loads(open(input_path, 'r').read())['data']
dataset = []

n = 0
for article in data:
	for paragraph in article['paragraphs']:
		text = paragraph['context']
		for qa in paragraph['qas']:
			dataset.append(create_item(text, qa['question'], qa['answers'][0]['text']))
			n += 1
			print(n)
			if n >= samples_n:
				pickle.dump(dataset, open(output_path, 'wb'))
				exit()
