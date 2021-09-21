from transformers import T5ForConditionalGeneration, T5Tokenizer, AdamW, T5Config
import argparse
import torch
from torch.utils.data import DataLoader
import load_data
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import pickle


def f1(pred_set, label_set):
    tp = len(pred_set.intersection(label_set))
    fp = len(pred_set - label_set)
    fn = len(label_set - pred_set)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 100 * (2 * precision * recall) / (precision + recall)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

my_parser = argparse.ArgumentParser(description='List the content of a folder')
my_parser.add_argument('-model_name', metavar='--model_name', type=str, default="t5-small")
my_parser.add_argument('-lr', metavar='--lr', type=float, default=5e-5)
my_parser.add_argument('-n_epochs', metavar='--n_epochs', type=int, default=10)
my_parser.add_argument('-shots', metavar='--shots', type=int, default=8)
my_parser.add_argument('-eval_size', metavar='--eval_size', type=int, default=50)
my_parser.add_argument('-pre_model', metavar='--pre_model', type=bool, default=False)
my_parser.add_argument('-dir_name', metavar='--dir_name', type=str, default=datetime.now().strftime('%d%m%Y_%H%M_%S'))
my_parser.add_argument('-load_path', metavar='--load_path', type=str, default=None)
my_parser.add_argument('-drop', metavar='--drop', type=float, default=0.1)


args = my_parser.parse_args()
path = os.path.join('runs', args.dir_name)
os.makedirs(path)


print(f"start time: {datetime.now().strftime('%H:%M:%S')}")
print(f'using device {device}')
print(f'using model: {args.model_name}')
print(f"path:{path}")
print(f"lr: {args.lr}")


tokenizer = T5Tokenizer.from_pretrained("t5-small")
tokenizer.sep_token = '<sep>'

if args.pre_model:
    model = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)
else:
    config = T5Config(decoder_start_token_id=tokenizer.convert_tokens_to_ids(['<pad>'])[0], dropout_rate=args.drop)
    model = T5ForConditionalGeneration(config=config).to(device)


optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

if args.load_path:
    checkpoint = torch.load(args.load_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


train_dataset = load_data.MyDataset('train_data.pkl', args.shots)
test_dataset = load_data.MyDataset('test_data.pkl', args.eval_size)


train_loader = DataLoader(train_dataset, batch_size=1)
test_loader = DataLoader(test_dataset, batch_size=1)

f1_train = []
f1_test = []

for epoch in range(args.n_epochs):
    print(epoch)
    running_loss = 0
    model.train()
    for i, batch in enumerate(tqdm(train_loader)):
        for (paragraph, label) in zip(*batch):
            paragraph = tokenizer(paragraph, return_tensors='pt').input_ids.to(device)
            label = tokenizer(label, return_tensors='pt').input_ids.to(device)

            loss = model(input_ids=paragraph, labels=label).loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    f1_train.append(running_loss/len(train_loader))

    model.eval()
    running_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader)):
            for (paragraph, label) in zip(*batch):
                paragraph = tokenizer(paragraph, return_tensors='pt').input_ids.to(device)
                label = tokenizer(label, return_tensors='pt').input_ids.to(device)
                pred = model.generate(paragraph)
                label_tokens = set(int(x) for x in label[0])
                pred_tokens = set(int(x) for x in pred[0])
                running_loss += f1(pred_tokens, label_tokens)
    running_loss = running_loss / len(test_loader)
    print(running_loss)
    f1_test.append(running_loss)

pickle.dump(f1_test, open(os.path.join(path, 'f1.pkl'), 'wb'))
plt.plot(list(range(len(f1_test))), f1_test)
plt.savefig(os.path.join(path, 'graph.png'))

# eval



