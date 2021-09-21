from transformers import T5ForConditionalGeneration, T5Tokenizer, T5Config
import torch
from torch.utils.data import DataLoader
import load_data
from datetime import datetime
import time
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import argparse
import pickle


def save_model(index, m, o, saved_loss, path):
    if args.save_model:
        torch.save({
            'sample': index,
            'model_state_dict': m.state_dict(),
            'optimizer_state_dict': o.state_dict(),
            'loss': saved_loss[-1],
        }, os.path.join(path, f"sample{index}.pt"))


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


my_parser = argparse.ArgumentParser(description='List the content of a folder')
my_parser.add_argument('-data_path', metavar='--data_path', type=str, default="all_data.pkl")
my_parser.add_argument('-model_name', metavar='--model_name', type=str, default="t5-small")
my_parser.add_argument('-lr', metavar='--lr', type=float, default=5e-5)
# my_parser.add_argument('-n_epochs', metavar='--n_epochs', type=int, default=500)
my_parser.add_argument('-data_size', metavar='--data_size', type=float, default=1)
my_parser.add_argument('-eval_size', metavar='--eval_size', type=int, default=500)
my_parser.add_argument('-pre_model', metavar='--pre_model', type=bool, default=False)
my_parser.add_argument('-batch_size', metavar='--batch_size', type=int, default=1)
my_parser.add_argument('-save_model', metavar='--save_model', type=bool, default=False)
my_parser.add_argument('-save_rate', metavar='--save_rate', type=float, default=2.)
my_parser.add_argument('-plot_rate', metavar='--plot_rate', type=float, default=0.02)
my_parser.add_argument('-train_time', metavar='--train_time', type=int, default=60)


args = my_parser.parse_args()
checkpoint_path = os.path.join("checkpoints", datetime.now().strftime('%d%m%Y_%H%M_%S') + f"{'_pre' if args.pre_model else ''}")


# data_path = "all_data.pkl"
# model_name = 't5-small'
# lr = 5e-5
# n_epochs = 10
# data_size = 0.002
# batch_size = 1
# save_model = False
# save_rate = 0.3

print(f"start time: {datetime.now().strftime('%H:%M:%S')}")
print(f'using device {device}')
print(f'using model: {args.model_name}')
print(f"checkpoint path:{checkpoint_path}")
print(f"lr: {args.lr}")
os.makedirs(checkpoint_path)

tokenizer = T5Tokenizer.from_pretrained("t5-small")
print(tokenizer.sep_token)
if args.pre_model:
    model = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)
else:
    config = T5Config(decoder_start_token_id=tokenizer.convert_tokens_to_ids(['<pad>'])[0])
    model = T5ForConditionalGeneration(config=config).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
dataset = load_data.MyDataset(args.data_path, args.data_size)

split_idx = args.eval_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [len(dataset) - split_idx, split_idx])

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

train_loss = []
test_loss = []

model.train()
start = time.time()
batch_index = 0
phase_size = int(len(train_loader) * args.plot_rate)

print(len(train_loader))
print(len(test_loader))
print(phase_size)
print('\n')

plot_count = 0
dumps = 0
for i, batch in enumerate(tqdm(train_loader)):
    if time.time() - start > args.train_time * 60:
        break
    try:
        paragraphs = tokenizer(batch[0], return_tensors='pt', padding=False).input_ids.to(device)
        labels = tokenizer(batch[1], return_tensors='pt', padding=False).input_ids.to(device)

        loss = model(input_ids=paragraphs, labels=labels).loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        paragraphs.to('cpu')
        labels.to('cpu')

    except:
        dumps+=1
        if dumps%1000==0:
            print(f"dumps: {dumps}")

    else:
        batch_index += 1
        if batch_index == phase_size:
            batch_index = 0
            plot_count += 1

            # update test
            model.eval()
            with torch.no_grad():
                running_test_loss = 0
                test_samples = 0
                for j, test_batch in enumerate(test_loader):
                    for (paragraph, label) in zip(*test_batch):
                        try:
                            paragraph = tokenizer(paragraph, return_tensors='pt').input_ids.to(device)
                            label = tokenizer(label, return_tensors='pt').input_ids.to(device)
                            loss = model(input_ids=paragraph, labels=label).loss
                            running_test_loss += loss.item()
                        except:
                            print("memory issue")
                        else:
                            test_samples += 1
                test_loss.append(running_test_loss / test_samples)
            print(f"loss: {test_loss[-1]}")

            the_input = 'The <extra_id_0> walks in <extra_id_1> park'
            a = model.generate(tokenizer(the_input, return_tensors='pt').input_ids.to(device))
            out = tokenizer.convert_ids_to_tokens(a[0].tolist())
            print(out, '\n')

            model.train()
            if plot_count % int(1/args.save_rate) == 0:
                save_model(i, model, optimizer, test_loss, checkpoint_path)

x_axis = [phase_size*x for x in range(1, len(test_loss)+1)]
plt.plot(x_axis, test_loss)
plt.savefig(os.path.join(checkpoint_path, "graph.png"))
save_model('final', model, optimizer, test_loss, checkpoint_path)
pickle.dump({
    'phase_size': phase_size,
    'len': len(test_loss),
    'loss': test_loss
}, open(os.path.join(checkpoint_path, "loss.pkl"), 'wb'))
