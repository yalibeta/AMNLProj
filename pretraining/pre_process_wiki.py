import pickle
from tqdm import tqdm
import random
from nltk.corpus import stopwords
import string
import os

files = 80
wiki_dir = '/home/yandex/AMNLP2021/sehaik/wiki_split'
max_size = 256
max_match = 6
stopwords_set = set(stopwords.words("english"))


def get_token(id):
    return f"<extra_id_{id}>"


def count_sub(a, s):
    bound = 0-len(s)
    n=0
    while True:
        bound = find_sub(a, s, bound+len(s))
        if bound==-1:
            return n
        n+=1

def find_sub(a, s, i=0):
    n = len(s)
    for j in range(i, len(a)):
        if s==a[j:j+len(s)]:
            return j
    return -1


def find_ngrams(p):
    label = ""
    ngrams = []
    for i in range(max_size):
        j=i
        while j<min(max_size, i+max_match):
            if p[j].lower() in stopwords_set:
                break
            if count_sub(p, p[i:j+1])<2:
                break
            j+=1
        if j==i:
            continue
        ngrams.append(p[i:j])
    ngrams.sort(key=lambda x: 0-len(x))
    id = 0
    for ngram in ngrams:
        k = count_sub(p, ngram)
        if k>=2:
            i = random.randrange(k)
            bound = 0-len(ngram)
            for j in range(k):
                bound = find_sub(p, ngram, bound+len(ngram))
                if j==i:
                    continue
                p = p[:bound] + [get_token(id)] + p[bound+len(ngram):] ## error
                label += get_token(id) + " " + " ".join(ngram) + " "
                id +=1
    if id>0:
        label += get_token(id)
        text = " ".join(p)
        return text, label
    return None
                
                
    
        
                


if __name__=="__main__":
    dataset = []

    for index in range(files):
        print(f"working on file_{index}")
        filename = os.path.join(wiki_dir, f"file_{index}")
        file = open(filename, encoding="utf8")
        wiki_data = file.read()
        wiki_data = wiki_data.split("<doc ")[1:-1]
        for i, article in enumerate(tqdm(wiki_data)):
            lines = article.split("\n")[2:]
            lines = [line for line in lines if ((not line.startswith("id=") and len(line)>0 and line!="</doc>"))]
            article = "\n".join(lines).translate(str.maketrans('', '', string.punctuation)).split(" ")
            n = len(article)
            while n>=max_size:
                p = article[len(article) - n: len(article) - n + max_size]
                sample = find_ngrams(p)
                if not sample is None:
                    dataset.append(find_ngrams(p))
                n -= max_size

        file.close()

    print(len(dataset))
    pickle.dump(dataset, open("all_data.pkl", 'wb'))

