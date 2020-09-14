from contextualized_topic_models.models.ctm import CTM
from contextualized_topic_models.utils.data_preparation import TextHandler
from contextualized_topic_models.utils.data_preparation import bert_embeddings_from_list
from contextualized_topic_models.datasets.dataset import CTMDataset
import pandas as pd
import numpy as np
import argparse
import logging

logging.basicConfig(level=logging.DEBUG)

parser = argparse.ArgumentParser()
parser.add_argument("--input")
parser.add_argument("--dataset")
parser.add_argument("--epochs", default=5, type=int)
args = parser.parse_args()

if args.dataset == "trec":
    df = pd.read_csv(args.input, sep="\t")
    texts = list(df['full_text'].values)
    ids = list(df['tweetId'].values)
    nr_topics = len(df['topicId'].unique())
elif args.dataset == "reuters":
    df = pd.read_json(args.input, orient='records', lines=True)
    texts = list(df['title'].values)
    ids = list(df['identifier'].values)
    nr_topics = len(df['topicId'].unique())
else:
    print('not implemented yet')
    exit()

print('nr of data samples:', len(texts))
print('nr topics:', nr_topics)

handler = TextHandler(texts)
handler.prepare()  # create vocabulary and training data

# load BERT data
training_bert = bert_embeddings_from_list(texts, 'bert-base-nli-mean-tokens')

training_dataset = CTMDataset(handler.bow, training_bert, handler.idx2token)

ctm = CTM(input_size=len(handler.vocab), bert_input_size=768, num_epochs=args.epochs, inference_type="combined",
          n_components=nr_topics, num_data_loader_workers=5)

ctm.fit(training_dataset)  # run the model

distribution = ctm.get_thetas(training_dataset)

best_match_topics = np.argmax(distribution, axis=1)

with open('predictions_CTM_' + args.dataset + '.txt', 'w') as fp:
    for ID, topicId in zip(ids, best_match_topics):
        fp.write(str(ID) + ' ' + str(topicId) + '\n')
