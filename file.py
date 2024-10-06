import pandas as pd
import torch
from transformers import DebertaTokenizer, DebertaModel
import numpy as np
import networkx as nx
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import torch.nn.functional as F
import torch.optim as optim
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import re
import string
import demoji
import base64
import io 
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from flask import Flask, request, jsonify, render_template

import random

import networkx as nx
import matplotlib.pyplot as plt

import itertools
from heapq import *

from scipy.sparse.linalg import eigsh
from heapq import *
import time
from joblib import Parallel, delayed

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Models
class TextClassificationModel(nn.Module):
    def __init__(self, deb_dim=768, n2v_dim=64, hidden_size=256, num_classes=4, num_layers=2, dropout=0.2):
        super(TextClassificationModel, self).__init__()
        self.do = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()

        self.lstm = nn.LSTM(input_size=deb_dim,
                            hidden_size=hidden_size,
                            bidirectional=True,
                            batch_first=True,
                            num_layers=num_layers,
                            dropout=dropout)

        self.c1 = torch.nn.Conv1d(n2v_dim, 128, 5)
        self.m1 = torch.nn.MaxPool1d(3)

        self.c2 = torch.nn.Conv1d(128, 64, 3)
        self.m2 = torch.nn.MaxPool1d(3)

        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, deberta_embeddings, node2vec_embeddings):

        # LSTM processing
        lstm_out, _ = self.lstm(self.do(deberta_embeddings))
        # Convolutional processing
        c1 = self.c1(lstm_out)
        c1 = self.relu(c1)
        c1 = self.m1(c1)

        c2 = self.c2(c1)
        c2 = self.relu(c2)
        c2 = self.m2(c2)

        pooled_out, _ = torch.max(c2, dim=2)
        fc1 = self.fc1(pooled_out)
        fc1 = self.relu(fc1)
        output = self.fc2(fc1)

        return output
    
class TextClassificationModel2(nn.Module):
    def __init__(self, deb_dim=768, n2v_dim=64, hidden_size=128, num_classes=4, num_layers=2, dropout=0.2):
        super(TextClassificationModel2, self).__init__()
        self.do = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()

        self.gru = nn.GRU(input_size=768,
                            hidden_size=hidden_size,
                            bidirectional=True,
                            batch_first=True,
                            num_layers=num_layers,
                            dropout=dropout)

        self.c1 = torch.nn.Conv1d(64, 128, 3)
        self.m1 = torch.nn.MaxPool1d(3)

        self.c2 = torch.nn.Conv1d(128, 64, 3)
        self.m2 = torch.nn.MaxPool1d(3)

        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, deberta_embeddings, node2vec_embeddings):

        gru_out, _ = self.gru(self.do(deberta_embeddings))

        c1 = self.c1(gru_out)
        c1 = self.relu(c1)
        c1 = self.m1(c1)

        c2 = self.c2(c1)
        c2 = self.relu(c2)

        pooled_out, _ = torch.max(c2, dim=2)
        fc1 = self.fc1(pooled_out)
        fc1 = self.relu(fc1)
        output = self.fc2(fc1)


        return output

class TextClassificationModel3(nn.Module):
    def __init__(self, deb_dim=768, n2v_dim=64, hidden_size=128, num_classes=4, num_layers=2, dropout=0.2):
        super(TextClassificationModel3, self).__init__()
        self.do = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()

        self.lstm = nn.LSTM(input_size=832,
                            hidden_size=hidden_size,
                            bidirectional=True,
                            batch_first=True,
                            num_layers=num_layers,
                            dropout=dropout)

        self.c1 = torch.nn.Conv1d(64, 48, 3)
        self.m1 = torch.nn.MaxPool1d(3)

        self.c2 = torch.nn.Conv1d(48, 32, 3)
        self.m2 = torch.nn.MaxPool1d(3)

        self.fc1 = nn.Linear(32, 16)
        self.fc2 = nn.Linear(16, 4)

    def forward(self, deberta_embeddings, node2vec_embeddings):
        deberta_embeddings = self.do(deberta_embeddings)

        combined_embeddings = torch.cat([deberta_embeddings,
                                         node2vec_embeddings.unsqueeze(1).expand(-1, 64, -1)],
                                        dim=2)
        lstm_out, _ = self.lstm(self.do(combined_embeddings))

        c1 = self.c1(lstm_out)
        c1 = self.relu(c1)
        c1 = self.m1(c1)

        c2 = self.c2(c1)
        c2 = self.relu(c2)

        pooled_out, _ = torch.max(c2, dim=2)
        fc1 = self.fc1(pooled_out)
        fc1 = self.relu(fc1)
        output = self.fc2(fc1)


        return output

class TextClassificationModel4(nn.Module):
    def __init__(self, deb_dim=768, n2v_dim=64, hidden_size=128, num_classes=4, num_layers=2, dropout=0.2):
        super(TextClassificationModel4, self).__init__()
        self.do = nn.Dropout(p=0.1)
        self.relu = nn.ReLU()

        self.lstm = nn.GRU(input_size=832,
                            hidden_size=hidden_size,
                            bidirectional=True,
                            batch_first=True,
                            num_layers=num_layers,
                            dropout=dropout)

        self.c1 = torch.nn.Conv1d(64, 256, 3)
        self.m1 = torch.nn.MaxPool1d(3)

        self.c2 = torch.nn.Conv1d(256, 128, 3)
        self.m2 = torch.nn.MaxPool1d(3)

        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 4)

    def forward(self, deberta_embeddings, node2vec_embeddings):

        deberta_embeddings = self.do(deberta_embeddings)

        combined_embeddings = torch.cat([deberta_embeddings,
                                         node2vec_embeddings.unsqueeze(1).expand(-1, 64, -1)],
                                        dim=2)
        lstm_out, _ = self.lstm(self.do(combined_embeddings))

        c1 = self.c1(lstm_out)
        c1 = self.relu(c1)
        c1 = self.m1(c1)

        c2 = self.c2(c1)
        c2 = self.relu(c2)

        pooled_out, _ = torch.max(c2, dim=2)
        fc1 = self.fc1(pooled_out)
        fc1 = self.relu(fc1)
        output = self.fc2(fc1)


        return output


class TextClassificationModel5(nn.Module):
    def __init__(self, deb_dim=768, n2v_dim=64, hidden_size=256, num_classes=4, num_layers=2, dropout=0.2):
        super(TextClassificationModel5, self).__init__()
        self.do = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()

        # Convolutional Layers
        self.c1 = nn.Conv1d(in_channels=deb_dim, out_channels=128, kernel_size=5)
        self.m1 = nn.MaxPool1d(kernel_size=3)

        self.c2 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3)
        self.m2 = nn.MaxPool1d(kernel_size=3)

        # BiLSTM Layer
        self.lstm = nn.LSTM(input_size=64,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            bidirectional=True,
                            batch_first=True,
                            dropout=dropout)

        # Fully Connected Layers
        self.fc1 = nn.Linear(in_features=hidden_size*2, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=num_classes)

    def forward(self, deberta_embeddings, node2vec_embeddings):
        # Convolutional processing
        deberta_embeddings = deberta_embeddings.permute(0, 2, 1)

        c1 = self.c1(deberta_embeddings)
        c1 = self.relu(c1)
        c1 = self.m1(c1)

        c2 = self.c2(c1)
        c2 = self.relu(c2)
        c2 = self.m2(c2)

        # Prepare for LSTM (batch_size, sequence_length, channels)
        c2 = c2.permute(0, 2, 1)

        # LSTM processing
        lstm_out, _ = self.lstm(self.do(c2))

        # Pooling and Flattening
        pooled_out, _ = torch.max(lstm_out, dim=1)

        # Fully connected layers
        fc1 = self.fc1(pooled_out)
        fc1 = self.relu(fc1)
        output = self.fc2(fc1)

        return output


class TextClassificationModel6(nn.Module):
    def __init__(self, deb_dim=768, n2v_dim=64, hidden_size=128, num_classes=4, num_layers=2, dropout=0.2):
        super(TextClassificationModel6, self).__init__()
        self.do = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()

        # Convolutional Layers
        self.c1 = nn.Conv1d(in_channels=deb_dim, out_channels=64, kernel_size=3)
        self.m1 = nn.MaxPool1d(kernel_size=3)

        self.c2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3)
        self.m2 = nn.MaxPool1d(kernel_size=3)

        # BiGRU Layer
        self.gru = nn.GRU(input_size=32,
                          hidden_size=hidden_size,
                          bidirectional=True,
                          batch_first=True,
                          num_layers=num_layers,
                          dropout=dropout)

        # Fully Connected Layers
        self.fc1 = nn.Linear(in_features=hidden_size*2, out_features=16)
        self.fc2 = nn.Linear(in_features=16, out_features=num_classes)

    def forward(self, deberta_embeddings, node2vec_embeddings):
        # Convolutional processing
        deberta_embeddings = deberta_embeddings.permute(0, 2, 1)

        c1 = self.c1(deberta_embeddings)
        c1 = self.relu(c1)
        c1 = self.m1(c1)

        c2 = self.c2(c1)
        c2 = self.relu(c2)
        c2 = self.m2(c2)

        # Prepare for GRU (batch_size, sequence_length, channels)
        c2 = c2.permute(0, 2, 1)

        # BiGRU processing
        gru_out, _ = self.gru(self.do(c2))

        # Pooling and Flattening
        pooled_out, _ = torch.max(gru_out, dim=1)

        # Fully connected layers
        fc1 = self.fc1(pooled_out)
        fc1 = self.relu(fc1)
        output = self.fc2(fc1)

        return output

class TextClassificationModel7(nn.Module):
    def __init__(self, deb_dim=768, n2v_dim=64, hidden_size=128, num_classes=4, num_layers=2, dropout=0.2):
        super(TextClassificationModel7, self).__init__()
        self.do = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()

        self.cnn1 = nn.Conv1d(in_channels=832, out_channels=128, kernel_size=3, padding=1)
        self.cnn2 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)

        self.lstm = nn.LSTM(input_size=64,
                            hidden_size=hidden_size,
                            bidirectional=True,
                            batch_first=True,
                            num_layers=num_layers,
                            dropout=dropout)

        self.fc1 = nn.Linear(hidden_size * 2, 16)
        self.fc2 = nn.Linear(16, num_classes)

    def forward(self, deberta_embeddings, node2vec_embeddings):
        deberta_embeddings = self.do(deberta_embeddings)

        combined_embeddings = torch.cat([deberta_embeddings,
                                         node2vec_embeddings.unsqueeze(1).expand(-1, 64, -1)],
                                        dim=2)
        # CNN layers
        cnn_out = self.cnn1(combined_embeddings.permute(0, 2, 1))
        cnn_out = self.relu(cnn_out)
        cnn_out = self.maxpool(cnn_out)

        cnn_out = self.cnn2(cnn_out)
        cnn_out = self.relu(cnn_out)
        cnn_out = self.maxpool(cnn_out)

        # Permute back to (batch, seq_len, feature)
        cnn_out = cnn_out.permute(0, 2, 1)

        # LSTM layers
        lstm_out, _ = self.lstm(self.do(cnn_out))

        # Max pooling over the sequence length
        pooled_out, _ = torch.max(lstm_out, dim=1)
        fc1 = self.fc1(pooled_out)
        fc1 = self.relu(fc1)
        output = self.fc2(fc1)


        return output


class TextClassificationModel8(nn.Module):
    def __init__(self, deb_dim=768, n2v_dim=64, hidden_size=128, num_classes=4, num_layers=2, dropout=0.2):
        super(TextClassificationModel8, self).__init__()
        self.do = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()
        self.cnn1 = nn.Conv1d(in_channels=832, out_channels=256, kernel_size=3, padding=1)
        self.cnn2 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.gru = nn.GRU(input_size=128, hidden_size=hidden_size, bidirectional=True, batch_first=True, num_layers=num_layers, dropout=dropout)
        self.fc1 = nn.Linear(hidden_size * 2, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, deberta_embeddings, node2vec_embeddings):
        deberta_embeddings = self.do(deberta_embeddings)
        combined_embeddings = torch.cat([deberta_embeddings, node2vec_embeddings.unsqueeze(1).expand(-1, 64, -1)], dim=2)
        cnn_out = self.cnn1(combined_embeddings.permute(0, 2, 1))
        cnn_out = self.relu(cnn_out)
        cnn_out = self.maxpool(cnn_out)
        cnn_out = self.cnn2(cnn_out)
        cnn_out = self.relu(cnn_out)
        cnn_out = self.maxpool(cnn_out)
        cnn_out = cnn_out.permute(0, 2, 1)
        gru_out, _ = self.gru(self.do(cnn_out))
        pooled_out, _ = torch.max(gru_out, dim=1)
        fc1 = self.fc1(pooled_out)
        fc1 = self.relu(fc1)
        output = self.fc2(fc1)
        return output



# Text preprocessing
class SingleTextDataset(Dataset):
    def __init__(self, text, tokenizer, max_length=64):
        self.text = text
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return 1  # Only one text input

    def __getitem__(self, idx):
        encoded_text = self.tokenizer(
            self.text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {key: val.squeeze() for key, val in encoded_text.items()}
# Load graph and embeddings
graph_path = './Dataset/twitter15_graph.graphml.xml'
embeddings_path = './Embeddings/embeddings.pickle'

G = nx.read_graphml(graph_path)
with open(embeddings_path, 'rb') as f:
    embeddings = pickle.load(f)

# Load DeBERTa tokenizer and model
tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-base')
deberta_model = DebertaModel.from_pretrained('microsoft/deberta-base')

# Define text cleaning functions
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def remove_punctuation(text):
    cleaned_text = re.sub(r'[^A-Za-z0-9\s]+', '', text)
    return cleaned_text, 'Removed punctuation'

def remove_url_word(text):
    cleaned_text = text.replace("URL", "")
    return cleaned_text, 'Removed URLs'

def remove_emojis(text):
    cleaned_text = demoji.replace(text, '')
    return cleaned_text, 'Removed emojis'

def remove_stop_words(text):
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    cleaned_text = ' '.join(filtered_words)
    return cleaned_text, 'Removed stop words'

def lemmatize_text(text):
    words = text.split()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    cleaned_text = ' '.join(lemmatized_words)
    return cleaned_text, 'Lemmatized text'

def clean_text(text):
    steps = []
    text, step = remove_url_word(text)
    steps.append((text, step))
    text, step = remove_punctuation(text)
    steps.append((text, step))
    text, step = remove_emojis(text)
    steps.append((text, step))
    text, step = remove_stop_words(text)
    steps.append((text, step))
    text, step = lemmatize_text(text)
    steps.append((text, step))
    return text, steps

def get_node2vec_embeddings(source_tweet_id, embeddings):
    return embeddings.get(str(source_tweet_id), np.zeros(64))

models = []
# Initialize and load the models
model = TextClassificationModel8(hidden_size=512, num_layers=4, dropout=0.2)
model_load_path = './Pretrained_Models/model8.pth'  # Update with your local path
model.load_state_dict(torch.load(model_load_path, map_location=torch.device('cpu')))  # Load on CPU
model.eval()
models.append(model)

model1 = TextClassificationModel(hidden_size=256, num_layers=3, dropout=0.2)
model1_load_path = './Pretrained_Models/model1.pth'  # Update with your local path
model1.load_state_dict(torch.load(model1_load_path, map_location=torch.device('cpu')))  # Load on CPU
model1.eval()
models.append(model1)

model2 = TextClassificationModel2(hidden_size=512, num_layers=3, dropout=0.1)
model2_load_path = './Pretrained_Models/model2.pth'  # Update with your local path
model2.load_state_dict(torch.load(model2_load_path, map_location=torch.device('cpu')))  # Load on CPU
model2.eval()
models.append(model2)

model3 = TextClassificationModel3(hidden_size=256, num_layers=4, dropout=0.2)
model3_load_path = './Pretrained_Models/model3.pth'  # Update with your local path
model3.load_state_dict(torch.load(model3_load_path, map_location=torch.device('cpu')))  # Load on CPU
model3.eval()
models.append(model3)

model4 = TextClassificationModel4(hidden_size=512, num_layers=3, dropout=0.1)
model4_load_path = './Pretrained_Models/model4.pth'  # Update with your local path
model4.load_state_dict(torch.load(model4_load_path, map_location=torch.device('cpu')))  # Load on CPU
model4.eval()
models.append(model4)

model5 = TextClassificationModel5(hidden_size=256, num_layers=4, dropout=0.1)
model5_load_path = './Pretrained_Models/model5.pth'  # Update with your local path
model5.load_state_dict(torch.load(model5_load_path, map_location=torch.device('cpu')))  # Load on CPU
model5.eval()
models.append(model5)

model6 = TextClassificationModel6(hidden_size=256, num_layers=4, dropout=0.2)
model6_load_path = './Pretrained_Models/model6.pth'  # Update with your local path
model6.load_state_dict(torch.load(model6_load_path, map_location=torch.device('cpu')))  # Load on CPU
model6.eval()
models.append(model6)

model7 = TextClassificationModel7(hidden_size=512, num_layers=4, dropout=0.2)
model7_load_path = './Pretrained_Models/model7.pth'  # Update with your local path
model7.load_state_dict(torch.load(model7_load_path, map_location=torch.device('cpu')))  # Load on CPU
model7.eval()
models.append(model7)

class_labels = {0: "false", 1: "non-rumor", 2: "true", 3: "unverified"}

# Set up Flask application
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    custom_text = request.form['text']
    cleaned_text, cleaning_steps = clean_text(custom_text)
    dataset = SingleTextDataset(cleaned_text, tokenizer)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    model_predictions = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for model_num, model in enumerate(models, start=1):
        model.to(device)
        for batch in dataloader:
            batch = {key: val.to(device) for key, val in batch.items()}
            with torch.no_grad():
                outputs = deberta_model(**batch)
                deberta_embeddings = outputs.last_hidden_state.squeeze(0).detach().cpu().numpy()

        source_tweet_id = "custom_id"  # Update as needed
        node2vec_embeddings = get_node2vec_embeddings(source_tweet_id, embeddings)
        deberta_embeddings = torch.tensor(deberta_embeddings, dtype=torch.float).unsqueeze(0)
        node2vec_embeddings = torch.tensor(node2vec_embeddings, dtype=torch.float).unsqueeze(0)
        with torch.no_grad():
            output = model(deberta_embeddings, node2vec_embeddings)
        _, predicted_class = torch.max(output, dim=1)
        predicted_label = class_labels[predicted_class.item()]
        model_predictions[f'Model{model_num}'] = predicted_label

    print(model_predictions)
    return jsonify({'predictions': model_predictions, 'cleaning_steps': cleaning_steps})


@app.route('/network-immunization')
def network_immunization():
    return render_template('network_immunization.html')

@app.route('/run-immunization', methods=['POST'])
def run_immunization():
    # Graph setup and immunization process
    G = nx.DiGraph()
    graph_path = './Dataset/twitter15_graph.graphml.xml'
    G = nx.read_graphml(graph_path)

    harmful_nodes = [node for node in G.nodes if 'harmful' in G.nodes[node]]

    sampled_harmful_nodes = random.sample(harmful_nodes, int(0.05 * len(harmful_nodes)))

    sample_size = int(0.05 * len(G.nodes))
    sampled_nodes = random.sample(list(G.nodes()), sample_size)

    for node in sampled_harmful_nodes:
        if node not in sampled_nodes:
            sampled_nodes.append(node)

    sampled_subgraph = G.subgraph(sampled_nodes).copy()
    sampled_subgraph = sampled_subgraph.to_undirected()

    class PriorityQueue:
        def __init__(self, initlist=[]):
            self.counter = itertools.count()
            self.entry_finder = {}
            self.pq = []
            for el in initlist:
                entry = [-el[0], next(self.counter), el[1]]
                self.pq.append(entry)
                self.entry_finder[el[1]] = entry
            heapify(self.pq)
            self.REMOVED = '<removed-task>'

        def update_task_add(self, task, add_value):
            priority = 0
            if task in self.entry_finder:
                entry = self.entry_finder.pop(task)
                entry[-1] = self.REMOVED
                priority = entry[0]
            count = next(self.counter)
            entry = [priority-add_value, count, task]
            self.entry_finder[task] = entry
            heappush(self.pq, entry)

        def add_task(self, task, priority=0):
            if task in self.entry_finder:
                self.remove_task(task)
            count = next(self.counter)
            entry = [-priority, count, task]
            self.entry_finder[task] = entry
            heappush(self.pq, entry)

        def remove_task(self, task):
            entry = self.entry_finder.pop(task)
            entry[-1] = self.REMOVED

        def pop_task(self):
            while self.pq:
                priority, count, task = heappop(self.pq)
                if task is not self.REMOVED:
                    del self.entry_finder[task]
                    return task
            raise KeyError('pop from an empty priority queue')

    class Solver:
        def __init__(self, G, k, **params):
            if len(G) == 0:
                raise Exception("Graph can not be empty")
            if k == 0:
                raise Exception("k should be greater than 0")
            self.G = G.copy()
            self.k = int(k)
            self.log = {}
            self.log['created'] = time.time()
            self.params = params
            self.clear()

        def clear(self):
            pass

        def get_name(self):
            return self.__class__.__name__

    class SparseShieldSolver(Solver):
        def sparse_shield(self):
            G = self.G.to_undirected()
            nodelist = list(G.nodes())
            M = len(G)
            indexes = list(range(M))
            inverse_index = {nodelist[i]: i for i in indexes}

            t1 = time.time()
            A = nx.to_scipy_sparse_array(G, nodelist=nodelist, weight=None, dtype='f')
            W, V = eigsh(A, k=1, which='LM')
            max_eig = W[0]
            max_eigvec = V[:, 0].reshape((V.shape[0],))

            self.log["Eigenvalue"] = max_eig

            harmful_nodes = {node for node in G.nodes if G.nodes[node].get('harmful', False)}

            pk = PriorityQueue()
            for i, node in enumerate(nodelist):
                score = 2 * max_eig * (max_eigvec[i] ** 2)
                if node in harmful_nodes:
                    score *= 0.5
                pk.add_task(i, score)

            S = set()
            while len(S) < self.k and len(pk.pq) > 0:
                next_best = pk.pop_task()
                node_index = next_best
                is_harmful = node_index in harmful_nodes
                if not is_harmful:
                    S.add(node_index)
                    for n in G.neighbors(nodelist[node_index]):
                        j = inverse_index[n]
                        if j not in S:
                            pk.add_task(j, -2 * max_eigvec[node_index] * max_eigvec[j])

            t2 = time.time()
            self.log['Total time'] = t2 - t1

            return [nodelist[i] for i in S]

        def run(self):
            blocked = self.sparse_shield()
            self.log['Blocked nodes'] = [str(node) for node in blocked]
            return blocked

    k = int(0.05 * len(sampled_subgraph.nodes))

    solver = SparseShieldSolver(sampled_subgraph, k)
    blocked_nodes = solver.run()

    for node in blocked_nodes:
        sampled_subgraph.nodes[node]['immunized'] = True

    class Simulator:
        def __init__(self, G, seeds):
            self.G = G
            self.seeds = seeds
            self.blocked = {}
            self.log = {}

        def add_blocked(self, name, node_set):
            self.blocked[name] = list(node_set)

        def run(self, iterations, num_threads=22):
            assert(sum([not self.G.has_node(n) for n in self.seeds]) == 0)
            for key in self.blocked:
                blocked_list = self.blocked[key]
                assert(sum([not self.G.has_node(n) for n in blocked_list]) == 0)
            self.log['iterations'] = iterations
            iteration_results = []
            results = Parallel(n_jobs=num_threads)(
                delayed(self.run_iteration)() for i in range(iterations))
            for result in results:
                iteration_results.append(result)
            self.log.update(self.merge_results_across_iterations(iteration_results))
            return self.log

        def run_iteration(self):
            return self.simuation_as_possible_world()

        def simuation_as_possible_world(self):
            t1 = time.time()
            front_nodes = self.seeds
            active_series = []
            active_series.append(len(front_nodes))
            active = set()
            active.update(self.seeds)

            iterations = 0
            active_edges = set()
            active_subgraph = nx.DiGraph()
            active_subgraph.add_nodes_from([key for key in active])

            while len(front_nodes) > 0:
                front_edges = self.get_front_edges(front_nodes)
                active_edges.update(front_edges)
                front_nodes = [e[1] for e in front_edges if e[1] not in active]
                active.update(front_nodes)
                active_series.append(active_series[-1] + len(front_nodes))
                iterations += 1

            active_subgraph.add_edges_from(active_edges)
            results = {}
            results['iterations until termination in unblocked graph'] = iterations
            results['active nodes in unblocked graph'] = len(active_subgraph)
            results['active series in unblocked graph'] = active_series
            results['solvers'] = {}
            for blocked_set_name in self.blocked:
                blocked_list = self.blocked[blocked_set_name]
                results['solvers'][blocked_set_name] = {}
                active_subgraph_with_blocked = active_subgraph.subgraph(
                    [node for node in active_subgraph.nodes() if node not in blocked_list])
                active_subgraph_with_blocked = self.get_reachable_subgraph_from_seeds(
                    active_subgraph_with_blocked)
                activated_node_amount = len(active_subgraph_with_blocked)
                saved_node_amount = results['active nodes in unblocked graph'] - \
                    activated_node_amount
                results['solvers'][blocked_set_name]['activated nodes'] = activated_node_amount
                results['solvers'][blocked_set_name]['saved nodes'] = saved_node_amount
                results['solvers'][blocked_set_name]['fraction of saved nodes to active nodes'] = saved_node_amount / \
                    results['active nodes in unblocked graph']
                results['solvers'][blocked_set_name]['active series in blocked graph'] = self.get_active_series_with_blocked(
                    active_subgraph, blocked_list)
            t2 = time.time()
            results['simulation time'] = t2 - t1
            return results

        def get_reachable_subgraph_from_seeds(self, G):
            G = G.copy()
            G.add_node("superseed")
            G.add_edges_from([("superseed", n) for n in self.seeds])
            node_subset = nx.node_connected_component(G.to_undirected(), "superseed")
            node_subset.remove("superseed")
            return G.subgraph(node_subset)

        def get_active_series_with_blocked(self, active_subgraph, blocked_list):
            if not set(blocked_list).issubset(active_subgraph):
                active_subgraph = active_subgraph.subgraph(
                    [node for node in active_subgraph.nodes() if node not in blocked_list])
            return active_subgraph

        def get_front_edges(self, front_nodes):
            active_front_edges = set()
            for e in self.G.edges(front_nodes):
                # Use a default weight if the edge does not have a weight attribute
                weight = self.G.edges[e[0], e[1]].get('weight', 1.0)
                if random.random() <= weight:
                    active_front_edges.add(e)
            return active_front_edges

        def merge_results_across_iterations(self, results):
            merged = {}
            solvers = set()
            
            # Iterate through each result
            for result in results:
                # Update set of solvers
                solvers.update(result['solvers'].keys())
                
                # Merge non-'solvers' keys
                for key in result:
                    if key != 'solvers':
                        if key not in merged:
                            merged[key] = []
                        merged[key].append(result[key])
            
            # Calculate mean for non-'solvers' keys
            merged = {key: np.mean(merged[key]) for key in merged}
            
            # Initialize 'solvers' dictionary in merged
            merged['solvers'] = {}
            
            # Iterate over each solver
            for solver_key in solvers:
                # Initialize dictionary for each solver key
                merged['solvers'][solver_key] = {}
                
                # Iterate over results again to collect metrics
                for result in results:
                    if solver_key in result['solvers']:
                        for metric in result['solvers'][solver_key]:
                            if metric not in merged['solvers'][solver_key]:
                                merged['solvers'][solver_key][metric] = []
                            merged['solvers'][solver_key][metric].append(result['solvers'][solver_key][metric])
            
            # Calculate mean for metrics in 'solvers' section
            for solver_key in merged['solvers']:
                for metric in merged['solvers'][solver_key]:
                    # Convert values to numeric type if possible
                    values = merged['solvers'][solver_key][metric]
                    numeric_values = [float(v) for v in values if isinstance(v, (int, float))]
                    if numeric_values:
                        merged['solvers'][solver_key][metric] = np.mean(numeric_values)
                    else:
                        merged['solvers'][solver_key][metric] = np.nan  # Handle case where no numeric values
            
            return merged

    def get_top_k_nodes_by_degree(graph, k):
        degree_dict = dict(graph.degree())
        top_k_nodes = sorted(degree_dict, key=degree_dict.get, reverse=True)[:k]
        return top_k_nodes

    seed_size = int(0.01 * len(sampled_subgraph.nodes))
    seeds = get_top_k_nodes_by_degree(sampled_subgraph, seed_size)

    simulator = Simulator(sampled_subgraph, seeds)
    simulator.add_blocked('SparseShield', blocked_nodes)
    results = simulator.run(30)

    for key, value in results.items():
        if key == 'solvers':
            for solver, metrics in value.items():
                for metric, metric_value in metrics.items():
                    print(f"{solver} - {metric}: {metric_value:.4f}")
        else:
            print(f"{key}: {value:.4f}")

    results_str = ""
    for key, value in results.items():
        if key == 'solvers':
            for solver, metrics in value.items():
                for metric, metric_value in metrics.items():
                    results_str += f"{solver} - {metric}: {metric_value:.4f}\n"
        else:
            results_str += f"{key}: {value:.4f}\n"

    pos = nx.spring_layout(sampled_subgraph)
    plt.figure(figsize=(12, 12))
    nx.draw_networkx_nodes(sampled_subgraph, pos, node_color='b', node_size=50)
    nx.draw_networkx_edges(sampled_subgraph, pos, alpha=0.5)
    nx.draw_networkx_nodes(sampled_subgraph, pos, nodelist=blocked_nodes, node_color='r', node_size=50)

    plt.title('Sampled Subgraph with Immunized and Harmful Nodes Highlighted')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()

    return jsonify({'results': results_str, 'graph_image': image_base64})

if __name__ == '__main__':
    app.run(port=5000, debug=True)  # Run Flask app locally on port 5000
