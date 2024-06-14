# From https://opensource.salesforce.com/OmniXAI/latest/tutorials/nlp_imdb.html

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import sklearn
from sklearn.datasets import fetch_20newsgroups
import json
import os
import pickle
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

from torch.optim import AdamW

from omnixai.data.text import Text
from omnixai.preprocessing.text import Word2Id
from omnixai.explainers.tabular.agnostic.L2X.utils import Trainer, InputData, DataLoader
from omnixai.explainers.nlp import NLPExplainer
from omnixai.visualization.dashboard import Dashboard

from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModelForSequenceClassification, pipeline
from sklearn.feature_extraction.text import CountVectorizer

import sys
import time
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline



def text_explain_script(data_file=None,socket=None,model_name='Roberta'):
    if data_file is not None:
        input_text = data_file
    else:
        input_text = "This is the default value"

    x = Text([input_text])

    # We apply a simple CNN model for this text classification task. 
    # Note that the method forward has two inputs inputs (token ids) and masks (the sentence masks). 
    # Note that the first input of the model must be the token ids.
    if model_name == 'CNN':
        class TextModel(nn.Module):

            def __init__(self, num_embeddings, num_classes, **kwargs):
                super().__init__()
                self.num_embeddings = num_embeddings
                self.embedding_size = kwargs.get("embedding_size", 50)
                self.embedding = nn.Embedding(self.num_embeddings, self.embedding_size)
                self.embedding.weight.data.normal_(mean=0.0, std=0.01)

                hidden_size = kwargs.get("hidden_size", 100)
                kernel_sizes = kwargs.get("kernel_sizes", [3, 4, 5])
                if type(kernel_sizes) == int:
                    kernel_sizes = [kernel_sizes]

                self.activation = nn.ReLU()
                self.conv_layers = nn.ModuleList([
                    nn.Conv1d(self.embedding_size, hidden_size, k, padding=k // 2) for k in kernel_sizes])
                self.dropout = nn.Dropout(0.2)
                self.output_layer = nn.Linear(len(kernel_sizes) * hidden_size, num_classes)

            def forward(self, inputs, masks):
                embeddings = self.embedding(inputs)
                x = embeddings * masks.unsqueeze(dim=-1)
                x = x.permute(0, 2, 1)
                x = [self.activation(layer(x).max(2)[0]) for layer in self.conv_layers]
                outputs = self.output_layer(self.dropout(torch.cat(x, dim=1)))
                if outputs.shape[1] == 1:
                    outputs = outputs.squeeze(dim=1)
                return outputs

        # A Text object is used to represent a batch of texts/sentences. The package omnixai.preprocessing.text provides some transforms related to text data such as Tfidf and Word2Id.
        
        # Load the training and test datasets
        train_data = pd.read_csv('IMDB.csv', sep=',')
        n = int(0.8 * len(train_data))
        x_train = Text(train_data["review"].values[:n])
        y_train = train_data["sentiment"].values[:n]
        y_train = [0 if y == 'negative' else 1 for y in y_train]

        x_test = Text(train_data["review"].values[n:])
        y_test = train_data["sentiment"].values[n:]
        y_test = [0 if y == 'negative' else 1 for y in y_test]
        class_names = ["negative", "positive"]
        # The transform for converting words/tokens to IDs
        if os.path.exists("word2id_transform.pkl"):
            with open("word2id_transform.pkl", "rb") as f:
                transform = pickle.load(f)
        else:
            transform = Word2Id().fit(x_train)
            with open("word2id_transform.pkl", "wb") as f:
                pickle.dump(transform, f)


        #The preprocessing function converts a batch of texts into token IDs and the masks. The outputs of the preprocessing function must fit the inputs of the model.
        max_length = 256
        device = "cuda" if torch.cuda.is_available() else "cpu"

        def preprocess(X: Text):
            samples = transform.transform(X)
            max_len = 0
            for i in range(len(samples)):
                max_len = max(max_len, len(samples[i]))
            max_len = min(max_len, max_length)
            inputs = np.zeros((len(samples), max_len), dtype=int)
            masks = np.zeros((len(samples), max_len), dtype=np.float32)
            for i in range(len(samples)):
                x = samples[i][:max_len]
                inputs[i, :len(x)] = x
                masks[i, :len(x)] = 1
            return inputs, masks

        if os.path.exists("cnn-imdb.pth"):
            model = TextModel(
                    num_embeddings=transform.vocab_size,
                    num_classes=len(class_names)
            ).to(device)
            model.load_state_dict(torch.load("cnn-imdb.pth"))
            
        else:
            #We now train the CNN model and evaluate its performance.
            model = TextModel(
                num_embeddings=transform.vocab_size,
                num_classes=len(class_names)
            ).to(device)

            Trainer(
                optimizer_class=torch.optim.AdamW,
                learning_rate=1e-3,
                batch_size=128,
                num_epochs=10,
            ).train(
                model=model,
                loss_func=nn.CrossEntropyLoss(),
                train_x=transform.transform(x_train),
                train_y=y_train,
                padding=True,
                max_length=max_length,
                verbose=True
            )

            model.eval()
            data = transform.transform(x_test)
            data_loader = DataLoader(
                dataset=InputData(data, [0] * len(data), max_length),
                batch_size=32,
                collate_fn=InputData.collate_func,
                shuffle=False
            )
            outputs = []
            for inputs in data_loader:
                value, mask, target = inputs
                y = model(value.to(device), mask.to(device))
                outputs.append(y.detach().cpu().numpy())
            outputs = np.concatenate(outputs, axis=0)
            predictions = np.argmax(outputs, axis=1)
            print('Test accuracy: {}'.format(
                sklearn.metrics.f1_score(y_test, predictions, average='binary')))
            
            torch.save(model.state_dict(), "cnn-imdb.pth")

        # The preprocessing function
        preprocess_func = lambda x: tuple(torch.tensor(y).to(device) for y in preprocess(x))
        # The postprocessing function
        postprocess_func = lambda logits: torch.nn.functional.softmax(logits, dim=1)
        # Initialize a NLPExplainer
        explainer = NLPExplainer(
            explainers=["ig", "lime", "polyjuice"],
            mode="classification",
            model=model,
            preprocess=preprocess_func,
            postprocess=postprocess_func,
            params={"ig": {"embedding_layer": model.embedding,
                        "id2token": transform.id_to_word},
                    "polyjuice":{"early_stopping" : True}       
                    }
        )

    
    elif model_name == "Roberta":
        # Using a RoBERTa-based model for sentiment analysis
        class_names = ["negative", "positive"]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        base_model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=2).to(device)

        class CustomRobertaModel(nn.Module):
            def __init__(self, base_model):
                super(CustomRobertaModel, self).__init__()
                self.base_model = base_model

            def forward(self, input_ids, attention_mask):
                outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits  # Extract logits from SequenceClassifierOutput
                return logits

        model = CustomRobertaModel(base_model)

        def preprocess_func(text):
            inputs = tokenizer(text.to_str(), return_tensors="pt", padding=True, truncation=True, max_length=256)
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            return input_ids, attention_mask

        def postprocess_func(logits):
            logits = logits.detach()
            return torch.nn.functional.softmax(logits, dim=1)

        def custom_predict(text):
            input_ids, attention_mask = preprocess_func(text)
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            return postprocess_func(logits)

        explainer = NLPExplainer(
            explainers=["ig", "lime", "polyjuice"],
            mode="classification",
            model=model,
            preprocess=preprocess_func,
            postprocess=postprocess_func,
            params={"ig": {"embedding_layer": base_model.roberta.embeddings,
                        "id2token": {i: tok for tok, i in tokenizer.vocab.items()}},
                    "polyjuice": {"early_stopping": True}}
        )

        explainer.predict_function = custom_predict



    else:
        raise ValueError("Modello non supportato!")
  
    # Generates explanations
    local_explanations = explainer.explain(x)



    # Launch a dashboard for visualization
    dashboard = Dashboard(
        instances=x,
        local_explanations=local_explanations,
        class_names=class_names
    )
    
    if socket is not None:
        time.sleep(5)
        socket.emit('dashboard_status', {'running': True})
    dashboard.show()


if __name__ == '__main__':
    text_explain_script()