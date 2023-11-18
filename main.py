
from scipy.io.arff import loadarff
import torch
from sklearn.metrics import confusion_matrix, classification_report
from glob import glob
import time
import copy
import shutil
import seaborn as sns
import numpy as np
from matplotlib import rc
from pylab import rcParams
import pandas as pd
import matplotlib.pyplot as plt

from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from sklearn.model_selection import train_test_split
from model import RecurrentAutoencoder
from test import test_model
from train import train_model

from utils import create_dataset

sns.set(style='whitegrid', palette='muted', font_scale=1.2)

HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]
sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))
rcParams['figure.figsize'] = 12, 8

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



raw_data =  loadarff('ECG5000_TRAIN.arff')
train = pd.DataFrame(raw_data[0])

train['target'] = train['target'].apply(lambda x: int(x))


raw_data = loadarff('ECG5000_TEST.arff')
test = pd.DataFrame(raw_data[0])

test['target'] = test['target'].apply(lambda x: int(x))

df = train.append(test)
df = df.sample(frac=1.0)


CLASS_NORMAL = 1

class_names = ['Normal','R on T','PVC','SP','UB']


normal_df = df[df.target == CLASS_NORMAL].drop(labels='target', axis=1)
anomaly_df = df[df.target != CLASS_NORMAL].drop(labels='target', axis=1)



train_df, val_df = train_test_split(
  normal_df,
  test_size=0.15,
  random_state=RANDOM_SEED
)

val_df, test_df = train_test_split(
  val_df,
  test_size=0.33, 
  random_state=RANDOM_SEED
)
     

train_sequences = train_df.astype(np.float32).to_numpy().tolist()
val_sequences = val_df.astype(np.float32).to_numpy().tolist()
test_sequences = test_df.astype(np.float32).to_numpy().tolist()
anomaly_sequences = anomaly_df.astype(np.float32).to_numpy().tolist()




train_dataset, seq_len, n_features = create_dataset(train_df)
val_dataset, _, _ = create_dataset(val_df)
test_normal_dataset, _, _ = create_dataset(test_df)
test_anomaly_dataset, _, _ = create_dataset(anomaly_df)


model = RecurrentAutoencoder(seq_len, n_features, 128)
model = model.to(device)



if __name__ == '__main__':
  
  model, history = train_model(
  model, 
  train_dataset, 
  val_dataset, 
  n_epochs=150
  )
  
  MODEL_PATH = 'model.pth'

  torch.save(model, MODEL_PATH)

  model = torch.load('model.pth')
  model = model.to(device)

  test_model(model, train_dataset, test_normal_dataset, test_anomaly_dataset)

