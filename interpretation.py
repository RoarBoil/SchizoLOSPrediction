# -*- coding;utf-8 -*-
"""
File Create By: Yihang Bao
email: baoyihang@sjtu.edu.cn
"""

import torch
import shap
from torch.utils.data import DataLoader, TensorDataset
from model import NeuralNetwork
import pandas as pd
from matplotlib import pyplot as plt


X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')
y_train = pd.read_csv('y_train.csv')
y_test = pd.read_csv('y_test.csv')
device = torch.device("cuda")
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long).to(device)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long).to(device)
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=32)

input_size = X_train.shape[1]
model = NeuralNetwork(input_size)
model.load_state_dict(torch.load('./weights/81_901_94.pth'))
model.to(device)
model.eval()
test_loader_1 = DataLoader(test_dataset, batch_size=100)
data, _ = next(iter(test_loader_1))

feature_names = X_train.columns

explainer = shap.GradientExplainer(model, data)
shap_values = explainer.shap_values(data)
shap_need = explainer(data)

# general feature importance (bar plot)
plt.figure(figsize=(12, 20), dpi=500)
shap.plots.bar(shap_need[:,:,1], max_display=20)
plt.show()

# general feature importance (bee swarm plot)
shap.plots.beeswarm(shap_need[:,:,1], max_display=20,color_bar=True)

# individual feature importance
shap.plots.bar(explainer(data)[19,:,1], max_display=20, show=False)
fig = plt.gcf()
fig.dpi = 500
plt.show()