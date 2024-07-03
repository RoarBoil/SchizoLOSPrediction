# -*- coding;utf-8 -*-
"""
File Create By: Yihang Bao
email: baoyihang@sjtu.edu.cn
"""
import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from model import NeuralNetwork
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.model_selection import ParameterGrid
import numpy as np


def standardize(df, norm_features, _):
    for feat in norm_features[:-1]:
        if feat in ['sex', 'marry']:
            continue
        if feat == 'dis_days':
            df[feat] = df[feat].apply(lambda x: 0 if x < 30 else (0.5 if x < 600 else 1))
        df[f'{feat}_log_normalized'] = (df[feat] - df[feat].mean()) / df[feat].std()
    return df
s

file_path = ''  # Path to the dataset
df_1 = pd.read_csv(file_path, header=0)
df_1.dropna(subset=['label'], inplace=True)

df_2 = pd.DataFrame()
df_2['sex'] = df_1['性别'].apply(lambda x: 1 if x in ['男性'] else 0)
df_2['marry'] = df_1['婚姻状态'].apply(lambda x: 1 if x in ['已婚'] else (0.5 if x in ['离婚', '丧偶'] else 0))
df_2['smoke_1'] = df_1['smoke'].apply(lambda x: 1 if x in ['true'] else 0)
df_2['chongdong'] = df_1['冲动'].apply(lambda x: 1 if x in ['是'] else 0)
df_2['xiaoji'] = df_1['消极'].apply(lambda x: 1 if x in ['是'] else 0)
df_2['waipao'] = df_1['外跑'].apply(lambda x: 1 if x in ['是'] else 0)
df_2['label'] = df_1['label']
df_2['入院时间'] = df_1['入院时间']

df_2.dropna(subset=['label'], inplace=True)
df_3 = standardize(df_2, list(df_2.columns), df_2)  # feature standardization
sorted_df = df_3.sort_values(by='入院时间', ascending=False)

latest_label_1 = sorted_df[sorted_df['label'] == 1].head(50)
latest_label_0 = sorted_df[sorted_df['label'] == 0].head(50)
test_df = pd.concat([latest_label_1, latest_label_0])
train_df = sorted_df.drop(test_df.index)

X_test = test_df.drop(columns=['label', '入院时间']).to_numpy()
y_test = test_df['label'].to_numpy()

X_train = train_df.drop(columns=['label', '入院时间']).to_numpy()
y_train = train_df['label'].to_numpy()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hyperparams = {
    'learning_rate': [],
    'batch_size': [],
    'epochs': [],
}


def train_evaluate_model(X_train, y_train, X_val, y_val, params):
    train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long).to(device)

    train_dataset = TensorDataset(train_tensor, y_train_tensor)
    val_dataset = TensorDataset(val_tensor, y_val_tensor)

    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)

    model = NeuralNetwork(X_train.shape[1])
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])

    for epoch in range(params['epochs']):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = F.cross_entropy(outputs, y_batch)
            loss.backward()
            optimizer.step()

    # Evaluation on val set
    model.eval()
    with torch.no_grad():
        output = model(val_tensor)
        probs = F.softmax(output, dim=1).cpu().numpy()
        auc_score = roc_auc_score(y_val_tensor.cpu().numpy(), probs[:, 1])
        return auc_score


best_auc = 0
best_params = None
for params in ParameterGrid(hyperparams):
    all_aucs = []
    for i in range(len(X_train)):  # Leave-One-Out Cross-Validation
        X_val = X_train[i].reshape(1, -1)
        y_val = y_train[i].reshape(1)
        X_train_cv = np.delete(X_train, i, axis=0)
        y_train_cv = np.delete(y_train, i, axis=0)
        auc_score = train_evaluate_model(X_train_cv, y_train_cv, X_val, y_val, params)
        all_aucs.append(auc_score)
    avg_auc = np.mean(all_aucs)

    if avg_auc > best_auc:
        best_auc = avg_auc
        best_params = params

print(f'Best hyperparameters: {best_params}, Best AUC: {best_auc:.3f}')

best_model = NeuralNetwork(X_train.shape[1])
best_model.to(device)

optimizer = torch.optim.Adam(best_model.parameters(), lr=best_params['learning_rate'])

train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)

train_dataset = TensorDataset(train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)

for epoch in range(best_params.get('epochs', 5)):
    best_model.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = best_model(X_batch)
        loss = F.cross_entropy(outputs, y_batch)
        loss.backward()
        optimizer.step()

X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=best_params['batch_size'], shuffle=False)

best_model.eval()
test_preds = []
test_targets = []
total = 0
correct = 0
with torch.no_grad():
    for X, y in test_loader:
        outputs = best_model(X)
        probs = F.softmax(outputs, dim=1)
        test_preds.extend(probs[:, 1].cpu().numpy())
        test_targets.extend(y.cpu().numpy())
        _, predicted = torch.max(outputs.data, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()

test_auc = roc_auc_score(test_targets, test_preds)
accuracy = 100 * correct / total
print(f'Test Accuracy: {accuracy:.3f}%, AUC: {test_auc:.3f}')
test_fpr, test_tpr, _ = roc_curve(test_targets, test_preds)
test_auc = auc(test_fpr, test_tpr)



# model.load_state_dict(torch.load('./weights/81_901_94.pth'))