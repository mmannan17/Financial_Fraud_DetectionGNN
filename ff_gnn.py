import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



node_features = pd.read_csv('elliptic_txs_features.csv', header=None)
edges = pd.read_csv('elliptic_txs_edgelist.csv')
classes = pd.read_csv('elliptic_txs_classes.csv')

node_features = node_features.merge(classes, left_on=0, right_on='txId')
node_features.rename(columns={0: 'txId', 1: 'time_step'}, inplace=True)

features = node_features.iloc[:, 2:-1].values
labels = node_features['class'].values
time_steps = node_features['time_step'].values

# Updated label mapping
label_mapping = {'unknown': -1, '1': 0, '2': 1}
labels = np.array([label_mapping[label] for label in labels])

x = torch.tensor(features, dtype=torch.float).to(device)
y = torch.tensor(labels, dtype=torch.long).to(device)


unique_nodes = pd.unique(edges.values.ravel())
node_mapping = {node: i for i, node in enumerate(unique_nodes)}

# Re-mapping the edge indices
edges_mapped = edges.applymap(node_mapping.get)
edge_index = torch.tensor(edges_mapped.values.T, dtype=torch.long).to(device)

train_mask = torch.tensor((time_steps <= 34), dtype=torch.bool).to(device)
val_mask = torch.tensor((time_steps > 34) & (time_steps < 43), dtype=torch.bool).to(device)
test_mask = torch.tensor((time_steps >= 43), dtype=torch.bool).to(device)

data = Data(x=x, edge_index=edge_index, y=y)
data.train_mask = train_mask
data.val_mask = val_mask
data.test_mask = test_mask
data = data.to(device)



class FraudDetectionGNN(nn.Module):
    def __init__(self, num_features, hidden_channels):
        super(FraudDetectionGNN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = nn.Linear(hidden_channels, 2)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.lin(x)
        return x

model = FraudDetectionGNN(num_features=data.num_features, hidden_channels=64).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()



train_losses = []
val_losses = []
val_accuracies = []

def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    mask = data.train_mask & (data.y >= 0)
    loss = criterion(out[mask], data.y[mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def validate():
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        mask = data.val_mask & (data.y >= 0)
        pred = out[mask].argmax(dim=1)
        val_loss = criterion(out[mask], data.y[mask])
        val_acc = accuracy_score(data.y[mask].cpu(), pred.cpu())
    return val_loss.item(), val_acc

num_epochs = 50
best_val_acc = 0
patience = 5
patience_counter = 0

for epoch in range(1, num_epochs + 1):
    loss = train()
    val_loss, val_acc = validate()
    train_losses.append(loss)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)
    print(f'Epoch {epoch:02d}, Loss: {loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print('Early stopping triggered.')
            break



# Plot training metrics
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.savefig('training_metrics.png')
plt.close()



model.load_state_dict(torch.load('best_model.pth'))
model.eval()

with torch.no_grad():
    out = model(data.x, data.edge_index)
    mask = data.test_mask & (data.y >= 0)
    pred = out[mask].argmax(dim=1)
    prob = F.softmax(out[mask], dim=1)[:, 1]
    test_true = data.y[mask].cpu()
    test_pred = pred.cpu()
    test_prob = prob.cpu()

test_acc = accuracy_score(test_true, test_pred)
test_precision = precision_score(test_true, test_pred)
test_recall = recall_score(test_true, test_pred)
test_f1 = f1_score(test_true, test_pred)
conf_matrix = confusion_matrix(test_true, test_pred)

print(f'Test Accuracy:  {test_acc:.4f}')
print(f'Test Precision: {test_precision:.4f}')
print(f'Test Recall:    {test_recall:.4f}')
print(f'Test F1 Score:  {test_f1:.4f}')
print('Confusion Matrix:')
print(conf_matrix)

# Plot confusion matrix
plt.figure(figsize=(6, 5))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['Licit', 'Illicit'], rotation=45)
plt.yticks(tick_marks, ['Licit', 'Illicit'])
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.close()

# Plot ROC curve
fpr, tpr, thresholds = roc_curve(test_true, test_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.savefig('roc_curve.png')
plt.close()



torch.save(model.state_dict(), 'fraud_detection_gnn_final.pth')

print(edge_index.shape)
print(edge_index.max().item(), edge_index.min().item())

# Check the shape and range of edge_index
print("Edge index shape:", edge_index.shape)
print("Max index in edge_index:", edge_index.max().item())
print("Min index in edge_index:", edge_index.min().item())

# Check the number of nodes
print("Number of nodes:", x.size(0))
