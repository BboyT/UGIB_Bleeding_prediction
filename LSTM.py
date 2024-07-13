import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import roc_auc_score
import pandas as pd
import torch.nn.functional as F
from  tqdm import  tqdm
from torch.utils.data.sampler import WeightedRandomSampler
from imblearn.under_sampling import ClusterCentroids
from imblearn.over_sampling import SMOTE
if __name__ == '__main__':
    root = './data_re_logistic_regression'
    a = pd.read_excel(str(root + ('/before_icu.xlsx')), engine='openpyxl')
    b = pd.read_excel(str(root + ('/first_day_icu.xlsx')), engine='openpyxl')
    label = pd.read_excel(str(root + ('/label.xlsx')), engine='openpyxl')

    class BinaryClassifier(nn.Module):
        def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
            super(BinaryClassifier, self).__init__()

            self.hidden_dim = hidden_dim
            self.num_layers = num_layers

            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_dim, output_dim)
            self.sig = nn.Sigmoid()

        def forward(self, x):
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

            out, _ = self.lstm(x, (h0, c0))
            out = self.fc(out[:, -1, :])
            out = self.sig(out)
            return out

    input_dim = 20
    hidden_dim = 128
    num_layers = 8
    output_dim = 1
    model = BinaryClassifier(input_dim, hidden_dim, num_layers, output_dim)

    A = torch.tensor(a.values, dtype=torch.float)
    B = torch.tensor(b.values, dtype=torch.float)
    Y = torch.tensor(label.values.squeeze(), dtype=torch.float)

    A = F.normalize(A, dim=0)
    B = F.normalize(B, dim=0)

    inputs = torch.cat((A, B), dim=1).unsqueeze(1)
    inputs_2d = inputs.view(inputs.size(0), -1)


    dataset = TensorDataset(inputs, Y)

    train_size = int(0.7 * len(dataset))
    test_size = len(dataset) - train_size
    torch.manual_seed(2023)
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Extract labels from train_dataset
    train_labels = torch.tensor([label for _, label in train_dataset])

    # Convert train labels to int
    train_labels = train_labels.long()

    # Calculate class weights
    class_sample_count = torch.tensor([(train_labels == t).sum() for t in torch.unique(train_labels, sorted=True)])
    weight = 1. / class_sample_count.float()

    # Assign weight to each sample
    samples_weight = torch.tensor([weight[t] for t in train_labels])

    # Create WeightedRandomSampler
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

    train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    best_auc = 0
    num_epochs = 1000

    # Training Loop
    bar=tqdm(range(num_epochs))
    for epoch in bar:
        for i, (inputs, labels) in enumerate(train_loader):
            model.train()
            outputs = model(inputs)
            loss = criterion(outputs, labels.view(-1, 1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation Loop
        with torch.no_grad():
            model.eval()
            valid_outputs = []
            valid_labels = []
            for inputs, labels in test_loader:
                outputs = model(inputs)
                valid_outputs.extend(outputs.detach().numpy().flatten())
                valid_labels.extend(labels.numpy().flatten())
            auc = roc_auc_score(valid_labels, valid_outputs)
            if auc > best_auc:
                best_auc = auc
                torch.save(model.state_dict(), 'best_model.pt')
                print(f"Epoch {epoch + 1}, AUC improved to {best_auc}")
            # bar.set_description(f"Epoch {epoch + 1}, AUC improved to {best_auc}")


