import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from loadmnist import train_loader


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)

        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.leaky_relu(x, 0.001)

        x = self.conv2(x)
        x = F.leaky_relu(x, 0.001)

        x = F.max_pool2d(x, 2)

        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = F.leaky_relu(x, 0.001)

        x = self.fc2(x)

        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("using:", device)

model = SimpleCNN().to(device=device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 10

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for x, y in train_loader:
        # transport to GPU
        x = x.to(device)
        y = y.to(device)

        # initialize
        optimizer.zero_grad()

        # prediction
        output = model(x)

        # loss
        loss = criterion(output, y)

        # backward
        loss.backward()

        # renew
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} Loss: {total_loss:.3f}")
