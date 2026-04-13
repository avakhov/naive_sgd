import os
import json
import torch
import torch.nn as nn
from figures import points

torch.manual_seed(123)
os.makedirs("out", exist_ok=True)

SNAP_POINTS = 100
NUM_SNAPSHOTS = 20

class SimpleNN(nn.Module):
    def __init__(self, n0, n1, n2, n3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n0, n1),
            nn.Tanh(),
            nn.Linear(n1, n2),
            nn.Tanh(),
            nn.Linear(n2, n3),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


dataset = points("heart", 100)

t_data = torch.tensor([[row[0]] for row in dataset], dtype=torch.float32)
xy_data = torch.tensor([[row[1], row[2]] for row in dataset], dtype=torch.float32)

epochs = 6000
model = SimpleNN(n0=1, n1=20, n2=20, n3=2)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)
snap_epochs = {i * (epochs - 1) // (NUM_SNAPSHOTS - 1) for i in range(NUM_SNAPSHOTS)}
t_snap = torch.tensor([[i / SNAP_POINTS] for i in range(SNAP_POINTS + 1)], dtype=torch.float32)

snapshots = []

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    pred = model(t_data)
    loss = ((pred - xy_data) ** 2).mean()
    loss.backward()
    optimizer.step()
    scheduler.step()

    if epoch % 100 == 0:
        print(f"epoch={epoch}, loss={loss.item():.6f}")

    if epoch in snap_epochs:
        model.eval()
        with torch.no_grad():
            out = model(t_snap).numpy()
        snapshots.append({
            "epoch": epoch,
            "x": out[:, 0].tolist(),
            "y": out[:, 1].tolist(),
        })

target_x = [row[1] for row in dataset]
target_y = [row[2] for row in dataset]

output = {
    "target": {"x": target_x, "y": target_y},
    "snapshots": snapshots,
}

with open("out/output.json", "w") as f:
    json.dump(output, f)

print("out/output.json saved.")
