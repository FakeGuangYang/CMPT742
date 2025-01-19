from datetime import datetime
from model import UNet
from dataloader import CellData
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
import os

from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb

# Parameters
# learning rate
lr = 1e-3
# number of training epochs
epoch_n = 50
# input image-mask size
image_size = 512
# root directory of project
root_dir = os.getcwd()
# training batch size
batch_size = 8
# use checkpoint model for training
load = False
# use GPU for training
gpu = True
# data directory
data_dir = os.path.join(root_dir, 'data/cells/')

train_set = CellData(data_dir=data_dir, size=image_size)
train_loader = DataLoader(train_set, batch_size=4, shuffle=True)

test_set = CellData(data_dir=data_dir, size=image_size, train=False, augment_data=False)
test_loader = DataLoader(test_set, batch_size=4)

# see if it's macbook or windows
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

# device = torch.device('cuda:0' if gpu else 'cpu')

# model = UNet().to('cuda:0').to(device)
model = UNet().to(device)

if load:
    print('loading model')
    model.load_state_dict(torch.load('checkpoint.pt'))

# loss function
criterion = nn.CrossEntropyLoss()

# optimizer
# optimizer = optim.Adam(model.parameters(), lr=lr, momentum=0.99, weight_decay=0.0005)
optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.85, 0.0999), weight_decay=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-5)

# initialize weights and bias
wandb.login(key="f8ac37ce5fae7e4decd2a1ea5bc3711f690f5205")
wandb.init(project="Assignment 2", name="Image Segmentation")
# initialize hyperparameters
config = {
    "learning_rate": 1e-3,
    "epochs": 50,
    "batch_size": 8,
    "weight_decay": 1e-4
}

wandb.config.update(config)

model.train()

# record start time
start_time = datetime.now()

# record

for epoch in range(epoch_n):
    epoch_loss = 0
    model.train()
    for i, data in enumerate(train_loader):
        image, label = data
        # image = image.unsqueeze(1).to(device)
        image = image.to(device)
        label = label.long().to(device)
        label = label.squeeze(1)

        pred = model(image)

        crop_x = (label.shape[1] - pred.shape[2]) // 2
        crop_y = (label.shape[2] - pred.shape[3]) // 2

        label = label[:, crop_x: label.shape[1] - crop_x, crop_y: label.shape[2] - crop_y]

        loss = criterion(pred, label)

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        epoch_loss += loss.item()

        print('batch %d --- Loss: %.4f' % (i, loss.item() / batch_size))
    print('Epoch %d / %d --- Loss: %.4f' % (epoch + 1, epoch_n, epoch_loss / train_set.__len__()))

    torch.save(model.state_dict(), 'checkpoint.pt')

    model.eval()

    total = 0
    correct = 0
    total_loss = 0

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            image, label = data

            # image = image.unsqueeze(1).to(device)
            image = image.to(device)
            label = label.long().to(device)
            label = label.squeeze(1)

            pred = model(image)
            crop_x = (label.shape[1] - pred.shape[2]) // 2
            crop_y = (label.shape[2] - pred.shape[3]) // 2

            label = label[:, crop_x: label.shape[1] - crop_x, crop_y: label.shape[2] - crop_y]

            loss = criterion(pred, label)
            total_loss += loss.item()

            _, pred_labels = torch.max(pred, dim=1)

            total += label.shape[0] * label.shape[1] * label.shape[2]
            correct += (pred_labels == label).sum().item()

        scheduler.step(total_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print('Accuracy: %.4f ---- Loss: %.4f' % (correct / total, total_loss / test_set.__len__()))
        print("Current LR: %.7f" % current_lr)
        wandb.log({"epoch": epoch + 1, "train_loss": epoch_loss / train_set.__len__(), "val_loss": total_loss / test_set.__len__()})

# record end time
end_time = datetime.now()
training_duration_seconds = int((end_time - start_time).total_seconds())
print("The training took us " + str(training_duration_seconds) + " seconds for " + str(epoch_n) + " epochs.")

# save model
torch.save(model.state_dict(), "model.pth")
wandb.save("model.pth")

# testing and visualization
model.eval()

output_masks = []
output_labels = []

with torch.no_grad():
    for i in range(test_set.__len__()):
        image, label = test_set.__getitem__(i)
        label = label[0, :, :]

        # input_image = image.unsqueeze(0).unsqueeze(0).to(device)
        input_image = image.unsqueeze(0).to(device)
        pred = model(input_image)

        output_mask = torch.max(pred, dim=1)[1].cpu().squeeze(0).numpy()

        crop_x = (label.shape[0] - output_mask.shape[0]) // 2
        crop_y = (label.shape[1] - output_mask.shape[1]) // 2
        label = label[crop_x: label.shape[0] - crop_x, crop_y: label.shape[1] - crop_y].numpy()

        output_masks.append(output_mask)
        output_labels.append(label)

fig, axes = plt.subplots(test_set.__len__(), 2, figsize=(20, 20))

for i in range(test_set.__len__()):
    axes[i, 0].imshow(output_labels[i])
    axes[i, 0].axis('off')
    axes[i, 1].imshow(output_masks[i])
    axes[i, 1].axis('off')

plt.show()
