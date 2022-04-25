import cv2
import numpy as np
from sqlalchemy import Float
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models.vggnet import vggnet16

### Training parameters ###
EPOCHS = 2
BATCH_SIZE = 4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', DEVICE)

MODEL_PATH = 'vggnet16_animated_or_real.pth'

DATA_TRANSFORM = transforms.Compose([transforms.ToTensor(), transforms.Resize((224,224))])


def train_model():

    ### Creating the model ###
    print('CREATING MODEL')
    model = vggnet16()
    model.to(DEVICE)

    ### Training ###
    print('GETTING DATA')
    train_data = ImageFolder('./Dataset/Train', transform=DATA_TRANSFORM)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    print('TRAINING MODEL')
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    for epoch in range(EPOCHS):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # zero the parameter gradients
            optimizer.zero_grad()

            # get the inputs; data is a list of [inputs, labels]
            inputs, img_class = data
            inputs = inputs.to(DEVICE)
            class_map = []
            for j in img_class:
                if j == 2:
                    class_map.append([0, 1])
                else:
                    class_map.append([1, 0])
            class_map = torch.tensor(class_map, dtype=torch.float).to(DEVICE)

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, class_map)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            #if i % 100 == 99:  # print every 100 mini-batches
            print('[%d, %5d] loss: %.15f' %
                (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

    ### Save the model ###
    print('SAVING MODEL')
    torch.save(model.state_dict(), MODEL_PATH)


def test():
    pass


if __name__ == '__main__':
    test()
    train_model()
