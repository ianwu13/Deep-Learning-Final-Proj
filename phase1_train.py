import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models.vggnet import vggnet16
from models.resnet import resnet34

### IMPORTANT ###
LOAD_PAST_MODEL = False

### Training parameters ###
EPOCHS = 15
BATCH_SIZE = 4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', DEVICE)

MODEL_PATH = 'resnet34_animated_or_real.pth'

DATA_TRANSFORM = transforms.Compose([transforms.ToTensor(), transforms.Resize((224,224))])


def train_model():

    ### Creating the model ###
    print('CREATING MODEL')
    model = resnet34()
    if LOAD_PAST_MODEL:
        model.load_state_dict(torch.load(MODEL_PATH))
    model.to(DEVICE)

    ### Training ###
    print('GETTING DATA')
    train_data = ImageFolder('./Dataset/Train', transform=DATA_TRANSFORM)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    print('TRAINING MODEL')
    # NOT BCELOSS()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    for epoch in range(EPOCHS):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # zero the parameter gradients
            optimizer.zero_grad()

            # get the inputs; data is a list of [inputs, labels]
            inputs, img_class = data
            inputs = inputs.to(DEVICE)
            img_class = torch.where(img_class == 2, 1, 0).to(DEVICE)
            
            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, img_class)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            #if i % 100 == 99:  # print every 100 mini-batches
            print('[%d, %5d] loss: %.15f' %
                (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

            if epoch%2 == 0:
                torch.save(model.state_dict(), MODEL_PATH)


    ### Save the model ###
    print('SAVING MODEL')
    torch.save(model.state_dict(), MODEL_PATH)


if __name__ == '__main__':
    train_model()
