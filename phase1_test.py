import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models.vggnet import vggnet16

### Training parameters ###
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


MODEL_PATH = 'vggnet16_animated_or_real.pth'

DATA_TRANSFORM = transforms.Compose([transforms.ToTensor(), transforms.Resize((224,224))])


def test_model():

    ### Creating the model ###
    print('CREATING MODEL')
    model = vggnet16()
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    model.to(DEVICE)

    ### Training ###
    print('GETTING DATA')
    test_data = ImageFolder('./Dataset/Test', transform=DATA_TRANSFORM)
    test_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # zero the parameter gradients
        optimizer.zero_grad()

        # get the inputs; data is a list of [inputs, labels]
        inputs, img_class = data
        inputs = inputs.to(DEVICE)
        class_map = []
        for i in img_class:
            if i == 2:
                class_map.append([0, 1])
            else:
                class_map.append([1, 0])
        class_map = torch.tensor(class_map, dtype=torch.float).to(DEVICE)

        # forward + backward + optimize
        outputs = model(inputs)

        # print statistics
        running_loss += loss.item()
        if i % 100 == 99:  # print every 100 mini-batches
            print('[%d, %5d] loss: %.3f' %
                (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0


if __name__ == '__main__':
    test_model()
