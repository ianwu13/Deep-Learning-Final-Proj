import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models.vggnet import vggnet16
from models.resnet import resnet34

### Training parameters ###
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


MODEL_PATH = 'resnet34_animated_or_real_or_anime.pth'

DATA_TRANSFORM = transforms.Compose([transforms.ToTensor(), transforms.Resize((224,224))])


def test_model():

    ### Creating the model ###
    print('CREATING MODEL')
    model = resnet34(num_classes=3)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    model.to(DEVICE)

    ### Training ###
    print('GETTING DATA')
    test_data = ImageFolder('./Dataset/Test', transform=DATA_TRANSFORM)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=True, num_workers=0)

    count = 0
    correct_count = 0
    for i, data in enumerate(test_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, img_class = data
        inputs = inputs.to(DEVICE)
        img_class = img_class.to(DEVICE)

        # forward + backward + optimize
        outputs = model(inputs)
        if torch.argmax(outputs) == img_class:
            correct_count += 1
        count += 1

    print(f'Accuracy: {correct_count/count}')


if __name__ == '__main__':
    test_model()
