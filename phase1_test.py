import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from models.vggnet import vggnet16
from models.resnet import resnet34
from models.fpn_resnet import resnet_fpn
from models.fpn_resnet2 import res_fpn_adv_decoder

### Training parameters ###
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


MODEL_PATH = 'resnet_fpn_adv_decoder_animated_or_real_15.pth'

DATA_TRANSFORM = transforms.Compose([transforms.ToTensor(), transforms.Resize((224,224))])


def test_model():

    ### Creating the model ###
    print('CREATING MODEL')
    model = res_fpn_adv_decoder()
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    model.to(DEVICE)

    ### Training ###
    print('GETTING DATA')
    test_data = ImageFolder('./Dataset/Test', transform=DATA_TRANSFORM)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=True, num_workers=0)

    count = 0
    correct_count = 0
    Y = []
    Y_pred = []
    for i, data in enumerate(test_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, img_class = data
        inputs = inputs.to(DEVICE)
        gt_class = 0
        if img_class[0] == 2:
            gt_class = 1
        Y.append(gt_class)

        # forward + backward + optimize
        outputs = model(inputs)
        if torch.argmax(outputs) == gt_class:
            correct_count += 1
        Y_pred.append(int(torch.argmax(outputs)))
        count += 1

    print(f'{MODEL_PATH}')
    print(classification_report(Y, Y_pred, target_names=['Animated', 'DUTS'], zero_division=0))
    print(confusion_matrix(Y, Y_pred))

if __name__ == '__main__':
    test_model()
