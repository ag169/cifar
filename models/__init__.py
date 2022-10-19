from models.cnn_1 import CNN1
from models.cnn_2 import CNN2
from models.resnext_50 import ResNeXT50
from models.resnet18 import ResNet18


def get_model(model_arch):
    model_arch = model_arch.lower()
    if model_arch == 'cnn1':
        model = CNN1()
    elif model_arch == 'cnn2':
        model = CNN2()
    elif model_arch == 'r18':
        model = ResNet18()
    elif model_arch == 'rnxt50':
        model = ResNeXT50()
    else:
        raise ValueError('Invalid model arch!')

    return model

