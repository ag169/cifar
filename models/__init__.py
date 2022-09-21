from models.cnn_1 import CNN1


def get_model(model_arch):
    model_arch = model_arch.lower()
    if model_arch == 'cnn1':
        model = CNN1()
    else:
        raise ValueError('Invalid model arch!')

    return model

