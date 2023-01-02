
import torch
import torchvision.models as models

# from ..my_models.efficientnet import EfficientNet


def network(name, pretrained = True, num_classes = 1000, whole_network = True ):

    if name == "mobilenet_v2":
        model_nn = models.mobilenet_v2(pretrained=pretrained)

        for param in model_nn.parameters():
            param.requires_grad = whole_network
        # num_ftrs = model_nn.classifier.in_features
        model_nn.classifier = torch.nn.Sequential(
                torch.nn.Dropout(0.5),
                torch.nn.Linear(model_nn.last_channel, num_classes )
                )

    if name == "mobilenet_v3_large":
        model_nn = models.mobilenet_v3_large(pretrained=pretrained)

        for param in model_nn.parameters():
            param.requires_grad = whole_network

        num_ftrs = model_nn.classifier[3].in_features

        model_nn.classifier[3]= torch.nn.Linear(num_ftrs, num_classes)
                
    if name == "mobilenet_v3_small":
        model_nn = models.mobilenet_v3_small(pretrained=pretrained)

        for param in model_nn.parameters():
            param.requires_grad = whole_network

        num_ftrs = model_nn.classifier[3].in_features

        model_nn.classifier[3]= torch.nn.Linear(num_ftrs, num_classes)

    if name == "resnet18":
        model_nn = models.resnet18(pretrained=pretrained)

        for param in model_nn.parameters():
            param.requires_grad = whole_network

        num_ftrs = model_nn.fc.in_features
        model_nn.fc = torch.nn.Linear(num_ftrs, num_classes)

    if name == "resnet34":
        model_nn = models.resnet34(pretrained=pretrained)

        for param in model_nn.parameters():
            param.requires_grad = whole_network

        num_ftrs = model_nn.fc.in_features
        model_nn.fc = torch.nn.Linear(num_ftrs, num_classes)

    if name == "resnet50":
        model_nn = models.resnet50(pretrained=pretrained)

        for param in model_nn.parameters():
            param.requires_grad = whole_network

        num_ftrs = model_nn.fc.in_features
        model_nn.fc = torch.nn.Linear(num_ftrs, num_classes)

    if name == "efficientnet_b0":
        model_nn = models.efficientnet_b0(pretrained=pretrained)

        for param in model_nn.parameters():
            param.requires_grad = whole_network

        lastconv_output_channels = model_nn.classifier[1].in_features

        model_nn.classifier[1] = torch.nn.Linear(lastconv_output_channels, num_classes)

    if name == "efficientnet_b1":
        model_nn = models.efficientnet_b1(pretrained=pretrained)

        for param in model_nn.parameters():
            param.requires_grad = whole_network

        lastconv_output_channels = model_nn.classifier[1].in_features

        model_nn.classifier[1] = torch.nn.Linear(lastconv_output_channels, num_classes)

    if name == "my_efficientnet_b0":

        model_nn = EfficientNet(version="b0", num_classes=num_classes)


    return model_nn



# if(__name__ == "__main__"):

#     nn = network("mobilenetv2")