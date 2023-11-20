import time


import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt


from layers import Normalization, ContentLoss, StyleLoss
from utils import imshow


content_layers_default = ["conv4_2"]
style_layers_default = ["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vgg19_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
vgg19_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

vgg_19_layers_names = {
    0: "conv1_1",
    1: "relu1_1",
    2: "conv1_2",
    3: "relu1_2",
    4: "pool1",
    5: "conv2_1",
    6: "relu2_1",
    7: "conv2_2",
    8: "relu2_2",
    9: "pool2",
    10: "conv3_1",
    11: "relu3_1",
    12: "conv3_2",
    13: "relu3_2",
    14: "conv3_3",
    15: "relu3_3",
    16: "conv3_4",
    17: "relu3_4",
    18: "pool3",
    19: "conv4_1",
    20: "relu4_1",
    21: "conv4_2",
    22: "relu4_2",
    23: "conv4_3",
    24: "relu4_3",
    25: "conv4_4",
    26: "relu4_4",
    27: "pool4",
    28: "conv5_1",
    29: "relu5_1",
    30: "conv5_2",
    31: "relu5_2",
    32: "conv5_3",
    33: "relu5_3",
    34: "conv5_4",
    35: "relu5_4",
    36: "pool5",
}


def create_nst_model(
    pretrained_vgg19_model,
    content_img,
    style_img,
    normalization_mean,
    normalization_std,
    pooling_mode="avg",
    content_layers=content_layers_default,
    style_layers=style_layers_default,
    device=device,
):

    normalization = Normalization(normalization_mean, normalization_std).to(device)

    nst_model = nn.Sequential(normalization)
    content_loss_layers = []
    style_loss_layers = []

    c = 1
    s = 1
    for i, layer_i in enumerate(pretrained_vgg19_model.features.children()):
        name_layer_i = vgg_19_layers_names[i]

        if (name_layer_i.startswith("pool")) and (pooling_mode == "avg"):
            layer_i = nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False)

        if name_layer_i.startswith("relu"):
            layer_i = nn.ReLU(inplace=False)

        nst_model.add_module(name_layer_i, layer_i)

        if name_layer_i in content_layers:
            content_target = nst_model(content_img)
            content_loss_layer = ContentLoss(content_target)
            nst_model.add_module(f"content_loss_layer_{c}", content_loss_layer)
            c += 1

            content_loss_layers.append(content_loss_layer)

        if name_layer_i in style_layers:
            style_target = nst_model(style_img)
            style_loss_layer = StyleLoss(style_target)
            nst_model.add_module(f"style_loss_layer_{s}", style_loss_layer)
            s += 1

            style_loss_layers.append(style_loss_layer)

        if (len(style_loss_layers) == len(style_layers)) and (
            len(content_loss_layers) == len(content_layers)
        ):
            break

    for content_loss_layer in content_loss_layers:
        content_loss_layer.mode = "loss"

    for style_loss_layer in style_loss_layers:
        style_loss_layer.mode = "loss"

    # to save video-memory
    torch.cuda.empty_cache()

    return nst_model, content_loss_layers, style_loss_layers


def get_input_optimizer(input_img):
    return optim.LBFGS([input_img.requires_grad_()])


def run_style_transfer(
    pretrained_vgg19_model,
    content_img,
    style_img,
    input_img=None,
    save_folder=None,
    normalization_mean=vgg19_normalization_mean,
    normalization_std=vgg19_normalization_std,
    content_layers=content_layers_default,
    style_layers=style_layers_default,
    num_steps=300,
    style_weight=1e5,
    content_weight=1,
    return_history_loss=False,
):

    if input_img is None:
        input_img = torch.randn(
            size=content_img.size(), requires_grad=True, device=device
        )

    start = time.time()
    print("Building the style transfer model..")

    nst_model, content_loss_layers, style_loss_layers = create_nst_model(
        pretrained_vgg19_model=pretrained_vgg19_model,
        content_img=content_img,
        style_img=style_img,
        normalization_mean=normalization_mean,
        normalization_std=normalization_std,
        content_layers=content_layers,
        style_layers=style_layers,
    )

    for p in nst_model.parameters():
        p.requires_grad = False

    optimizer = get_input_optimizer(input_img)

    print("Optimizing..")
    run = [0]
    while run[0] <= num_steps:

        def closure():
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()

            nst_model(input_img)

            all_style_loss = 0
            all_content_loss = 0

            for sl in style_loss_layers:
                all_style_loss += sl.loss
            for cl in content_loss_layers:
                all_content_loss += cl.loss

            all_style_loss *= style_weight
            all_content_loss *= content_weight

            loss = all_style_loss + all_content_loss
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                # print("run {}:".format(run))
                # print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                #    all_style_loss, all_content_loss))
                # print()
                string_to_print = "run {} steps \n".format(run[0])
                string_to_print += "Style Loss: {:.2f} Content Loss: {:.2f} \n".format(
                    all_style_loss.item(), all_content_loss.item()
                )
                string_to_print += "Spend time: {:.1f}".format(time.time() - start)

                plt.figure()
                if save_folder is None:
                    imshow(input_img, title=string_to_print)
                else:
                    imshow(
                        input_img,
                        title=string_to_print,
                        save_folder=save_folder,
                        num_step=run[0],
                        style_weight=style_weight,
                        content_weight=content_weight,
                    )

                if return_history_loss:
                    global history_loss
                    history_loss.append(
                        (all_content_loss.item(), all_style_loss.item())
                    )

            return all_style_loss + all_content_loss

        optimizer.step(closure)

    input_img.data.clamp_(0, 1)

    return input_img
