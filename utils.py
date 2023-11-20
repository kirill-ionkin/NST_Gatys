import os


import torch
import torchvision.transforms as transforms

from PIL import Image
import matplotlib.pyplot as plt


imgsize = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def gram_matrix(x):

    batch_size, c, h, w = x.size()
    x_flat = x.view(c, h * w)

    return torch.mm(x_flat, x_flat.t()).div(batch_size * c * h * w)


loader = transforms.Compose(
    [transforms.Resize(imgsize), transforms.CenterCrop(imgsize), transforms.ToTensor()]
)


unloader = transforms.ToPILImage()


def image_loader(image_name):
    image = Image.open(image_name)
    global imgsize
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


def imshow(
    tensor,
    title=None,
    save_folder=None,
    num_step=None,
    style_weight=None,
    content_weight=None,
):

    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)

    plt.tick_params(
        axis="both",
        which="both",
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False,
    )

    if title is not None:
        plt.title(title)

    if save_folder is not None:
        path_to_save = os.path.join(
            os.path.abspath(save_folder),
            "c_{:.1e}_s_{:.1e}_step_{}.jpg".format(
                int(content_weight), int(style_weight), num_step
            ),
        )
        image.save(path_to_save, format="JPEG")

    plt.imshow(image)
    plt.pause(0.001)


def plot_loss_history(history_loss, s, c, visual_threshold=None, path_to_save=None):

    plt.figure(figsize=(15, 10))
    fig, (ax_1, ax_2) = plt.subplots(2, 1)
    fig.set_size_inches((15, 10))

    ax_1.plot(
        [50 * i for i in range(1, 41)],
        [loss[0] for loss in history_loss],
        label="content loss",
        color="orange",
        marker="o",
    )
    ax_1.plot(
        [50 * i for i in range(1, 41)],
        [loss[1] for loss in history_loss],
        label="style loss",
        marker="o",
    )
    ax_1.grid(True)
    ax_1.set_title(f"style(s_weight={s}) and content(c_weight={c}) loss")
    ax_1.set_xlabel("steps")
    ax_1.legend()

    loss = [loss[0] + loss[1] for loss in history_loss]
    ax_2.plot(
        [50 * i for i in range(1, 41)], loss, label="loss", color="green", marker="o"
    )
    ax_2.grid(True)
    if visual_threshold is not None:
        ax_2.vlines(
            x=visual_threshold,
            ymin=min(loss),
            ymax=max(loss),
            linestyles="dashed",
            colors="r",
        )
    ax_2.set_title("history loss and visual threshold")
    ax_2.set_xlabel("steps")
    ax_2.legend()

    plt.tight_layout()
    fig.savefig(path_to_save, format="png")


def create_collage(
    path_to_folder,
    list_of_images,
    path_to_save,
    save_name,
    n_rows=4,
    n_cols=11,
    img_size=256,
):
    list_of_images_path = [
        os.path.join(path_to_folder, image) for image in list_of_images
    ]

    all_img_width = img_size * n_cols
    all_img_height = img_size * n_rows

    collage_img = Image.new(mode="RGB", size=(all_img_width, all_img_height))

    images = []
    for path in list_of_images_path:
        img = Image.open(path)
        img.thumbnail((img_size, img_size))
        images.append(img)

    i = 0
    x = 0
    y = 0
    for _ in range(n_rows):
        for _ in range(n_cols):
            collage_img.paste(images[i], (x, y))
            i += 1
            x += img_size
        y += img_size
        x = 0

    collage_img.save(os.path.join(path_to_save, save_name))
