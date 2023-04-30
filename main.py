import numpy as np
from preprocessing.data_preprocessing import prepare
from metrics.metrics import dice_coefficient
from keras.metrics import OneHotIoU
from training.training import train_unet
from models.unet import UNet
from models.attention_unet import AttentionUnet
from data_loader.lits_dataloader import LiverTumorDataloader
from pathlib import Path
input_shape = (128, 128, 1)
target_shape = (input_shape[0], input_shape[1])
metrics = [OneHotIoU(3, [0, 1, 2], "IoU"), dice_coefficient]
filters = [16, 32, 64, 128, 256]
if __name__ == '__main__':
    data_file = Path("LiverTumorDataset/images.npy")
    if not data_file.exists():
        dataloader = LiverTumorDataloader("LiverTumorDataset/", target_shape=target_shape)
        images, masks = dataloader.load_dataset()

        np.save("LiverTumorDataset/images.npy", images)
        np.save("LiverTumorDataset/masks.npy", masks)

    images = np.load("LiverTumorDataset/images.npy")
    masks = np.load("LiverTumorDataset/masks.npy")

    train_images, train_masks, test_images, test_masks, test_images_to_plot = prepare(images, masks, num_classes=3)
    #
    # unet = UNet(input_shape, metrics, filters, summary=False)
    # unet_model = unet.create_model()
    #
    # train_unet(unet, unet_model, train_images, train_masks)
    #
    att_unet = AttentionUnet(input_shape, metrics, filters, summary=False)
    att_unet_model = att_unet.create_model()

    train_unet(att_unet, att_unet_model, train_images, train_masks)