from data_loader.lits_dataloader import LiverTumorDataloader
from evaluation.test_models import test_models
from preprocessing.data_preprocessing import prepare
from training.training import create_models

target_shape = (128, 128)
train = False
test = False
if __name__ == '__main__':
    dataloader = LiverTumorDataloader("LiverTumorDataset/", target_shape=target_shape)

    images, masks = dataloader.load_dataset()

    train_images, train_masks, test_images, test_masks, test_images_to_plot = prepare(images, masks, num_classes=3)

    unet_model, att_unet_model, lstm_unet_model, att_lstm_unet_model = create_models(train_images, train_masks)

    if test:
        test_models(unet_model, att_unet_model, lstm_unet_model, att_lstm_unet_model, test_images, test_masks)
