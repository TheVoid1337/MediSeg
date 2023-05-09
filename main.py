from data_loader.lits_dataloader import LiverTumorDataloader
from preprocessing.data_preprocessing import prepare
from training.training import train_models
from evaluation.train_eval import TrainEvaluator

target_shape = (128, 128)
train = False
if __name__ == '__main__':

    # dataloader = LiverTumorDataloader("LiverTumorDataset/", target_shape=target_shape)
    #
    # images, masks = dataloader.load_dataset()
    #
    # train_images, train_masks, test_images, test_masks, test_images_to_plot = prepare(images, masks, num_classes=3)
    #
    # if train:
    #     train_models(train_images, train_masks)

    train_eval = TrainEvaluator("nets/logs/", "results/training_plots/")

    train_eval.create_plots()