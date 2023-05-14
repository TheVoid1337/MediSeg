import pandas as pd
import matplotlib.pyplot as plt


def create_boxplot(file_path: str, filename: str):
    dataframe = pd.read_csv(file_path)
    unet = dataframe["unet_iou"]
    att_unet = dataframe["att_unet_iou"]
    lstm_unet = dataframe["lstm_unet_iou"]
    att_lstm_unet = dataframe["att_lstm_unet_iou"]

    plt.boxplot([unet, att_unet, lstm_unet, att_lstm_unet], labels=["U-Net", "Attention U-Net",
                                                                    "LSTM U-Net", "Attention U-Net"], showfliers=False,
                showmeans=True)
    plt.ylabel("Jaccard-Koeffizient")
    plt.savefig(f"results/graphics/{filename}_iou.jpg")
    plt.close()

    unet = dataframe["unet_dice"]
    att_unet = dataframe["att_unet_dice"]
    lstm_unet = dataframe["lstm_unet_dice"]
    att_lstm_unet = dataframe["att_lstm_unet_dice"]

    plt.boxplot([unet, att_unet, lstm_unet, att_lstm_unet], labels=["U-Net", "Attention U-Net",
                                                                    "LSTM U-Net", "Attention U-Net"], showfliers=False,
                showmeans=True)
    plt.ylabel("Dice-Koeffizient")
    plt.savefig(f"results/graphics/{filename}_dice.jpg")
    plt.close()
