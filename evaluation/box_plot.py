import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def create_boxplot(file_path: str, filename: str):
    dataframe = pd.read_csv(file_path)
    unet = dataframe["unet_iou"]
    plt.figure(figsize=(9,9))
    plt.rcParams.update({'font.size': 18})
    plt.tight_layout()
    print("U-Net IoU:" + str(np.round(unet.mean(), 3)) + "\tStandard Deviation:" + str(np.round(unet.std(), 3)))
    att_unet = dataframe["att_unet_iou"]
    print("Attention U-Net IoU:" + str(np.round(att_unet.mean(), 3)) +
          "\tStandard Deviation:" + str(np.round(att_unet.std(), 3)))
    lstm_unet = dataframe["lstm_unet_iou"]
    print("LSTM U-Net IoU:" + str(np.round(lstm_unet.mean(),3)) +
          "\tStandard Deviation:" + str(np.round(lstm_unet.std(),3)))
    att_lstm_unet = dataframe["att_lstm_unet_iou"]
    print("Attention LSTM U-Net IoU:" + str(np.round(att_lstm_unet.mean(),3)) +
          "\tStandard Deviation:" + str(np.round(att_lstm_unet.std(),3)))

    plt.boxplot([unet, att_unet, lstm_unet, att_lstm_unet], labels=["U-Net", "Attention",
                                                                    "LSTM", "Attention & LSTM"], showfliers=False,
                showmeans=True)
    plt.ylabel("Jaccard-Koeffizient")
    plt.savefig(f"results/graphics/{filename}_iou.jpg")
    plt.close()

    plt.figure(figsize=(9,9))
    plt.rcParams.update({'font.size': 18})
    plt.tight_layout()
    unet = dataframe["unet_dice"]
    print("U-Net Dice:" + str(np.round(unet.mean(),3)) + "\tStandard Deviation" + str(np.round(unet.std(),3)))
    att_unet = dataframe["att_unet_dice"]
    print("Attention U-Net Dice:" + str(np.round(att_unet.mean(),3)) +
          "\tStandard Deviation:" + str(np.round(att_unet.std(),3)))
    lstm_unet = dataframe["lstm_unet_dice"]
    print("LSTM U-Net Dice:" + str(np.round(lstm_unet.mean(),3)) +
          "\tStandard Deviation:" + str(np.round(lstm_unet.std(),3)))
    att_lstm_unet = dataframe["att_lstm_unet_dice"]
    print("Attention LSTM U-Net Dice:" + str(np.round(att_lstm_unet.mean(),3)) +
          "\tStandard Deviation:" + str(np.round(att_lstm_unet.std(),3)))

    plt.boxplot([unet, att_unet, lstm_unet, att_lstm_unet], labels=["U-Net", "Attention",
                                                                    "LSTM", "Attention & LSTM"], showfliers=False,
                showmeans=True,showcaps=True)
    plt.ylabel("Dice-Koeffizient")
    plt.savefig(f"results/graphics/{filename}_dice.jpg")
    plt.close()
