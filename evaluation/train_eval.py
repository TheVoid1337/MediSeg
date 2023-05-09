import os
import matplotlib.pyplot as plt

import pandas as pd


class TrainEvaluator:
    def __init__(self, log_file_path: str, image_path: str):
        self.path = log_file_path
        self.image_path = image_path

    def read_logs(self):
        log_files = os.listdir(self.path)
        log_data = []
        for file in log_files:
            log_data.append(pd.read_csv(f"{self.path}{file}", sep=",", index_col="epoch"))
        return log_data, log_files

    def create_loss_plot(self, log_data: list[pd.DataFrame], names: list[str]):
        for log, name in zip(log_data, names):
            log["loss"].plot()
            log["val_loss"].plot()
            plt.legend(["Trainingsverlust", "Validierungsverlust"])
            plt.xlabel("Epoche")
            plt.ylabel("Kreuzentropieverlust")
            img_name = name.replace(".log", "_loss.jpg")
            plt.savefig(f"{self.image_path}{img_name}")
            plt.close()

    def create_dice_plot(self, log_data: list[pd.DataFrame], names: list[str]):
        for log, name in zip(log_data, names):
            log["dice_coefficient"].plot()
            log["val_dice_coefficient"].plot()
            plt.legend(["Dice-Koeffizient Training", "Dice-Koeffizient Validierung"])
            plt.xlabel("Epoche")
            plt.ylabel("Dice-Koeffizient")
            img_name = name.replace(".log", "_dice.jpg")
            plt.savefig(f"{self.image_path}{img_name}")
            plt.close()

    def create_iou_plot(self, log_data: list[pd.DataFrame], names: list[str]):
        for log, name in zip(log_data, names):
            log["IoU"].plot()
            log["val_IoU"].plot()
            plt.legend(["IoU Training", "IoU Validierung"])
            plt.xlabel("Epoche")
            plt.ylabel("IoU")
            img_name = name.replace(".log", "_iou.jpg")
            plt.savefig(f"{self.image_path}{img_name}")
            plt.close()

    def create_plots(self):
        log_data, file_names = self.read_logs()
        self.create_loss_plot(log_data, file_names)
        self.create_dice_plot(log_data, file_names)
        self.create_iou_plot(log_data, file_names)
