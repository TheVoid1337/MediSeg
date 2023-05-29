import random

import matplotlib.pyplot as plt
import numpy as np
from keras.models import Model


def create_images(test_images, test_masks, test_images_to_plot, unet: Model, att_unet: Model,
                  lstm_unet: Model, att_lstm_unet: Model, num_images=20):
    for i in range(num_images):
        img = random.randint(0, len(test_images))
        fig, axes = plt.subplots(4, 3, figsize=(7, 7))
        rows = ['{}'.format(row) for row in ['U-Net', 'Attention U-Net', 'LSTM U-Net', 'Attention LSTM U-Net']]
        for ax, row in zip(axes[:, 0], rows):
            ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - 5, 0),
                        xycoords=ax.yaxis.label, textcoords='offset points',
                        size='large', ha='right', va='center')
            ax.set_xticks([])
            ax.set_yticks([])

        # U-Net
        plt.subplot(431)
        plt.title("CT-Eingabe")
        plt.imshow(test_images_to_plot[img], cmap="gray")

        plt.subplot(432)
        test_mask = np.argmax(test_masks[img], axis=-1)
        plt.axis("off")
        plt.title("Vorgabe")
        plt.imshow(test_mask, cmap="bone")

        plt.subplot(433)
        test_image = np.expand_dims(test_images[img], axis=0)
        pred = unet.predict(test_image)
        pred = np.argmax(pred, axis=-1)[0, :, :]
        plt.axis("off")
        plt.title("Vorhersage")
        plt.imshow(pred, cmap="bone")

        # Attention U-Net
        plt.subplot(434)
        plt.title("CT-Eingabe")
        plt.imshow(test_images_to_plot[img], cmap="gray")

        plt.subplot(435)
        plt.title("Vorgabe")
        test_mask = np.argmax(test_masks[img], axis=-1)
        plt.axis("off")
        plt.imshow(test_mask, cmap="bone")

        plt.subplot(436)
        test_image = np.expand_dims(test_images[img], axis=0)
        pred = att_unet.predict(test_image)
        pred = np.argmax(pred, axis=-1)[0, :, :]
        plt.axis("off")
        plt.title("Vorhersage")
        plt.imshow(pred, cmap="bone")

        # LSTM U-Net
        plt.subplot(437)
        plt.title("CT-Eingabe")
        plt.imshow(test_images_to_plot[img], cmap="gray")

        plt.subplot(438)
        plt.title("Vorgabe")
        test_mask = np.argmax(test_masks[img], axis=-1)
        plt.axis("off")
        plt.imshow(test_mask, cmap="bone")

        plt.subplot(439)
        test_image = np.expand_dims(test_images[img], axis=0)
        pred = lstm_unet.predict(test_image)
        pred = np.argmax(pred, axis=-1)[0, :, :]
        plt.axis("off")
        plt.title("Vorhersage")
        plt.imshow(pred, cmap="bone")

        # Attention LSTM U-Net
        plt.subplot(4, 3, 10)
        plt.title("CT-Eingabe")
        plt.imshow(test_images_to_plot[img], cmap="gray")
        plt.subplot(4, 3, 11)
        plt.title("Vorgabe")
        test_mask = np.argmax(test_masks[img], axis=-1)
        plt.axis("off")
        plt.imshow(test_mask, cmap="bone")

        plt.subplot(4, 3, 12)
        plt.title("Vorhersage")
        test_image = np.expand_dims(test_images[img], axis=0)
        pred = att_lstm_unet.predict(test_image)
        pred = np.argmax(pred, axis=-1)[0, :, :]
        plt.axis("off")
        plt.imshow(pred, cmap="bone")

        fig.tight_layout()

        plt.savefig(f"results/plots/prediction_{i}.jpg")
        plt.close()
