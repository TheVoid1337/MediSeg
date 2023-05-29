import nibabel as nib
import os
from pathlib import Path
import numpy as np
import pandas as pd
import cv2


class LiverTumorDataloader:
    def __init__(self, dataset_path: str, target_shape=(128, 128)):
        """
        The LiverTumorDataLoader loads the Liver Tumor Dataset by creating a dataframe if not exists.
        :param dataset_path: path to the dataset.
        :param target_shape: preferred target shape in which the data should be scaled to.
        """
        self.path = dataset_path
        self.target_shape = target_shape

    def read_files(self, data_frame: pd.DataFrame):
        """
        Loads the image data from the dataframe
        :param data_frame: pandas dataframe which contains directory paths and file names.
        :return: images, masks (numpy array)
        """
        images = []
        masks = []

        for row in data_frame.iterrows():
            image = str(row[1]["dirname"]) + "/" + str(row[1]["filename"])
            image = nib.load(image).get_fdata()
            image = self.reshape_images(image, np.float32)

            mask = str(row[1]["mask_dirname"]) + "/" + str(row[1]["mask_filename"])
            mask = nib.load(mask).get_fdata()
            mask = self.reshape_images(mask, np.uint8)

            images.append(image)
            masks.append(mask)

        images = np.concatenate(images)
        masks = np.concatenate(masks)

        return np.array(images).astype(np.float32), np.array(masks).astype(np.uint8)

    def reshape_images(self, image_data, data_type):
        """
        Reshapes the images to the required target_shape. cv2.resize is used to resize the images with the
        area interpolation algorithm.
        :param image_data: images to reshape
        :param data_type: numpy data type for rescaling image memory (float32 or uint8)
        :return: stack of rescaled images
        """

        images = []
        for i in range(0, image_data.shape[2]):
            image = cv2.resize(image_data[:, :, i], self.target_shape, interpolation=cv2.INTER_AREA)
            images.append(image)
        images = np.array(images).astype(data_type)
        return images

    def create_dataframe(self):
        """
        Creates/reads the required dataframe from the LiverTumorDataset and puts the metadata.csv into the
        root dataset directory if not exists.
        :return: pandas dataframe
        """
        file_path = Path(f"{self.path}/metadata.csv")
        if not file_path.is_file():

            file_list = []
            for dirname, _, filenames in os.walk(self.path):
                for filename in filenames:
                    file_list.append((dirname, filename))

            df_files = pd.DataFrame(file_list, columns=['dirname', 'filename'])
            df_files.sort_values(by=['filename'], ascending=True)
            df_files["mask_dirname"] = ""
            df_files["mask_filename"] = ""
            for i in range(131):
                ct = f"volume-{i}.nii"
                mask = f"segmentation-{i}.nii"

                df_files.loc[df_files['filename'] == ct, 'mask_filename'] = mask
                df_files.loc[df_files['filename'] == ct, 'mask_dirname'] = f"{self.path}/segmentations"

            df_files = df_files[df_files.mask_filename != ''].sort_values(by=['filename']).reset_index(drop=True)

            df_files.to_csv(f"{self.path}/metadata.csv", sep=",", index=False)
        else:
            df_files = pd.read_csv(f"{self.path}/metadata.csv", sep=",")
        return df_files

    def filter_not_relevant_data(self, img_list, mask_list):
        """
        Shrinks the dataset to relevant data only. Relevant data is defined as the amount of mask labels unequal to zero
        representing in sum of at least 1% of an image. The zero Label represents the background.
        :param img_list: image data.
        :param mask_list: mask data.
        :return: list of necessary images and masks for training.
        """
        images = []
        masks = []

        for i in range(len(mask_list)):
            val, counts = np.unique(mask_list[i], return_counts=True)
            if (1 - (counts[0] / counts.sum())) > 0.001:
                if len(counts) == 3:
                    if (counts[2] / counts.sum()) > 0.001:
                        images.append(img_list[i])
                        masks.append(mask_list[i])
                else:
                    images.append(img_list[i])
                    masks.append(mask_list[i])

        return np.array(images).astype(np.float32), np.array(masks).astype(np.uint8)

    def load_dataset(self):
        """
        Loads the Liver Tumor Dataset from a dataframe. If the dataframe does not exist, the dataloader will create it.
        :return: images, masks (numpy array)
        """
        data_file = Path(f"{self.path}images.npy")
        if not data_file.exists():
            df = self.create_dataframe()
            images, masks = self.read_files(df)
            return self.filter_not_relevant_data(images, masks)
        else:
            return self.load_numpy_data()

    def save_data(self, images, masks):
        """
        Save the data in a numpy binary format.
        :param images: image data.
        :param masks: mask data.
        :return: nothing
        """
        np.save(f"{self.path}images.npy", images)
        np.save(f"{self.path}masks.npy", masks)

    def load_numpy_data(self):
        """
        Loads the dataset from the datasets directory.
        :return: dataset with image and mask data as numpy array.
        """
        images = np.load(f"{self.path}images.npy")
        masks = np.load(f"{self.path}masks.npy")
        return images, masks
