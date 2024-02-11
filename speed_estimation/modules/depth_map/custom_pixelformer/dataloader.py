import os

import numpy as np
import torch
import torch.utils.data.distributed
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import secrets


def _is_pil_image(img):
    """Checks if the input is a PIL Image object.
    Parameters:
        - img (Image.Image): Input image to be checked.
    Returns:
        - bool: True if input is a PIL Image object, False otherwise.
    Processing Logic:
        - Check if input is an instance of Image.Image.
        - Returns a boolean value."""
    
    return isinstance(img, Image.Image)


def _is_numpy_image(img):
    """This function checks if the input is a numpy array representing an image.
    Parameters:
        - img (numpy.ndarray): The input image to be checked.
    Returns:
        - bool: True if the input is a numpy array representing an image, False otherwise.
    Processing Logic:
        - Check if input is a numpy array.
        - Check if input has 2 or 3 dimensions."""
    
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def preprocessing_transforms(mode):
    """"Creates a Compose object with a ToTensor transform based on the specified mode.
    Parameters:
        - mode (str): The mode to be used for the ToTensor transform. Valid modes are 'rgb' and 'grayscale'.
    Returns:
        - transforms.Compose: A Compose object with a ToTensor transform based on the specified mode.
    Processing Logic:
        - Creates a Compose object with a single ToTensor transform.
        - The mode parameter determines the type of ToTensor transform to be used.
        - Valid modes are 'rgb' for RGB images and 'grayscale' for grayscale images.
        - The ToTensor transform converts the input image to a PyTorch tensor.
    """"
    
    return transforms.Compose([ToTensor(mode=mode)])


class NewDataLoader(object):
    def __init__(
        self, args, mode, *, file_list: "list[str]", data_path: str, do_kb_crop: bool
    ):
        """Function:
        def __init__(
            self, args, mode, *, file_list: "list[str]", data_path: str, do_kb_crop: bool
        ):
            Initializes the DataLoader object for training, testing, or online evaluation.
            Parameters:
                - args (type): Command line arguments.
                - mode (str): The mode in which the DataLoader object will be used.
                - file_list (list[str]): A list of file names.
                - data_path (str): The path to the data.
                - do_kb_crop (bool): A boolean value indicating whether or not to crop the data.
            Returns:
                - DataLoader: The DataLoader object for the specified mode.
            Processing Logic:
                - Initializes the DataLoader object with the specified arguments.
                - Checks the mode to ensure it is valid.
                - Creates a DataLoader object with the specified mode, file list, data path, and crop setting."""
        
        if mode == "test":
            self.testing_samples = DataLoadPreprocess(
                mode,
                file_list=file_list,
                data_path=data_path,
                do_kb_crop=do_kb_crop,
                transform=preprocessing_transforms(mode),
            )
            self.data = DataLoader(
                self.testing_samples, 1, shuffle=False, num_workers=1
            )

        else:
            print(
                "mode should be one of 'train, test, online_eval'. Got {}".format(mode)
            )


class DataLoadPreprocess(Dataset):
    def __init__(
        self,
        mode,
        """()
        "Initializes a data loader object for a specific dataset and mode.
        Parameters:
            - mode (str): The mode of the data loader, either 'train' or 'test'.
            - file_list (list[str]): A list of file names to be loaded.
            - data_path (str): The path to the dataset.
            - do_kb_crop (bool): Whether to perform cropping on the images.
            - transform (callable, optional): A function/transform that takes in an image and returns a transformed version. Default: None.
        Returns:
            - DataLoader: A data loader object for the specified dataset and mode.
        Processing Logic:
            - Initializes the data loader object.
            - Sets the mode, file list, data path, and cropping option.
            - Sets the transform function if provided.
            - Converts the images to tensors using the ToTensor() function.
        """"
        
        *,
        file_list: "list[str]",
        data_path: str,
        do_kb_crop: bool,
        transform=None
    ):
        self.file_list = file_list
        self.data_path = data_path
        self.do_kb_crop = do_kb_crop

        self.mode = mode
        self.transform = transform
        self.to_tensor = ToTensor

    def __getitem__(self, idx):
        """"""
        
        sample_path = self.file_list[idx]
        # focal = float(sample_path.split()[2])
        focal = 518.8579

        data_path = self.data_path

        image_path = os.path.join(data_path, "./" + sample_path.split()[0])
        image = np.asarray(Image.open(image_path), dtype=np.float32) / 255.0

        if self.do_kb_crop is True:
            height = image.shape[0]
            width = image.shape[1]
            top_margin = int(height - 352)
            left_margin = int((width - 1216) / 2)
            image = image[
                top_margin : top_margin + 352, left_margin : left_margin + 1216, :
            ]

        sample = {"image": image, "focal": focal}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def random_crop(self, img, depth, height, width):
        """Randomly crops an image and its corresponding depth map to the specified dimensions.
        Parameters:
            - img (numpy array): The original image to be cropped.
            - depth (numpy array): The original depth map to be cropped.
            - height (int): The desired height of the cropped image and depth map.
            - width (int): The desired width of the cropped image and depth map.
        Returns:
            - img (numpy array): The cropped image.
            - depth (numpy array): The cropped depth map.
        Processing Logic:
            - Asserts that the image is larger than the desired height and width.
            - Asserts that the image and depth map have the same dimensions.
            - Randomly generates x and y coordinates within the image.
            - Crops the image and depth map to the specified dimensions using the generated coordinates."""
        
        assert img.shape[0] >= height
        assert img.shape[1] >= width
        assert img.shape[0] == depth.shape[0]
        assert img.shape[1] == depth.shape[1]
        x = secrets.SystemRandom().randint(0, img.shape[1] - width)
        y = secrets.SystemRandom().randint(0, img.shape[0] - height)
        img = img[y : y + height, x : x + width, :]
        depth = depth[y : y + height, x : x + width, :]
        return img, depth

    def __len__(self):
        """Function to return the length of the file list.
        Parameters:
            - self (object): Object of the class.
        Returns:
            - int: Length of the file list.
        Processing Logic:
            - Get the length of the file list.
            - Return the length."""
        
        return len(self.file_list)


class ToTensor(object):
    def __init__(self, mode):
        """Function to initialize a class with a given mode and normalization parameters.
        Parameters:
            - mode (str): The mode to be used for the class.
        Returns:
            - None: This function does not return any value.
        Processing Logic:
            - Initialize class with given mode.
            - Set normalization parameters.
            - Mean values are [0.485, 0.456, 0.406].
            - Standard deviation values are [0.229, 0.224, 0.225]."""
        
        self.mode = mode
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def __call__(self, sample):
        """Preprocesses and returns a sample from the dataset for training or testing.
        Parameters:
            - sample (dict): A dictionary containing the image, depth, focal, and path of the sample.
        Returns:
            - dict: A dictionary containing the preprocessed image, depth, focal, and path of the sample.
        Processing Logic:
            - Converts image to tensor.
            - Normalizes image.
            - If mode is "test", returns preprocessed image and focal.
            - If mode is "train", converts depth to tensor and returns preprocessed image, depth, and focal.
            - Otherwise, returns preprocessed image, depth, focal, has_valid_depth, and path."""
        
        image, focal = sample["image"], sample["focal"]
        image = self.to_tensor(image)
        image = self.normalize(image)

        if self.mode == "test":
            return {"image": image, "focal": focal}

        depth = sample["depth"]
        if self.mode == "train":
            depth = self.to_tensor(depth)
            return {"image": image, "depth": depth, "focal": focal}
        else:
            has_valid_depth = sample["has_valid_depth"]
            return {
                "image": image,
                "depth": depth,
                "focal": focal,
                "has_valid_depth": has_valid_depth,
                "path": sample["path"],
            }

    def to_tensor(self, pic):
        """Converts a PIL Image or numpy array to a torch tensor.
        Parameters:
            - pic (PIL Image or ndarray): The image to be converted.
        Returns:
            - img (Tensor): The converted tensor.
        Processing Logic:
            - Checks if pic is a PIL Image or numpy array.
            - Converts numpy array to tensor if applicable.
            - Handles different modes of PIL Image.
            - Transposes and converts the tensor to float if necessary.
        Example:
            to_tensor(pic)"""
        
        if not (_is_pil_image(pic) or _is_numpy_image(pic)):
            raise TypeError(
                "pic should be PIL Image or ndarray. Got {}".format(type(pic))
            )

        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img

        # handle PIL Image
        if pic.mode == "I":
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == "I;16":
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == "YCbCr":
            nchannel = 3
        elif pic.mode == "I;16":
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)

        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float()
        else:
            return img
