import os
import cv2
import rasterio
import rasterio.windows
import numpy as np

import torch
from PIL import Image
from samgeo.text_sam import LangSAM
from tqdm import tqdm


def array_to_image(
    array, output, source=None, dtype=None, compress="deflate", **kwargs
):
    """Save a NumPy array as a GeoTIFF using the projection information from an existing GeoTIFF file.

    Args:
        array (np.ndarray): The NumPy array to be saved as a GeoTIFF.
        output (str): The path to the output image.
        source (str, optional): The path to an existing GeoTIFF file with map projection information. Defaults to None.
        dtype (np.dtype, optional): The data type of the output array. Defaults to None.
        compress (str, optional): The compression method. Can be one of the following: "deflate", "lzw", "packbits", "jpeg". Defaults to "deflate".
    """

    from PIL import Image

    if isinstance(array, str) and os.path.exists(array):
        array = cv2.imread(array)
        array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)

    if output.endswith(".tif") and source is not None:
        with rasterio.open(source) as src:
            crs = src.crs
            transform = src.transform
            if compress is None:
                compress = src.compression

        # Determine the minimum and maximum values in the array
        min_value = np.min(array)
        max_value = np.max(array)

        if dtype is None:
            # Determine the best dtype for the array
            if min_value >= 0 and max_value <= 1:
                dtype = np.float32
            elif min_value >= 0 and max_value <= 255:
                dtype = np.uint8
            elif min_value >= -128 and max_value <= 127:
                dtype = np.int8
            elif min_value >= 0 and max_value <= 65535:
                dtype = np.uint16
            elif min_value >= -32768 and max_value <= 32767:
                dtype = np.int16
            else:
                dtype = np.float64

        # Convert the array to the best dtype
        array = array.astype(dtype)

        # Define the GeoTIFF metadata
        if array.ndim == 2:
            metadata = {
                "driver": "GTiff",
                "height": array.shape[0],
                "width": array.shape[1],
                "count": 1,
                "dtype": array.dtype,
                "crs": crs,
                "transform": transform,
            }
        elif array.ndim == 3:
            metadata = {
                "driver": "GTiff",
                "height": array.shape[0],
                "width": array.shape[1],
                "count": array.shape[2],
                "dtype": array.dtype,
                "crs": crs,
                "transform": transform,
            }

        if compress is not None:
            metadata["compress"] = compress
        else:
            raise ValueError("Array must be 2D or 3D.")

        # Create a new GeoTIFF file and write the array to it
        with rasterio.open(output, "w", **metadata) as dst:
            if array.ndim == 2:
                dst.write(array, 1)
            elif array.ndim == 3:
                for i in range(array.shape[2]):
                    dst.write(array[:, :, i], i + 1)

    else:
        img = Image.fromarray(array)
        img.save(output, **kwargs)


class MySubclass(LangSAM):
    def predict(
        self,
        image,
        text_prompt,
        box_threshold,
        text_threshold,
        output=None,
        mask_multiplier=255,
        dtype=np.uint8,
        save_args={},
        return_results=False,
        **kwargs,
    ):
        """
        Run both GroundingDINO and SAM model prediction.

        Parameters:
            image (Image): Input PIL Image.
            text_prompt (str): Text prompt for the model.
            box_threshold (float): Box threshold for the prediction.
            text_threshold (float): Text threshold for the prediction.
            output (str, optional): Output path for the prediction. Defaults to None.
            mask_multiplier (int, optional): Mask multiplier for the prediction. Defaults to 255.
            dtype (np.dtype, optional): Data type for the prediction. Defaults to np.uint8.
            save_args (dict, optional): Save arguments for the prediction. Defaults to {}.
            return_results (bool, optional): Whether to return the results. Defaults to False.

        Returns:
            tuple: Tuple containing masks, boxes, phrases, and logits.
        """
        image_np = image.transpose((1, 2, 0))
        image_pil = Image.fromarray(image_np[:, :, :3])
        
        self.image = image_pil

        boxes, logits, phrases = self.predict_dino(
            image_pil, text_prompt, box_threshold, text_threshold
        )
        masks = torch.tensor([])
        if len(boxes) > 0:
            masks = self.predict_sam(image_pil, boxes)
            masks = masks.squeeze(1)

        if boxes.nelement() == 0:  # No "object" instances found
            print("No objects found in the image.")
            # return
        else:
            # Create an empty image to store the mask overlays
            mask_overlay = np.zeros_like(
                image_np[..., 0], dtype=dtype
            )  # Adjusted for single channel

            for i, (box, mask) in enumerate(zip(boxes, masks)):
                # Convert tensor to numpy array if necessary and ensure it contains integers
                if isinstance(mask, torch.Tensor):
                    mask = (
                        mask.cpu().numpy().astype(dtype)
                    )  # If mask is on GPU, use .cpu() before .numpy()
                mask_overlay += ((mask > 0) * (i + 1)).astype(
                    dtype
                )  # Assign a unique value for each mask

            # Normalize mask_overlay to be in [0, 255]
            mask_overlay = (
                mask_overlay > 0
            ) * mask_multiplier  # Binary mask in [0, 255]

        if output is not None:
            array_to_image(mask_overlay, output, self.source, dtype=dtype, **save_args)
            
        self.masks = masks
        self.boxes = boxes
        self.phrases = phrases
        self.logits = logits
        self.mask_overlay = mask_overlay

        if return_results:
            return masks, boxes, phrases, logits, mask_overlay


def find_window(width, height, range_win= [1000,1500]):
    min_x, max_x = range_win
    size_win = None
    for x in range(max_x + 1, min_x, -1):
        if width % x > 100 and height % x > 100 and height % x < x and width % x < x:
            size_win = x
            break
    return size_win



def read_image_in_windows(out_fp, filename, text_prompt, chose_overlap=False):
    with rasterio.open(filename) as src:
        width = src.width
        height = src.height
        meta = src.meta
        meta.update({'count':1})
        
        if width*height > 1500*1500:
            window_size = find_window(width, height)
            if chose_overlap:
                overlap = window_size//4
            else:
                overlap = 0
            with rasterio.open(out_fp, 'w', **meta) as dst:
                for col in tqdm(range(0, width, window_size - overlap)):
                    for row in tqdm(range(0, height, window_size - overlap)):
                        width_win = min(window_size, width - col)
                        height_win = min(window_size, height - row)
                        window = rasterio.windows.Window(col, row, width_win, height_win)
                        # print(col, row, width_win, height_win)
                        data = src.read(window=window)
                        if np.all(data == 0):
                            mask_overlay = np.zeros((width_win, height_win)) + 4
                        else:
                            try:
                                _, _, _, _, mask_overlay= sam.predict(data, text_prompt, box_threshold=0.24, text_threshold=0.6, return_results=True)
                            except:
                                mask_overlay = np.ones((width_win, height_win))
                        dst.write(mask_overlay, window=window, indexes=1)
        else:
            data = src.read()
            sam.predict(data, text_prompt, box_threshold=0.24, text_threshold=0.24, output=out_fp)
        

text_prompt = "car"
overlab = 0
fp_image = r"/home/skm/SKM16/Data/Test_Docker/ml-models/pretrain-models/sam_text/v1/example-data/HaDong.tif"
# fp_image = r"/home/skm/SKM16/Data/Test_Docker/ml-models/pretrain-models/sam_text/v1/example-data/img7x1/Bi_sai_tree.tif"
fp_out_img = f"/home/skm/SKM16/Data/Test_Docker/ml-models/pretrain-models/sam_text/v1/RS/POC_AOI_Here_VN_Final_Image_10CM_{text_prompt}_mask_{overlab}_100-250.tif"
chkpnt_dir = r"/home/skm/SKM/WORK/ALL_CODE/WorkSpace_SKM_DucAnh/SAM/model/"
checkpoint = os.path.join(chkpnt_dir, "sam_vit_h_4b8939"+".pth")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
sam = MySubclass()
read_image_in_windows(fp_out_img, fp_image, text_prompt, overlab)

# sam.predict(image, text_prompt, box_threshold=0.24, text_threshold=0.24, return_results=True, output=fp_out_img)
# sam.raster_to_vector(fp_out_img, fp_out_shp)
