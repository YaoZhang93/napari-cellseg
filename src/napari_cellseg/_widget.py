"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/plugins/guides.html?#widgets

Replace code below according to your needs.
"""
from typing import TYPE_CHECKING

from magicgui import magic_factory

from napari.utils.notifications import show_info

if TYPE_CHECKING:
    import napari

from enum import Enum
import os
join = os.path.join
import time
import numpy as np
# from skimage.filters import threshold_otsu
# from skimage.measure import label
import torch
import monai
from monai.inferers import sliding_window_inference
from .models.unetr2d import UNETR2D
import time
from skimage import io, segmentation, morphology, measure, exposure
import pathlib

class ModelName(Enum):
    UNet = 'unet'
    VNet = 'vnet'
    UNETR = 'unetr'
    SwinUNETR = 'swinunetr'


def load_model(model_name, custom_model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_name == 'unet':
        model = monai.networks.nets.UNet(
                   spatial_dims=2,
                   in_channels=3,
                   out_channels=3,
                   channels=(16, 32, 64, 128, 256),
                   strides=(2, 2, 2, 2),
                   num_res_units=2,
               )
    elif model_name == 'vnet':
        model = monai.networks.nets.VNet(
                   spatial_dims=2,
                   in_channels=16,
                   out_channels=3,
                   dropout_prob=0.0
               ) 
    elif model_name == 'unetr':
        model = UNETR2D(
                    in_channels=3,
                    out_channels=3,
                    img_size=(256, 256),
                    feature_size=16,
                    hidden_size=768,
                    mlp_dim=3072,
                    num_heads=12,
                    pos_embed="perceptron",
                    norm_name="instance",
                    res_block=True,
                    dropout_rate=0.0,
                )
    elif model_name == 'swinunetr':
        model = monai.networks.nets.SwinUNETR(
                        img_size=(256, 256), 
                        in_channels=3, 
                        out_channels=3,
                        feature_size=24, # should be divisible by 12
                        spatial_dims=2
                    )
        if os.path.isfile(custom_model_path):
            checkpoint = torch.load(custom_model_path.resolve(), map_location=torch.device(device))
        elif os.path.isfile(join(os.path.dirname(__file__), 'work_dir/swinunetr/best_Dice_model.pth')):
            checkpoint = torch.load(join(os.path.dirname(__file__), 'work_dir/swinunetr/best_Dice_model.pth'), map_location=torch.device(device))
        else:
            torch.hub.download_url_to_file('https://zenodo.org/record/6792177/files/best_Dice_model.pth?download=1', join(os.path.dirname(__file__), 'work_dir/swinunetr/best_Dice_model.pth'))
            checkpoint = torch.load(join(os.path.dirname(__file__), 'work_dir/swinunetr/best_Dice_model.pth'), map_location=torch.device(device))

        model.load_state_dict(checkpoint['model_state_dict'])

    model = model.to(device)
    model.eval()

    return model


@magic_factory
def cellseg_widget(img_layer: "napari.layers.Image", model_name: ModelName, custom_model_path: pathlib.Path, threshold: float=0.5) -> "napari.types.LayerDataTuple":
    seg_labels = get_seg(preprocess(img_layer.data), model_name.value, custom_model_path, threshold)

    seg_layer = (seg_labels, {"name": f"{img_layer.name}_seg"}, "labels")
    return seg_layer


def normalize_channel(img, lower=1, upper=99):
    non_zero_vals = img[np.nonzero(img)]
    percentiles = np.percentile(non_zero_vals, [lower, upper])
    if percentiles[1] - percentiles[0] > 0.001:
        img_norm = exposure.rescale_intensity(img, in_range=(percentiles[0], percentiles[1]), out_range='uint8')
    else:
        img_norm = img
    return img_norm.astype(np.uint8)


def preprocess(img_data):
    if len(img_data.shape) == 2:
        img_data = np.repeat(np.expand_dims(img_data, axis=-1), 3, axis=-1)
    elif len(img_data.shape) == 3 and img_data.shape[-1] > 3:
        img_data = img_data[:,:, :3]
    else:
        pass
    pre_img_data = np.zeros(img_data.shape, dtype=np.uint8)
    for i in range(3):
        img_channel_i = img_data[:,:,i]
        if len(img_channel_i[np.nonzero(img_channel_i)])>0:
            pre_img_data[:,:,i] = normalize_channel(img_channel_i, lower=1, upper=99)
    return pre_img_data


def get_seg(pre_img_data, model_name, custom_model_path, threshold):
    model = load_model(model_name, custom_model_path)
    #%%
    roi_size = (256, 256)
    sw_batch_size = 4
    with torch.no_grad():
        t0 = time.time()
        test_npy01 = pre_img_data/np.max(pre_img_data)
        # test_tensor = torch.from_numpy(np.expand_dims(test_npy01, 0)).permute(0,3,1,2).type(torch.FloatTensor).to(device)
        test_tensor = torch.from_numpy(np.expand_dims(test_npy01, 0)).permute(0,3,1,2).type(torch.FloatTensor)
        test_pred_out = sliding_window_inference(test_tensor, roi_size, sw_batch_size, model)
        test_pred_out = torch.nn.functional.softmax(test_pred_out, dim=1) # (B, C, H, W)
        test_pred_npy = test_pred_out[0,1].cpu().numpy()
        # convert probability map to binary mask and apply morphological postprocessing
        test_pred_mask = measure.label(morphology.remove_small_objects(morphology.remove_small_holes(test_pred_npy>threshold),16))
        # tif.imwrite(join(output_path, img_name.split('.')[0]+'_label.tiff'), test_pred_mask, compression='zlib')
        t1 = time.time()
        # print(f'Prediction finished: {img_layer.name}; img size = {pre_img_data.shape}; costing: {t1-t0:.2f}s')
        print(f'Prediction finished; img size = {pre_img_data.shape}; costing: {t1-t0:.2f}s')
    show_info('segmentation finished')
    return test_pred_mask
