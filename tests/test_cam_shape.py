import numpy as np
import torch
import pytest
from yolo_cam.base_cam import EigenCAM
from yolo_cam.utils.image import scale_cam_image
from ultralytics import YOLO

@pytest.fixture
def dummy_frame():
    H, W = 123, 456
    return (np.random.rand(H, W, 3) * 255).astype(np.uint8)

def test_cam_mask_matches_input(dummy_frame):
    model = YOLO("yolov11n.pt") 
    target_layers = [model.model.model[-2]]
    cam = EigenCAM(model, target_layers, task="cls")

    grayscale_cam = cam(dummy_frame, eigen_smooth=True)[0]

    assert grayscale_cam.shape == dummy_frame.shape[:2], (
        f"CAM shape {grayscale_cam.shape} != image shape {dummy_frame.shape[:2]}"
    )

    cam_image = scale_cam_image(grayscale_cam[None], target_size=dummy_frame.shape[1::-1])[0]
    assert cam_image.shape == dummy_frame.shape
