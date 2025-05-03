import numpy as np
import pytest
from ultralytics import YOLO
from yolo_cam.eigen_cam import EigenCAM
from yolo_cam.utils.image import show_cam_on_image

@pytest.fixture
def dummy_frame():
    H, W = 123, 456
    return (np.random.rand(H, W, 3) * 255).astype(np.uint8)

def test_cam_mask_matches_input(dummy_frame):
    model = YOLO("yolo11n.pt")
    target_layers = [model.model.model[-2]]

    cam = EigenCAM(model, target_layers, task="cls")
    grayscale_cam = cam(dummy_frame, eigen_smooth=True)[0]

    assert grayscale_cam.shape == dummy_frame.shape[:2], (
        f"CAM shape {grayscale_cam.shape} != image shape {dummy_frame.shape[:2]}"
    )

    cam_image = show_cam_on_image(dummy_frame / 255.0, grayscale_cam, use_rgb=True)
    assert cam_image.shape == dummy_frame.shape, (
        f"Overlay shape {cam_image.shape} != image shape {dummy_frame.shape}"
    )
