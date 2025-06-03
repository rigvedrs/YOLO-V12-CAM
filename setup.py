from setuptools import setup, find_packages

setup(
    name='yolo_cam',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'torch',
        'opencv-python',
        'numpy',
        'matplotlib'
    ],
)
