from setuptools import setup, find_packages

__version__ = '1.0.0'

setup(
    name='a2c',
    version=__version__,
    url='https://github.com/mind/a2c',
    packages=find_packages(),
    install_requires=[
        'gym[mujoco,atari,classic_control]',
        'cloudpickle',
        'joblib',
        'opencv-python',
        'tensorflow >= 1.3.0',
    ],
)
