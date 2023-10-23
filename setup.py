from setuptools import setup
from setuptools import find_packages

setup(
    name="llpkt_single",
    version="0.1",
    description="Lifelong Personalized Knowledge Tracing",
    license="MIT",
    install_requires=["numpy",
                      "scipy",
                      "torch",
                      "pandas",
                      "tqdm",
                      "scikit-learn",
                      "scikit-image",
                      "Pillow",
                      "easydict",
                      "matplotlib",
                      "tensorboardX",
                      "torchvision",
                      "more-itertools"],
    package_data={"llpkt_single": ["README.md"]},
    packages=find_packages()
)
