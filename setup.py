from collections import defaultdict

from setuptools import find_packages, setup

with open("VERSION") as f:
    version = f.read()


with open("requirements.txt") as f:
    extras_require = defaultdict(list)
    section = None
    for line in f.read().splitlines():
        if line.startswith("#"):
            section = line[2:]
            continue
        extras_require[section].append(line)


install_requires = extras_require.pop("core")
dev, test = extras_require.pop("dev"), extras_require.pop("test")
extras_require["all"] = [dep for section in extras_require.values() for dep in section]
extras_require["dev"], extras_require["test"] = dev, test


setup(
    name="total-segmenter",
    version=version,
    description="Robust segmentation of 104 classes in CT images.",
    long_description=open("README.md").read(),
    url="https://github.com/wasserth/TotalSegmentator",
    author="Jakob Wasserthal",
    author_email="jakob.wasserthal@usb.ch",
    python_requires=">=3.8",
    license="Apache 2.0",
    packages=find_packages(),
    install_requires=install_requires,
    extras_require=dict(extras_require),
    entry_points={"console_scripts": ["totalsegmenter = totalsegmenter.cli:cli"]},
    zip_safe=False,
    classifiers=[
        "Intended Audience :: Science/Research",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering",
        "Operating System :: Unix",
        "Operating System :: MacOS",
    ],
)
