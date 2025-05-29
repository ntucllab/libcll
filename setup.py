from setuptools import setup

with open("./requirements.txt") as f:
    install_requires = f.read().splitlines()

setup(
    name="libcll",
    version="1.0.0",
    description="ibcll: an extendable python toolkit for complementary-label learning",
    author="N.-X. Ye, T.-H. Mai, H.-H. Wang, W.-I. Lin, H.-T. Lin",
    author_email="b09902008@csie.ntu.edu.tw, d10922024@csie.ntu.edu.tw, b09902033@csie.ntu.edu.tw, empennage98@gmail.com",
    url="https://github.com/ntucllab/libcll",
    install_requires=install_requires,
    classifiers=[
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    test_suite="libcll",
    packages=[
        "libcll",
        "libcll.datasets",
        "libcll.models",
        "libcll.strategies",
    ],
    package_dir={
        "libcll": "libcll",
        "libcll.datasets": "libcll/datasets",
        "libcll.models": "libcll/models",
        "libcll.strategies": "libcll/strategies",
    },
)
