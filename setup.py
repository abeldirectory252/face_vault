from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="face_vault",
    version="0.1.0",
    description="Fast face recognition with anti-spoofing and vector database",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="FaceVault",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "insightface>=0.7",
        "onnxruntime>=1.14",
        "opencv-python>=4.8",
        "numpy>=1.24",
        "scipy>=1.10",
        "faiss-cpu>=1.7",
        "Pillow>=9.0",
    ],
    extras_require={
        "gpu": ["onnxruntime-gpu>=1.14"],
    },
    package_data={
        "face_vault": ["../img/*.png"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
