README FILE (Running the notebook on Kaggle)

# CS 676 – Real vs Fake Face Detection Using Neural Networks

## Overview : 
This project trains and compares multiple deep learning models (Custom CNN, ResNet50, VGG16) to detect AI-generated (fake) faces from real ones.


## Prerequisites : 
- Python **3.8 – 3.10** (TensorFlow does not support Python 3.11+ well)
- pip
- GPU is recommended as the notebook trains for 10 epochs across 6 models. CPU tends to be too slow and kernel dies after sometime as the CPU is unable to handle the load of the data. Try running either of the two CPUs present in the Settings > Accelerator > either of the GPU

## Note : For getting additional hours of GPU time, try adding and linking your Google Pro account if (purchased or on student ID) to get more additional GPU hours.


## Dataset
This notebook is designed to run on **Kaggle** and uses a dataset hosted there.

1. Go to Kaggle and search for the dataset: `ritwikpatil/realvsfakeimages`
2. Add it to your Kaggle notebook under **Add Data**
3. The dataset should be accessible at:
   - `/kaggle/input/datasets/ritwikpatil/realvsfakeimages/Real faces`
   - `/kaggle/input/datasets/ritwikpatil/realvsfakeimages/Fake faces`
   - `/kaggle/input/datasets/ritwikpatil/realvsfakeimages/metadata.csv`

> If running **locally** (not on Kaggle), update these paths in the notebook to wherever you store the dataset on your machine.

## This the dataset https://www.kaggle.com/datasets/troykueh/real-vs-fake-faces-stylegan3, that we used to get the files. For some reason the dataset was not being pulled in the kaggle notebook, so added the files in the path as mentioned above.


## Running on Kaggle (Recommended)
1. Go to [kaggle.com](https://www.kaggle.com) and sign in
2. Click **Create** → **New Notebook**
3. Upload the `.ipynb` file via **File** → **Import Notebook**
4. Add the dataset as described above
5. Under **Settings** (right panel), set **Accelerator** to **GPU T4 x2** or **GPU P100**
6. Click **Run All**

All required libraries (TensorFlow, OpenCV, scikit-learn, etc.) are **pre-installed** on Kaggle — no additional installation needed.


### 2. Launch Jupyter and run

```bash
jupyter notebook CS676_FinalProj_AlgoDS.ipynb
```
Then click **Kernel** → **Restart & Run All**.


## Required Libraries Summary

| Library | Purpose |
|||
| `tensorflow` | Model building and training |
| `opencv-python` (cv2) | Image reading and processing |
| `pandas` | Data manipulation |
| `numpy` | Numerical operations |
| `matplotlib` | Plotting |
| `seaborn` | Heatmap visualization |
| `scikit-learn` | Metrics and train/test split |
| `scipy` | Statistical utilities |


## Expected Runtime

| Environment | Approximate Time |
|||
| Kaggle GPU | ~1–2 hours |
| Local GPU (NVIDIA) | ~2–4 hours |
| Local CPU only | 10+ hours (not recommended) |


## Notes
- The notebook will copy about 20,000 images into the path `/kaggle/working/images/` during its first execution. This step is bypassed automatically if the specified directory is present beforehand.
- ResNet50 and VGG16 models will be downloaded from the Internet during their first execution (approx. 100–500 MB each). 
Ensure that you are connected to the Internet.
- Image loading through Keras' generators ("flow_from_dataframe") allows for efficient image handling and saves memory.


