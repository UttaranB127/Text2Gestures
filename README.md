# Text2Gestures: A Transformer-Based Network for Generating Emotive Body Gestures for Virtual Agents

This is the readme to use the official code for the paper [Text2Gestures: A Transformer Network for Generating Emotive Body Gestures for Virtual Agents](http://arxiv.org/abs/2101.11101). Please add the following citation if you use our method.

```
@inproceedings{bhattacharya2020text2gestures,
author = {Uttaran Bhattacharya and Nicholas Rewkowski and Abhishek Banerjee and Pooja Guhan and Aniket Bera and Dinesh Manocha},
title = {Text2Gestures: A Transformer-Based Network for Generating Emotive Body Gestures for Virtual Agents},
booktitle = {2021 {IEEE} Conference on Virtual Reality and 3D User Interfaces (IEEE VR)},
publisher = {{IEEE}},
year      = {2021}
}
```

Our code is tested on Ubuntu 18.04 LTS with python 3.7. Please add the following citation in your work if you use our code:

## Installation

1. Clone this repository.

We use $BASE to refer to the base directory for this project (the directory containing `main.py`). Change present working directory to $BASE.

2. [Optional but recommended] Create a conda envrionment for the project and activate it.

```
conda create t2g-env python=3.7
conda activate t2g-env
```

3. Install the package requirements.

```
pip install -r requirements.txt
```
Note: You might need to manually uninstall and reinstall `matplotlib` and `kiwisolver` for them to work.

4. Install PyTorch following the [official instructions](https://pytorch.org/).
Note: You might need to manually uninstall and reinstall `numpy` for `torch` to work.

## Downloading the dataset
1. The original dataset is available for download [here](http://ebmdb.tuebingen.mpg.de/), but the samples need to be downloaded individually.

We have scraped the full dataset and made it available at this anonymous link: https://drive.google.com/file/d/1BhnC-puHTh0ax8hyq00Yfny2GK_Nz--k/view?usp=sharing

2. If downloading from our anonymous link, unzip the downloaded file to a directorty named "data", located at the same level at the project root (i.e., the project root and the data are sibling directories).

## Running the code
Run the `main.py` file with the appropriate command line arguments.
```
python main.py <args list>
```

The full list of arguments is available inside `main.py`.

For any argument not specificed in the command line, the code uses the default value for that argument.

On running `main.py`, the code will train the network and generate sample gestures post-training.

We also provide a pretrained model inside the `models` directory. Set the command-line argument `--train` to `False` to skip training and use this model directly for evaluation. The generated samples are stored in the automatically created `render` directory. We generate all the 145 test samples by deafult and also store the corresponding ground truth samples for comparison. We have tested that the samples, stored in `.bvh` files, are compatible with blender.
