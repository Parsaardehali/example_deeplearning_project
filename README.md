# Example project for training a Deeplearning model using PyTorch

## How to install?
1. Clone the repository
```bash
git clone https://github.com/Parsaardehali/example_deeplearning_project.git
```

```bash
cd example_deeplearning_project
```

2. Create a virtual environment (recommended)
```bash 
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```
## How to use?
This jupyter notebook gives an overall view of a simple project. Specially for a segmentation task.
```Example_project.ipynb```


## How to train the model?
First you need to have some data. You can simulate it by running 
```bash 
python Simulator.py
```
> Adjust the number of images (num_samples) and size of each image (img_size) in the last line of the script to what you need.


Preprocessing and splitting data into _train_ an _test_ sets is done by
```bash
python preprocess_split.py
```
> Adjust the split ratio of test data (test_size) in the script to what you need. Default is 0.2.


> Set  _data_dir_, _train_path_ and _test_path_ carefully .


> We do not do preprocessing here, but you can add it as a function in this script.


To train the model make sure your environment is activated by running 
```bash
source venv/bin/activate
```
And to run the training run ```python main.py --configconfigs/config_file.json```.
You can edit the config file to adjust the hyperparameters and paths for training and test datasets.

**Play around with loss function, for example using MSE loss instead of Dice loss.
You can also try different architectures and optimizers.**


## How to contribute?
If you have any suggestions or find any bugs, please create an issue or a pull request.


