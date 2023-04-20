# Verification and Validation for Intelligent Systems

In this project we will simulate a camera + detection algorithm behavioral change. The current experiment was tested for the road signs dataset
from Kaggle: https://www.kaggle.com/datasets/andrewmvd/road-sign-detection. However, the tutorial will contain steps for any custom datasets.

## Objective: 
 
 -Simulate intelligent systems behavioral changes given hardware changes (tolerance to manufacture, technological refresh and maintenance and)<br>
 -Further investigate verification and validation techniques for adaptive yet brittle systems

## Install: 
First clone the repo. 
 ```console
git clone https://github.com/s1m2e3/V-V-for-AI.git

```
  The required packages are:
  <br>
 -yolov5 : https://github.com/ultralytics/yolov5. Make sure to clone the yolov5 repo in the father directory of this repo(this repo and yolov5 repo shall be separated).
 
 ```console
#from https://github.com/ultralytics/yolov5
git clone https://github.com/ultralytics/yolov5  # clone
cd yolov5
pip install -r requirements.txt  # install
```
 -pytorch : https://pytorch.org/
  ```console
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
```
 -PIL : https://pillow.readthedocs.io/en/stable/
  ```console
pip install --upgrade Pillow
```

 -numpy : https://numpy.org/
  ```console
pip install numpy
```
 -tqdm : https://numpy.org/
  ```console
https://pypi.org/project/tqdm/#description
```
-sklearn: https://scikit-learn.org/stable/install.html
 ```console
pip install -U scikit-learn
```

-imagecorruptions:  #https://github.com/bethgelab/imagecorruptions
```console
pip3 install imagecorruptions
```
## Tutorial for Custom Datasets
After cloning the repo download your dataset in the father directory of the repo. So if the repo is located in: 
```console
 C:\Documents\V-V-for-AI 
```
   
  The dataset shall be stored in:
```console
 C:\Documents\your_dataset
```
This packages has been developed to work with XML annotations style for the labels of the pictures used for training and testing. Furthermore,
the custom datasets shall have the following structure: 
```console
 C:\Documents\your_dataset\images
 C:\Documents\your_dataset\annotations
```

Four types of image corruptions have been explored to test the consistency of the detection algorithm:

- Doubling Resolution
- Halving Resolution
- Brightness Change
- Pixel Damaging

There exists three scripts that control the experiment:

### Corruption Script:
-The corruption script which is in charge to manipulate the datasets. Typically speaking we only manipulate the testing dataset assuming that the 
algorithm was trained on the best condition possible (clean training data) and it is supposed to be tested in negative conditions. This could not be necessarily the case thou.
Sometimes the system could have on-line training capacities allowing it to adapt to newer conditions. In this case we would do the training with corrupted data and test it with the same corruption as well.

As an example assume that you only want to corrupt the testing set for doubling resolution. You will prompt the terminal:

```console
cd V-V-for-AI #move to the repo file if your terminal is not there yet
python3 main.py --corruption_name resolution_double --corrupt_train False --corrupt_test True
```

After the corruption script is run, a new directory shall be created with a directory name related to the corruption name used. This new directory will 
split the images and annotations in train, test and validation for the purpose of training and inference of the detection algorithm. Additionally the corruption script
will convert the annotation stiles from XML to txt files ready to be read by the yolov5 algorithm.

### Training Script: 
-The training script indicates to the yolov5 algorithm where the images are and the training parameters to be used. The latter could be read in: 
https://docs.ultralytics.com/yolov5/train_custom_data/#13-prepare-dataset-for-yolov5. For the sake of this experiment the training default parameters:

* --img 640
* --cfg yolov5s.yaml
* --hyp hyp.no-augmentation.yaml
* --batch 16
* --epochs 100
* --weights yolov5s.pt
* --workers 24

So for our previous case( doubling resolution and only corrupting the test data), what we would pass to the terminal is the following: 
```console

python3 train.py --corruption_name resolution_double
                 --corrupt_train False 
                 --corrupt_test True 
                 --img 640
                 --cfg yolov5s.yaml
                 --hyp hyp.no-augmentation.yaml
                 --batch 16
                 --epochs 100
                 --weights yolov5s.pt
                 --workers 24
```

After the training script was used, we would have trained an instance of the yolov5 algorithm. Yolov5 automatically creates a report of the training process
and also stores the trained model in the runs file inside the yolov5 directory. The training report is stored in ../yolov5/runs/train/composed_name_of_the_experiment.
This very directory will be useful for the inference part.

### Testing/Inference Script

Once the network is trained, we have to test its performance (behavior) and we will use the testing script for this. Similarly, yolov5 already 
has a script for this that we will use. It could be read on: https://github.com/ultralytics/yolov5/blob/master/val.py The parameters that we will manipulate are:

* --img 640 (it could be more or less given the pixels in our test data).
* --task test
* --save_txt True

Additionally we will have to provide the location of the data and the weights used for the inference but this is a consequence of the previous steps. So in the same example as before,
we would pass to the console:

```console

python3 test.py --corruption_name resolution_double
                 --corrupt_train False 
                 --corrupt_test True 
                 --img 640
                 --task test
                 --save_txt True
                 --weights runs/train/name_for_the_experiment/weights/best.pt
```

In this step, if we skip the weights parameter the baseline algorithm (no corruption on training data) will be used. This means that we shall only provide the weights parameter
if we have manipulated the training dataset and we want to use this model for future inference. In the context of the inteligent systems we would be thinking of on-line training for this type of application.









