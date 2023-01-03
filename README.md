# MachineLearning_Classification

Educational project to train several CNN neural networks on the dataset "Scene Classification: Simulation to Reality" 
(https://www.kaggle.com/datasets/birdy654/environment-recognition-simulation-to-reality)


## Dataset

<img
  src="/data/dataset_class_counts.png"
  alt="pic1"
  title="class counts"
  style="display: inline-block; margin: 0 auto; max-width: 300px">
  
<img
  src="/data/dataset_ratio_counts.png"
  alt="pic2"
  title="ratio counts"
  style="display: inline-block; margin: 0 auto; max-width: 300px">


## Train

Comparison diffrent models in training.

<img
  src="/train_task/tensorboard_results/example_different_networks.jpg"
  alt="pic3"
  title="CNN train curves"
  style="display: inline-block; margin: 0 auto; max-width: 300px">

Comparison diffrent parts of dataset used for training.

<img
  src="/train_task/tensorboard_results/example_cross_validation.jpg"
  alt="pic4"
  title="cross validation"
  style="display: inline-block; margin: 0 auto; max-width: 300px">
  
 Example of training and validation curve during training proces.
  
<img
  src="/train_task/tensorboard_results/example_train_valid_curves.jpg"
  alt="pic5"
  title="train valid curves"
  style="display: inline-block; margin: 0 auto; max-width: 300px">


## Eval

Evaluation one of the best model. Confussion matrix for test dataset.

<img
  src="/eval/results/conf_matrix.png"
  alt="pic6"
  title="Conf matrix"
  style="display: inline-block; margin: 0 auto; max-width: 300px">


## Libraries used:
```
pytorch
torchvision
numpy
albumentations
pandas
seaborn
matplotlib
```
