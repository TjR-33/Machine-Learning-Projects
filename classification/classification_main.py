# Import necessary libraries
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
from torcheval.metrics.functional import binary_f1_score, binary_accuracy
import os
from PIL import Image
from torchvision.transforms import v2 as transforms
import matplotlib.pyplot as plt
import time

# Import custom modules
from classification_dataset import PlantDataset 
from classification_model import PlantHealthClassifier, PlantHealthClassifier2, EnsembleModel, _train_model_with_early_stopping

# Device Agnostic Code
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}\n")

# Hyperparameters
BATCH_SIZE = 32
n=1 # Early Stopping parameter: number of steps between evaluations
p=7 # Early Stopping parameter: Patience
EPOCHS = 100
NUM_BOOTSTRAPS = 5
OPTIMIZER = lambda params, lr: optim.Adam(params=params, lr=lr)
LEARNING_RATE = 0.01
SCHEDULER = optim.lr_scheduler.LambdaLR
LAMBDA_LR = lambda epoch: 0.90 ** epoch
LOSS_FN = nn.BCEWithLogitsLoss()


# Set random seed for reproducibility
RANDOM_SEED = 42

# Set paths for data
PATH = "D:/IISc/Sem6-TejusRohatgi/DS 216 Machine Learning for Data Science/assignment/2/mlds-assignment-2"
train_root_dir = PATH + '/train/images_without_opacity'
test_root_dir = PATH + '/test/images_without_opacity'
CHANNELS=3

# save_dir = "ensemble_model_trial"
# save_dir = "ensemble_model_2"
# save_dir = "ensemble_model_3"
# save_dir = "ensemble_model_4"
# save_dir = "ensemble_model_5"
save_dir = "ensemble_model_7"
save_training_results = True


# Data Preparation
print("Loading Data\n")
os.chdir(PATH)

# Initialize lists to store image names and labels
train_image_list = []
train_binary_pred_list = []

if os.path.exists('train'):
    print("train folder exists")
    os.chdir('train')

    with open('train.csv', 'r') as f:
        f.readline()
        lines = f.read().splitlines()
        for i, line in enumerate(lines):
                        image_name, binary_pred, temp = line.split(',')
                        train_image_list.append(image_name)
                        train_binary_pred_list.append(int(binary_pred))
else:
    raise FileNotFoundError("train folder does not exist")

os.chdir(PATH)

# Initialize lists to store test image names and labels
test_image_list = []
test_binary_pred_list = []

# Check if test directory exists and load data
if os.path.exists('test'):
    print("test folder exists")
    os.chdir('test')

    with open('test.csv', 'r') as f:
        f.readline()
        lines = f.read().splitlines()
        for i, line in enumerate(lines):
                        image_name, binary_pred, temp = line.split(',')
                        test_image_list.append(image_name)
                        test_binary_pred_list.append(binary_pred)
else:
    raise FileNotFoundError("test folder does not exist")

os.chdir(PATH)

# Split the training data into training and validation sets
from sklearn.model_selection import train_test_split
train_image_list, val_image_list, train_binary_pred_list, val_binary_pred_list = train_test_split(train_image_list, train_binary_pred_list, test_size=0.2, random_state=RANDOM_SEED)

# Define transformations for the training and validation sets
train_transform = transforms.Compose([
    # transforms.RandomHorizontalFlip(),  # Randomly flip the images horizontally
    # transforms.RandomVerticalFlip(),  # Randomly flip the images vertically 
    # transforms.RandomAffine(
    #     degrees=(-30, 30),  # Rotate by degrees between -30 and 30
    #     translate=(0.1, 0.1),  # Translate by a fraction of image height/width (10%)
    #     scale=(0.95, 1.05),  # Scale by 95% to 105%
    #     shear=(-10, 10, -10, 10),  # Shear by -10 to 10 degrees
    # ), 
    # transforms.RandomChannelPermutation() ,# Randomly permute the channels 
    transforms.ToTensor()])  # Convert the images to PyTorch tensors
val_transform = transforms.Compose([transforms.ToTensor()])

train_dataset = PlantDataset(train_root_dir, train_image_list, train_binary_pred_list, transform=train_transform)
val_dataset = PlantDataset(train_root_dir, val_image_list, val_binary_pred_list, transform=val_transform)

# Create datasets for training and validation
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Set seeds for reproducibility
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)

# Train the model
model = EnsembleModel(model = PlantHealthClassifier2,input_channels=CHANNELS,  device=device, weighted_voting=True)

# Training
if device == "cuda":
    torch.cuda.empty_cache()
start = time.time()

test_loss, test_f1, test_accuracy, best_epoch_list, OOB_error_list, acc_list, f1_list, epoch_metrics_list = model.bagging(
                                                  num_models=NUM_BOOTSTRAPS,
                                                  train_dataset = train_dataset, 
                                                  test_dataset = val_dataset, 
                                                  batch_size=BATCH_SIZE,
                                                  num_epochs=EPOCHS,
                                                  optimizer_factory=OPTIMIZER, lr=LEARNING_RATE,
                                                  scheduler_factory=SCHEDULER, lambda_lr=LAMBDA_LR,
                                                  n=n,p=p,
                                                  get_training_results=save_training_results,save_dir=save_dir)

end= time.time()
print(f"Training time: {end-start} seconds\n")
print(f"Test Loss: {test_loss}")
print(f"Test F1 Score: {test_f1}")
print(f"Test Accuracy: {test_accuracy}")

# Saving Model
os.chdir(PATH)
model.save(save_dir=save_dir)

# Saving training and model details

if save_training_results:
    import datetime
    date_time = datetime.datetime.now().strftime("%d-%m-%Y, %H:%M:%S")

    with open(f'{PATH}/{save_dir}/training_results.txt', 'w') as f:
        f.write(f"Training Results\n")
        f.write(f"Date: {date_time}\n")
        f.write(f"Training Time = {end-start} s\n")
        f.write(f"Test Loss: {test_loss}\n")
        f.write(f"Test F1 Score: {test_f1}\n")
        f.write(f"Test Accuracy: {test_accuracy}\n")
         

    for i, model in enumerate(model.ensemble):
        # Writing the details of the model
        best_OOB_error = OOB_error_list[i]
        best_acc = acc_list[i]
        best_f1 = f1_list[i]
        best_epoch = best_epoch_list[i]
        epoch_metrics = epoch_metrics_list[i]
        epoch_count = epoch_metrics['epoch_count']
        train_loss_values = epoch_metrics['train_loss_values']
        val_loss_values = epoch_metrics['val_loss_values']
        train_f1_values = epoch_metrics['train_f1_values']
        val_f1_values = epoch_metrics['val_f1_values']
        train_accuracy_values = epoch_metrics['train_accuracy_values']
        val_accuracy_values = epoch_metrics['val_accuracy_values']

        with open(f'{PATH}/{save_dir}/model_{i}.txt', 'w') as f:
            f.write(f"Model {i} details\n")
            f.write(f"Date: {date_time}\n")
            f.write(f"Model: {model}\n")
            f.write(f"epoch_count, train_loss_values, val_loss_values, train_f1_values, val_f1_values, train_accuracy_values, val_accuracy_values\n")
            for j in range(len(epoch_count)):
                f.write(f"{epoch_count[j]}, {train_loss_values[j]}, {val_loss_values[j]}, {train_f1_values[j]}, {val_f1_values[j]}, {train_accuracy_values[j]}, {val_accuracy_values[j]}\n")
            f.write("Early Stopping results\n")
            f.write("Epoch, OOB_error, Accuracy, F1-Score\n")
            f.write(f"{best_epoch}, {best_OOB_error}, {best_acc}, {best_f1}\n")
        print(f"Model {i} details saved successfully\n")

del model

# Load the model for prediction
ensemble_model_load = EnsembleModel(model= PlantHealthClassifier,input_channels=CHANNELS, device=device, weighted_voting=True)
ensemble_model_load.load(save_dir=save_dir)

# Make predictions on the test set
predictions = []
for i in test_image_list:
    image = Image.open(test_root_dir + '/' + i)
    convert_tensor = transforms.ToTensor()
    image_tensor = convert_tensor(image).unsqueeze(0).to(device)
    # print(image_tensor.shape)
    prediction = ensemble_model_load.predict(image_tensor)
    # save predictions to a csv -> id,binary_pred
    predictions.append(prediction)

# Save predictions to a CSV file
os.chdir(PATH+'/'+save_dir)
with open('predictions.csv', 'w') as f:
    f.write('id,binary_pred\n')
    for i, prediction in enumerate(predictions):
        f.write(f'{test_image_list[i]},{prediction.item()}\n')
print("Predictions saved successfully\n")

del ensemble_model_load



