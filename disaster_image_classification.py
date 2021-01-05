from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import shutil
import piexif

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def prepare_dataset(dataset_name):
  def copy_files(file_name, dataset_name):
    # Copy Datasets to respective folders
    if file_name.endswith("train"):
      sub_folder = "train"
    elif file_name.endswith("dev"):
      sub_folder = "val"
    elif file_name.endswith("test"):
      sub_folder = "test"
    
    with open(os.path.join("Datasets", file_name), 'r') as fp:
      for record in fp.readlines():
        file, severity = record.strip().split()
        file = file.split('/')[-1]
        
        if not os.path.exists("Datasets/"+dataset_name+"/"+file):
          continue

        if int(severity) == 1 or int(severity) == 2:
          src_img = os.path.join("Datasets", dataset_name, file)
          piexif.remove(src_img)
          shutil.copy(src_img, os.path.join(dataset_path, sub_folder, "damage"))
        elif int(severity) == 0:
          src_img = os.path.join("Datasets", dataset_name, file)
          piexif.remove(src_img)
          shutil.copy(src_img, os.path.join(dataset_path, sub_folder, "nodamage"))

  filename = dataset_name.split("_")[0]
  base_ds_path = os.path.join("Datasets", dataset_name)
  dataset_path = os.path.join("data", dataset_name)

  train_filename = filename + ".train"
  val_filename = filename + ".dev"
  test_filename = filename + ".test"
  
  if not os.path.exists(dataset_path):
    os.mkdir(dataset_path)
    os.mkdir(os.path.join(dataset_path, "train"))
    os.mkdir(os.path.join(dataset_path, "train", "damage"))
    os.mkdir(os.path.join(dataset_path, "train", "nodamage"))

    os.mkdir(os.path.join(dataset_path, "val"))
    os.mkdir(os.path.join(dataset_path, "val", "damage"))
    os.mkdir(os.path.join(dataset_path, "val", "nodamage"))

    os.mkdir(os.path.join(dataset_path, "test"))
    os.mkdir(os.path.join(dataset_path, "test", "damage"))
    os.mkdir(os.path.join(dataset_path, "test", "nodamage"))

    copy_files(train_filename, dataset_name)
    copy_files(val_filename, dataset_name)
    copy_files(test_filename, dataset_name)

all_datasets = ["nepal_eq", "ecuador_eq", "ruby_typhoon", "matthew_hurricane"]
for ds in all_datasets:
  prepare_dataset(ds)

"""### Transforming data and creating dataloaders"""

# Data augmentation and normalization for training
def gen_dataloader(input_size, data_dir, batch_size):
  data_transforms = {
      'train': transforms.Compose([
          transforms.RandomResizedCrop(input_size),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ]),
      'val': transforms.Compose([
          transforms.Resize(input_size),
          transforms.CenterCrop(input_size),
          transforms.ToTensor(),
          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ]),

      'test': transforms.Compose([
          transforms.Resize(input_size),
          transforms.CenterCrop(input_size),
          transforms.ToTensor(),
          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ]),
  }

  # Create training and validation datasets
  image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val', 'test']}
  # Create training and validation dataloaders
  dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val', 'test']}

  return dataloaders_dict

"""# Defining functions for Training and Evaluating the models

### Train Model

This method trains the given model. It also prints the training loss and validation loss along with option to either plot or not plot the model's loss and accuracy on training and validation datasets
"""

def train_model(model, model_name, dataloaders, criterion, optimizer, num_epochs=25, plot_model=True):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    epochs_list = []

    train_loss_list = []
    val_loss_list = []

    train_acc_list = []
    val_acc_list = []

    for epoch in range(num_epochs):
        if (epoch % 10 == 0) or (epoch == num_epochs - 1):
          print()
          print('Epoch {}/{}'.format(epoch, num_epochs - 1))
          print('-' * 10)
        # Each epoch has a training and validation phase
        epochs_list.append(epoch)
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            if phase == 'train':
              train_loss_list.append(epoch_loss)
              train_acc_list.append(epoch_acc)
            elif phase == 'val':
              val_loss_list.append(epoch_loss)
              val_acc_list.append(epoch_acc)
            if (epoch % 10 == 0) or (epoch == num_epochs - 1):
              print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
          

    time_elapsed = time.time() - since
    print()
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    if plot_model == True:
      plot_model_training(epochs_list, train_loss_list, val_loss_list, train_acc_list, val_acc_list, model_name)

    # load best model weights
    model.load_state_dict(best_model_wts)
    
    return model

"""### Initialize model for training"""

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    model_ft = None
    if model_name == "vgg":
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft

# Helper function to plot the loss/accuracy graph for train/val datasets
def plot_model_training(epochs_list, train_loss_list, val_loss_list, train_acc_list, val_acc_list, model_name):
  # Training/Validation Loss
  plt.xlabel("Epochs")
  plt.ylabel("Loss")
  plt.plot(epochs_list, train_loss_list, label="Training Loss")
  plt.plot(epochs_list, val_loss_list, label="Validation Loss")
  plt.legend()
  plt.savefig(model_name + "_loss.jpg")
  plt.show()
  plt.clf()
  plt.cla()
  plt.close()
  
  # Training/Validation Accuracy
  plt.xlabel("Epochs")
  plt.ylabel("Accuracy")
  plt.plot(epochs_list, train_acc_list, label="Training Accuracy")
  plt.plot(epochs_list, val_acc_list, label="Validation Accuracy")
  plt.legend()
  plt.savefig(model_name + "_acc.jpg")
  plt.show()
  plt.clf()
  plt.cla()
  plt.close()
  
def evaluate_model(model, dataloaders, model_name):  
  # Iterate over data.
  tp, fp, tn, fn = [0, 0, 0, 0]
  for inputs, labels in dataloaders['test']:
    inputs = inputs.to(device)
    labels = labels.to(device)

    with torch.no_grad():
      outputs = model(inputs)
      _, preds = torch.max(outputs, 1)

    predictions = preds.tolist()
    ground_truth = labels.data.tolist()

    # statistics
    for p, t in zip(predictions, ground_truth):
      if (p == 1) and (t == 1):
        tp += 1
      elif (p == 0) and (t == 1):
        fn += 1
      elif (p == 1) and (t == 0):
        fp += 1
      elif (p == 0) and (t == 0):
        tn += 1

  # Print the model performance
  print("*" * 10 + "MODEL: " + model_name + " " + "*" * 10)
  print("accuracy: ", round((tp + tn) / (tp + fp + fn + tn), 3))
  precision = tp / (tp + fp)
  recall = tp / (tp + fn)
  print("precision: ", round(precision, 3))
  print("recall: ", round(recall, 3))
  print("f1-score: ", round(2 * recall * precision / (recall + precision), 3))
  print("*" * 30)

def get_parms_to_optimize(model_ft):
  # Send the model to GPU
  model_ft = model_ft.to(device)
  params_to_update = model_ft.parameters()
  print("Params to learn:")
  if feature_extract:
      params_to_update = []
      for name,param in model_ft.named_parameters():
          if param.requires_grad == True:
              params_to_update.append(param)
              print("\t",name)
  else:
      for name,param in model_ft.named_parameters():
          if param.requires_grad == True:
              print("\t",name)

  return optim.SGD(params_to_update, lr=0.001, momentum=0.9)

"""# Experimentation with Models for Different Datasets"""

# Parameters that are same for all models
num_classes = 2
batch_size = 32
num_epochs = 25
model_name = "vgg"
input_size = 224 # Size of input image for VGG 

# Setup the loss function
criterion = nn.CrossEntropyLoss()

"""### Matthew Hurricane"""

# # Top level data directory. Here we assume the format of the directory conforms
data_dir = "./data/matthew_hurricane"

# # Flag for feature extracting. When False, we finetune the whole model,
# #   when True we only update the reshaped layer params
# feature_extract = True

# # Initialize the model for this run
# model_ft = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
# dataloaders_dict = gen_dataloader(input_size, data_dir, batch_size)
# optimizer_ft = get_parms_to_optimize(model_ft)
# # Train and evaluate

# model_matthew_last = train_model(model_ft, "matthew_last", dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs)

# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = False

# Initialize the model for this run
model_ft = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
dataloaders_dict = gen_dataloader(input_size, data_dir, batch_size)
optimizer_ft = get_parms_to_optimize(model_ft)
# Train and evaluate
model_matthew_all = train_model(model_ft, "matthew_all", dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs)

# """### Ruby Typhoon"""

# # Top level data directory. Here we assume the format of the directory conforms
data_dir = "./data/ruby_typhoon"

# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = True

# Initialize the model for this run
model_ft = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
dataloaders_dict = gen_dataloader(input_size, data_dir, batch_size)
optimizer_ft = get_parms_to_optimize(model_ft)
# Train and evaluate
model_ruby_last = train_model(model_ft, "ruby_last", dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs)

# # Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = False

# Initialize the model for this run
model_ft = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
dataloaders_dict = gen_dataloader(input_size, data_dir, batch_size)
optimizer_ft = get_parms_to_optimize(model_ft)
# Train and evaluate
model_ruby_all = train_model(model_ft, "ruby_all", dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs)

# """### Ecquador Earthquake"""

# # Top level data directory. Here we assume the format of the directory conforms
data_dir = "./data/ecuador_eq"

# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = True

# Initialize the model for this run
model_ft = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
dataloaders_dict = gen_dataloader(input_size, data_dir, batch_size)
optimizer_ft = get_parms_to_optimize(model_ft)
# Train and evaluate
model_ecquador_last = train_model(model_ft, "ecquador_last", dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs)

# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = False

# Initialize the model for this run
model_ft = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
dataloaders_dict = gen_dataloader(input_size, data_dir, batch_size)
optimizer_ft = get_parms_to_optimize(model_ft)
# Train and evaluate
model_ecquador_all = train_model(model_ft, "ecquador_all", dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs)

# """### Nepal Earthquake"""

# Top level data directory. Here we assume the format of the directory conforms
data_dir = "./data/nepal_eq"

# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = True

# Initialize the model for this run
model_ft = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
dataloaders_dict = gen_dataloader(input_size, data_dir, batch_size)
optimizer_ft = get_parms_to_optimize(model_ft)
# Train and evaluate
model_nepal_last = train_model(model_ft, "nepal_last", dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs)

# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = False

# Initialize the model for this run
model_ft = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
dataloaders_dict = gen_dataloader(input_size, data_dir, batch_size)
optimizer_ft = get_parms_to_optimize(model_ft)
# Train and evaluate
model_nepal_all = train_model(model_ft, "nepal_all", dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs)

"""# Results in terms of Performance vs Data Size vs Number of Layers fine-tuned"""

# Training datasets in each category
matthew = 357
ruby = 500
ecquador = 1368
nepal = 11463

txt = ["  Matthew", "  Ruby", "  Ecquador", "  Nepal"]
max_val_acc = [85.7, 90.2, 94.4, 97.6] # Arranged according to txt
y = max_val_acc
x = [matthew, ruby, ecquador, nepal]

plt.scatter(x, max_val_acc)
plt.ylabel("Accuracy")
plt.xlabel("Datasets Count")
plt.title("Accuracy vs Dataset Size")
i = 0
for x, y in zip(x, y):
  plt.annotate(txt[i] + "(Datasize: " + str(x) + ")", (x, y))
  i += 1

plt.savefig("acc_vs_datasize.jpg")
plt.show()


"""Accuracy vs Dataset Size:

Performance vs Number of layers fine-tuned:

# Evolution of Loss and Accuracy for both training and validation sets (for two comparatively good and bad model)
"""


num_epochs = 100


## RELATIVELY BAD MODEL
# Top level data directory. Here we assume the format of the directory conforms
data_dir = "./data/matthew_hurricane"

# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = True

# Initialize the model for this run
model_ft = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
dataloaders_dict = gen_dataloader(input_size, data_dir, batch_size)
optimizer_ft = get_parms_to_optimize(model_ft)
# Train and evaluate

model_matthew_all = train_model(model_ft, "matthew_all_bad", dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs)



## RELATIVELY GOOD MODEL
# Top level data directory. Here we assume the format of the directory conforms
data_dir = "./data/ecuador_eq"

# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = False

# Initialize the model for this run
model_ft = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
dataloaders_dict = gen_dataloader(input_size, data_dir, batch_size)
optimizer_ft = get_parms_to_optimize(model_ft)
# Train and evaluate
model_ecquador_all = train_model(model_ft, "ecquador_all_good", dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs)


# # ## RELATIVELY GOOD MODEL
# Top level data directory. Here we assume the format of the directory conforms
data_dir = "./data/ruby_typhoon"

# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = False

# Initialize the model for this run
model_ft = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
dataloaders_dict = gen_dataloader(input_size, data_dir, batch_size)
optimizer_ft = get_parms_to_optimize(model_ft)
# Train and evaluate
model_ruby_all = train_model(model_ft, "ruby_all_good", dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs)




# """# Evaluation of Models

# Use of Best models to evaluate on the test dataset

#### Matthew Hurricane


# Top level data directory. Here we assume the format of the directory conforms
data_dir = "./data/matthew_hurricane"

# Initialize the model for this run
dataloaders_dict = gen_dataloader(input_size, data_dir, batch_size)
evaluate_model(model_matthew_all, dataloaders_dict, "matthew_hurricane_all")


# """#### Ruby Typhoon"""

# Top level data directory. Here we assume the format of the directory conforms
data_dir = "./data/ruby_typhoon"

# Initialize the model for this run
dataloaders_dict = gen_dataloader(input_size, data_dir, batch_size)

evaluate_model(model_ruby_all, dataloaders_dict, "ruby_typhoon_all")

"""#### Ecquador Earthquake"""

# Top level data directory. Here we assume the format of the directory conforms
data_dir = "./data/ecuador_eq"

# Initialize the model for this run
dataloaders_dict = gen_dataloader(input_size, data_dir, batch_size)
evaluate_model(model_ecquador_all, dataloaders_dict, "ecquador_earthquake_all")

"""#### Nepal Earthquake"""

# Top level data directory. Here we assume the format of the directory conforms
data_dir = "./data/nepal_eq"

# Initialize the model for this run
dataloaders_dict = gen_dataloader(input_size, data_dir, batch_size)
evaluate_model(model_nepal_all, dataloaders_dict, "nepal_earthquake_all")

