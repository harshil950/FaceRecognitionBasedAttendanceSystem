from torch import nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
from mlxtend.plotting import plot_confusion_matrix
from copy import deepcopy
import mlflow
import mlflow.pytorch
import torch

is_cuda = torch.cuda.is_available()

if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

data_dir = 'detected'
dataset = datasets.ImageFolder(data_dir, transform)

mapping = dataset.class_to_idx
classes = dataset.classes
print(mapping)
print(classes, len(classes))
num_classes = len(classes)

train_set, valid_set, test_set = random_split(dataset, [0.8, 0.1, 0.1])
print(len(train_set), len(valid_set), len(test_set))


class Vgg_face(nn.Module):

    def __init__(self):
        super(Vgg_face, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)
        self.fc6 = nn.Linear(in_features=25088, out_features=4096, bias=True)
        self.relu6 = nn.ReLU(inplace=True)
        self.dropout6 = nn.Dropout(p=0.5)
        self.fc7 = nn.Linear(in_features=4096, out_features=4096, bias=True)
        self.relu7 = nn.ReLU(inplace=True)
        self.dropout7 = nn.Dropout(p=0.5)
        self.fc8 = nn.Linear(in_features=4096, out_features=2622, bias=True)

    def forward(self, x0):
        x1 = self.conv1_1(x0)
        x2 = self.relu1_1(x1)
        x3 = self.conv1_2(x2)
        x4 = self.relu1_2(x3)
        x5 = self.pool1(x4)
        x6 = self.conv2_1(x5)
        x7 = self.relu2_1(x6)
        x8 = self.conv2_2(x7)
        x9 = self.relu2_2(x8)
        x10 = self.pool2(x9)
        x11 = self.conv3_1(x10)
        x12 = self.relu3_1(x11)
        x13 = self.conv3_2(x12)
        x14 = self.relu3_2(x13)
        x15 = self.conv3_3(x14)
        x16 = self.relu3_3(x15)
        x17 = self.pool3(x16)
        x18 = self.conv4_1(x17)
        x19 = self.relu4_1(x18)
        x20 = self.conv4_2(x19)
        x21 = self.relu4_2(x20)
        x22 = self.conv4_3(x21)
        x23 = self.relu4_3(x22)
        x24 = self.pool4(x23)
        x25 = self.conv5_1(x24)
        x26 = self.relu5_1(x25)
        x27 = self.conv5_2(x26)
        x28 = self.relu5_2(x27)
        x29 = self.conv5_3(x28)
        x30 = self.relu5_3(x29)
        x31_preflatten = self.pool5(x30)
        x31 = x31_preflatten.view(x31_preflatten.size(0), -1)
        x32 = self.fc6(x31)
        x33 = self.relu6(x32)
        x34 = self.dropout6(x33)
        x35 = self.fc7(x34)
        x36 = self.relu7(x35)
        x37 = self.dropout7(x36)
        x38 = self.fc8(x37)
        return x38

def load_vgg_face_weights(weights_path=None, **kwargs):
    """
    load imported model instance

    Args:
        weights_path (str): If set, loads model weights from the given path
    """
    model = Vgg_face()
    if weights_path:
        state_dict = torch.load(weights_path)
        model.load_state_dict(state_dict)
    return model

filename = "vgg_face.pth"

vgg_model = load_vgg_face_weights(weights_path=filename)

for param in vgg_model.parameters():
    param.requires_grad = False

final_in_features = vgg_model.fc8.in_features

vgg_model.fc8 = nn.Linear(final_in_features, num_classes)
vgg_model = vgg_model.to(device)

for param in vgg_model.parameters():
    if param.requires_grad:
        print(param.shape)

class Params(object):
    def __init__(self, batch_size, epochs):
        self.batch_size = batch_size
        self.epochs = epochs

args = Params(32, 30)

batch_size = args.batch_size

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, drop_last=True)

dataiter = iter(train_loader)
images, labels = next(dataiter)

print(images.shape)

print(images[1].shape)
print(labels.shape)

output = vgg_model(images.to(device))
print(output.shape)

def training_and_validation(model, epochs, lr=0.01):

    epoch_tr_loss,epoch_vl_loss = [],[]
    epoch_tr_acc,epoch_vl_acc = [],[]

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in tqdm(range(epochs)):
        train_losses = []
        train_acc = 0.0
        model.train()
        total_train = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            model.zero_grad()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            train_losses.append(loss.item())
            accuracy = (torch.max(outputs.data, 1)[1] == labels).sum().item()
            train_acc += accuracy
            total_train += labels.size(0)
            optimizer.step()

        val_losses = []
        val_acc = 0.0    
        model.eval()
        total_test = 0
        y_pred, y_valid = [], []
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                
                _, predicted = torch.max(outputs.data, 1)
                y_pred += predicted
                y_valid += labels

                val_loss = criterion(outputs, labels)
                val_losses.append(val_loss.item())

                accuracy = (predicted == labels).sum().item()
                val_acc += accuracy
                total_test += labels.size(0)

        epoch_train_loss = np.mean(train_losses)
        epoch_val_loss = np.mean(val_losses)
        epoch_train_acc = train_acc/total_train
        epoch_val_acc = val_acc/total_test
        epoch_tr_loss.append(epoch_train_loss)
        epoch_vl_loss.append(epoch_val_loss)
        epoch_tr_acc.append(epoch_train_acc)
        epoch_vl_acc.append(epoch_val_acc)
        mlflow.log_metric('train_loss', epoch_train_loss)
        mlflow.log_metric('valid_loss', epoch_val_loss)
        mlflow.log_metric('train_accuracy', epoch_train_acc)
        mlflow.log_metric('valid_accuracy', epoch_val_acc)
#         print(f'Epoch {epoch+1}') 
#         print(f'train_loss : {epoch_train_loss} val_loss : {epoch_val_loss}')
#         print(f'train_accuracy : {epoch_train_acc*100} val_accuracy : {epoch_val_acc*100}')
#         print(25*'==')

        if epoch == epochs-1:
            cm = confusion_matrix(torch.tensor(y_valid).cpu(), torch.tensor(y_pred).cpu())
            fig, ax = plot_confusion_matrix(conf_mat=cm, class_names=classes)
            cm_image_dir = "cm_images"
            if not os.path.exists(cm_image_dir):
                os.mkdir(cm_image_dir)
            image_path = '%s/%s.png' % (cm_image_dir, expt_id)
            plt.savefig(image_path)
            mlflow.log_artifact(image_path)
  
    fig = plt.figure(figsize = (20, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epoch_tr_acc, label='Train Acc')
    plt.plot(epoch_vl_acc, label='Validation Acc')
    plt.title("Accuracy")
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(epoch_tr_loss, label='Train loss')
    plt.plot(epoch_vl_loss, label='Validation loss')
    plt.title("Loss")
    plt.legend()
    plt.grid()

    # plt.show()


mlflow.set_tracking_uri("http://localhost:32000")
import logging
logging.getLogger("mlflow").setLevel(logging.DEBUG)


mlflow.create_experiment('Vgg face 100 exp1')
mlflow.set_experiment('Vgg face 100 exp1')

for lr in [0.0005, 0.001, 0.003, 0.005, 0.007]:
    expt_id = '%d' % (int(lr)*1000)
    print('\nLR = %f\n' % (lr))
    model = deepcopy(vgg_model)

    with mlflow.start_run() as run:
        for key, value in vars(args).items():
            mlflow.log_param(key, value)
        mlflow.log_param('lr', lr)
        training_and_validation(model, args.epochs, lr)
        mlflow.pytorch.log_model(model, "models")


# model_load = mlflow.pytorch.load_model("mlartifacts/{}/{}/artifacts/models".format("354716052899237312", "397d9dfd3db84e4791b8cd3f1c554a90"))

# def evaluation(dataloader, model):
#     y_valid, y_pred = [], []
#     for data in tqdm(dataloader):
#         inputs, labels = data
#         inputs, labels = inputs.to(device), labels.to(device)
#         outputs = model(inputs)
#         _, predicted = torch.max(outputs.data, 1)
#         y_pred += predicted
#         y_valid += labels
#     print(classification_report(y_valid, y_pred))
#     cm = confusion_matrix(y_valid, y_pred)
#     fig, ax = plot_confusion_matrix(conf_mat=cm, class_names=classes)
#     plt.show()

# evaluation(test_loader, model_load)
