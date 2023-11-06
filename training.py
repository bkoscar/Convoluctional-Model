# from torch.optim import Adam
# from data_preprocessing import DataSet
# from model import SimpleCNN
# import torch.nn as nn
# import torch
# import numpy as np
# import torchvision
# import matplotlib.pyplot as plt

# model = SimpleCNN()
# # Define the loss function with Classification Cross-Entropy loss and an optimizer with Adam optimizer
# loss_fn = nn.CrossEntropyLoss()
# optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# # Function to save the model
# def saveModel():
#     path = "./model/model.pth"
#     torch.save(model.state_dict(), path)


# # Function to test the model with the test dataset and print the accuracy for the test images
# def testAccuracy():
#     model.eval()
#     accuracy = 0.0
#     total = 0.0
#     with torch.no_grad():
#         for data in test_loader:
#             images, labels = data
#             images = images.to(device)  # Mover los datos de entrada a la GPU
#             labels = labels.to(device)  # Mover las etiquetas a la GPU
#             # run the model on the test set to predict labels
#             outputs = model(images)
#             # the label with the highest energy will be our prediction
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             accuracy += (predicted == labels).sum().item()
#     # compute the accuracy over all test images
#     accuracy = (100 * accuracy / total)
#     return accuracy


# # Training function. We simply have to loop over our data iterator and feed the inputs to the network and optimize.
# def train(num_epochs):
#     best_accuracy = 0.0
#     # Define your execution device
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     print("The model will be running on", device, "device")
#     # Convert model parameters and buffers to CPU or Cuda
#     model.to(device)
#     for epoch in range(num_epochs):  # loop over the dataset multiple times
#         running_loss = 0.0
#         running_acc = 0.0
#         for i, (images, labels) in enumerate(train_loader, 0):
#             # get the inputs
#             images = images.to(device)
#             labels = labels.to(device)
#             # zero the parameter gradients
#             optimizer.zero_grad()
#             # predict classes using images from the training set
#             outputs = model(images)
#             # compute the loss based on model output and real labels
#             loss = loss_fn(outputs, labels)
#             # backpropagate the loss
#             loss.backward()
#             # adjust parameters based on the calculated gradients
#             optimizer.step()

#             # Let's print statistics for every 1,000 images
#             running_loss += loss.item()     # extract the loss value
#             if i % 1000 == 999:    
#                 # print every 1000 (twice per epoch) 
#                 print('[%d, %5d] loss: %.3f' %
#                       (epoch + 1, i + 1, running_loss / 1000))
#                 # zero the loss
#                 running_loss = 0.0
#         # Compute and print the average accuracy fo this epoch when tested over all 10000 test images
#         accuracy = testAccuracy()
#         print('For epoch', epoch+1,'the test accuracy over the whole test set is %d %%' % (accuracy))
#         # we want to save the model if the accuracy is the best
#         if accuracy > best_accuracy:
#             saveModel()
#             best_accuracy = accuracy
            

from torch.optim import Adam
from data_preprocessing import DataSet
from model import SimpleCNN
import torch.nn as nn
import torch
import numpy as np
import torchvision
import matplotlib.pyplot as plt

model = SimpleCNN()
# Define the loss function with Classification Cross-Entropy loss and an optimizer with Adam optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Function to save the model
def saveModel():
    path = "./model/model1.pth"
    torch.save(model.state_dict(), path)

# Function to test the model with the test dataset and print the accuracy for the test images
def testAccuracy():
    model.eval()
    accuracy = 0.0
    total = 0.0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.to(device)  # Move the input data to the GPU
            labels = labels.to(device)  # Move the labels to the GPU
            # Run the model on the test set to predict labels
            outputs = model(images)
            # The label with the highest energy will be our prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            accuracy += (predicted == labels).sum().item()
    # Compute the accuracy over all test images
    accuracy = (100 * accuracy / total)
    return accuracy

# Training function. We simply have to loop over our data iterator and feed the inputs to the network and optimize.
def train(num_epochs):
    best_accuracy = 0.0
    # Define your execution device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("The model will be running on", device, "device")
    # Convert model parameters and buffers to CPU or Cuda
    model.to(device)
    for epoch in range(num_epochs):  # Loop over the dataset multiple times
        running_loss = 0.0
        running_acc = 0.0
        for i, (images, labels) in enumerate(train_loader, 0):
            # Get the inputs
            images = images.to(device)
            labels = labels.to(device)
            # Zero the parameter gradients
            optimizer.zero_grad()
            # Predict classes using images from the training set
            outputs = model(images)
            # Compute the loss based on model output and real labels
            loss = loss_fn(outputs, labels)
            # Backpropagate the loss
            loss.backward()
            # Adjust parameters based on the calculated gradients
            optimizer.step()

            # Print statistics for every 1,000 images
            running_loss += loss.item()  # Extract the loss value
            if i % 1000 == 999:
                # Print every 1000 (twice per epoch)
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 1000))
                running_loss = 0.0  # Zero the loss

        # Compute and print the average accuracy for this epoch when tested over all test images
        accuracy = testAccuracy()
        print('For epoch', epoch+1, 'the test accuracy over the whole test set is %d %%' % (accuracy))
        # We want to save the model if the accuracy is the best
        if accuracy > best_accuracy:
            saveModel()
            best_accuracy = accuracy

# Function to show the images
def imageshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# Function to test the model with a batch of images and show the labels predictions
def testBatch(batch_size):
    # get batch of images from the test DataLoader  
    images, labels = next(iter(test_loader))
    print(labels)
    print(images)

    # show all images as one image grid
    imageshow(torchvision.utils.make_grid(images))
   
    # Show the real labels on the screen 
    print('Real labels: ', ' '.join('%5s' % classes[labels[j]] 
                               for j in range(batch_size)))
  
    # Let's see what if the model identifiers the  labels of those example
    outputs = model(images)
    
    # We got the probability for every 10 labels. The highest (max) probability should be correct label
    _, predicted = torch.max(outputs, 1)
    
    # Let's show the predicted labels on the screen to compare with the real ones
    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] 
                              for j in range(batch_size)))



if __name__ == '__main__':
        # Establecer semilla para NumPy
    np.random.seed(42)

    # Establecer semilla para PyTorch
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    classes = ('cat', 'rabbit')
    dataset = DataSet(root_dir='/media/oscar/Seagate/machine_learning_engineer/CNNs/data/')
    train_loader, test_loader = dataset.get_loader()

    # Let's build our model
    train(num_epochs=5)
    print('Finished Training')

    # Test which classes performed well
    testAccuracy()

    # Let's load the model we just created and test the accuracy per label
    model = SimpleCNN()
    path = "./model/model1.pth"
    model.load_state_dict(torch.load(path))

    # Test with batch of images
    testBatch(4)
