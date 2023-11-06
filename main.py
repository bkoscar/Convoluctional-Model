import numpy as np
import torch
from data_preprocessing import DataSet
from model import SimpleCNN
from training import testAccuracy, train, testBatch

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
train(num_epochs=5, train_loader=train_loader)
print('Finished Training')

# Test which classes performed well
testModelAccuracy()

# Let's load the model we just created and test the accuracy per label
model = SimpleCNN()
path = "./model/model.pth"
model.load_state_dict(torch.load(path))

# Test with batch of images
testBatch(test_loader=test_loader)
