from torchmetrics import Accuracy

# Initialize metric
accuracy = Accuracy()
uba = accuracy(3, 4)
