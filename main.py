import torch
import torch.nn as nn
import torch.optim as optim
from dataset import load_data
from model import initialize_model
from train import train_model, visualize_model, log_val_predictions

def main():
    csv_file = 'Dina_HER2_dataset/dina_labels.csv'
    img_dir = 'Dina_HER2_dataset'
    num_classes = 4
    batch_size = 32
    num_epochs = 25
    checkpoint_path = 'best_checkpoint.pth'
    predictions_csv = 'val_predictions.csv'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load data
    dataloaders, dataset_sizes, class_names = load_data(csv_file, img_dir, batch_size)

    # Initialize model
    model = initialize_model(num_classes)
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Train the model
    model = train_model(model, dataloaders, dataset_sizes, criterion, optimizer, device, num_epochs, checkpoint_path)

    # Visualize the model
    visualize_model(model, dataloaders, class_names, device)

    # Log validation predictions
    log_val_predictions(model, dataloaders, class_names, device, predictions_csv)

if __name__ == '__main__':
    main()
