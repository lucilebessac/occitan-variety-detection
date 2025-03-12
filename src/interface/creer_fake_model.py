# create_dummy_model.py
import torch
import os

# Define the CNN model class (same as your original)
class OccitanCNN(torch.nn.Module):
    def __init__(self, fasttext_embedding_dim=300, nb_filtres=100):
        super(OccitanCNN, self).__init__()
        # Couches convolutives
        self.conv1 = torch.nn.Conv1d(in_channels=fasttext_embedding_dim, out_channels=nb_filtres, kernel_size=2)
        self.conv2 = torch.nn.Conv1d(in_channels=fasttext_embedding_dim, out_channels=nb_filtres, kernel_size=3)
        self.conv3 = torch.nn.Conv1d(in_channels=fasttext_embedding_dim, out_channels=nb_filtres, kernel_size=4)
        # Dropout pour éviter l'overfitting
        self.dropout = torch.nn.Dropout(0.5)
        # Couche "fully connected layer" pour la classification
        self.fc = torch.nn.Linear(nb_filtres*3, 3)  # 3 car on a 3 couches, puis 3 car on a 3 classes

    def forward(self, x):
        # x est un tensor de taille (batch_size, max_len, fasttext_embedding_dim)
        x = x.permute(0, 2, 1)  # On permute les dimensions
        # On applique les convolutions + ReLu
        conv1 = torch.nn.functional.relu(self.conv1(x))
        conv2 = torch.nn.functional.relu(self.conv2(x))
        conv3 = torch.nn.functional.relu(self.conv3(x))
        # Max pooling
        pool1 = torch.nn.functional.max_pool1d(conv1, conv1.size(2)).squeeze(2)
        pool2 = torch.nn.functional.max_pool1d(conv2, conv2.size(2)).squeeze(2)
        pool3 = torch.nn.functional.max_pool1d(conv3, conv3.size(2)).squeeze(2)
        # Concaténation
        x = torch.cat((pool1, pool2, pool3), dim=1)
        # Dropout
        x = self.dropout(x)
        # Couche fully connected
        x = self.fc(x)
        # Couche Softmax
        x = torch.nn.functional.softmax(x, dim=1)
        return x

def main():
    # Create a model instance
    model = OccitanCNN()
    
    # Make sure the CNN directory exists
    os.makedirs("CNN", exist_ok=True)
    
    # Save the model
    torch.save(model.state_dict(), "CNN/model.pth")
    print("Dummy model created and saved to CNN/model.pth")
    
    # Test loading the model
    test_model = OccitanCNN()
    test_model.load_state_dict(torch.load("CNN/model.pth"))
    test_model.eval()
    print("Model loaded successfully!")
    
    # Create a dummy input tensor to test the model
    dummy_input = torch.rand(1, 56, 300)  # Batch size 1, 56 tokens, 300-dim embeddings
    with torch.no_grad():
        output = test_model(dummy_input)
    
    print("Model output shape:", output.shape)
    print("Sample output:", output)
    print("Predicted class:", torch.argmax(output, dim=1).item())

if __name__ == "__main__":
    main()