import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from helper_scenario import get_distances_between_agents


class DistancePredictor(nn.Module):
    """Neural network model for predicting distances between rectangles."""

    def __init__(self, n_features):
        """
        Initialize the DistancePredictor model.

        Args:
            n_features (int): Number of input features.
        """
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(n_features, 64),
            # nn.ReLU(),  # Not second-order differentiable
            nn.Tanh(),  # Second-order differentiable
            nn.Linear(64, 64),
            # nn.ReLU(),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.model(x)


class SafetyMarginEstimatorModule:
    """Class for estimating safety margin between two car-like robots modeled by rectangles."""

    def __init__(
        self,
        length=0.16,
        width=0.08,
        path_nn="sigmarl/assets/nn_sm_predictors/sm_predictor_tanh.pth",
    ):
        """
        Initialize the SafetyMarginEstimatorModule.

        Args:
            length (float): Length of the rectangle.
            width (float): Width of the rectangle.
        """
        self.length = length
        self.width = width
        self.n_features = 3
        self.x1 = 0.0
        self.y1 = 0.0
        self.heading1 = 0.0
        self.path_nn = path_nn

        pos_normalizer = length
        angle_normalizer = np.pi

        self.feature_normalizer = torch.tensor(
            [pos_normalizer, pos_normalizer, angle_normalizer]
        ).unsqueeze(0)
        self.label_normalizer = pos_normalizer

        # Define feature ranges
        self.x_min = -2 * self.length
        self.x_max = 2 * self.length
        self.y_min = -2 * self.length
        self.y_max = 2 * self.length
        self.heading_min = -np.pi
        self.heading_max = np.pi

    def get_rectangle_vertices(self, x, y, theta):
        """
        Compute the vertices of a rectangle given its center, orientation, length, and width.

        Args:
            x (float): X-coordinate of the rectangle center.
            y (float): Y-coordinate of the rectangle center.
            theta (float): Orientation angle of the rectangle in radians.

        Returns:
            np.ndarray: Array of rectangle vertices.
        """
        hl = self.length / 2.0
        hw = self.width / 2.0
        local_vertices = np.array([[-hl, -hw], [-hl, hw], [hl, hw], [hl, -hw]])
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        global_vertices = (R @ local_vertices.T).T + np.array([x, y])
        return global_vertices

    def generate_training_data(self):
        """
        Generate training data for the distance prediction model.

        Returns:
            tuple: Features and labels for training.
        """
        num_values = 31
        heading_values = np.linspace(self.heading_min, self.heading_max, num_values)
        position_values = np.linspace(self.x_min, self.x_max, num_values)
        X2, Y2, H2 = np.meshgrid(
            position_values, position_values, heading_values, indexing="ij"
        )
        X2, Y2, H2 = X2.flatten(), Y2.flatten(), H2.flatten()
        num_samples = len(X2)
        print(f"Number of training samples: {num_samples}")

        features = np.zeros((num_samples, self.n_features))
        labels = np.zeros(num_samples)

        for i in range(num_samples):
            x2, y2, heading2 = X2[i], Y2[i], H2[i]
            rect1_vertices = torch.tensor(
                self.get_rectangle_vertices(self.x1, self.y1, self.heading1)
            )
            rect2_vertices = torch.tensor(self.get_rectangle_vertices(x2, y2, heading2))
            vertices = torch.stack([rect1_vertices, rect2_vertices], dim=0).unsqueeze(0)
            distance = get_distances_between_agents(
                vertices, distance_type="mtv", is_set_diagonal=False
            )[0, 0, 1]
            features[i, :] = [x2, y2, heading2]
            labels[i] = distance

        features = torch.tensor(features, dtype=torch.float32) / self.feature_normalizer
        labels = (
            torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
            / self.label_normalizer
        )
        return features, labels

    def train_model(self, features, labels, k_folds=5):
        """
        Train the distance prediction model using k-fold cross-validation.

        Args:
            features (torch.Tensor): Training features.
            labels (torch.Tensor): Training labels.
            save_path (str): Path to save the trained model.
            k_folds (int): Number of folds for cross-validation.

        Returns:
            DistancePredictor: Trained model.
        """
        self.net = DistancePredictor(self.n_features)
        train_losses, val_losses = [], []

        for fold in range(k_folds):
            print(f"Fold {fold + 1}/{k_folds}")
            X_train, X_val, y_train, y_val = train_test_split(
                features, labels, test_size=1 / k_folds, random_state=fold
            )
            train_loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(X_train, y_train),
                batch_size=64,
                shuffle=True,
            )
            val_loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(X_val, y_val),
                batch_size=64,
                shuffle=False,
            )

            criterion = nn.MSELoss()
            optimizer = optim.Adam(self.net.parameters(), lr=0.001)
            num_epochs = 200

            for epoch in range(num_epochs):
                self.net.train()
                running_loss = 0.0
                for inputs, targets in train_loader:
                    optimizer.zero_grad()
                    outputs = self.net(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item() * inputs.size(0)
                epoch_loss = running_loss / len(train_loader.dataset)
                train_losses.append(epoch_loss)

                self.net.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for inputs, targets in val_loader:
                        outputs = self.net(inputs)
                        loss = criterion(outputs, targets)
                        val_loss += loss.item() * inputs.size(0)
                val_loss /= len(val_loader.dataset)
                val_losses.append(val_loss)

                print(
                    f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.6f}, Val Loss: {val_loss:.6f}"
                )

        torch.save(self.net.state_dict(), self.path_nn)
        print(f"Model saved to {self.path_nn}")
        return self.net

    def load_model(self):
        """
        Load a pre-trained model from a file.

        Args:
            load_path (str): Path to the model file.

        Returns:
            DistancePredictor: Loaded model.
        """
        self.net = DistancePredictor(self.n_features)
        self.net.load_state_dict(torch.load(self.path_nn))
        self.net.eval()
        print(f"Model loaded from {self.path_nn}")
        return self.net

    def test_model(self, net):
        """
        Test the trained model on random samples and evaluate its performance.

        Args:
            net (DistancePredictor): Trained model.
        """
        num_test_samples = 1000
        x2_test = np.random.uniform(self.x_min, self.x_max, num_test_samples)
        y2_test = np.random.uniform(self.y_min, self.y_max, num_test_samples)
        heading2_test = np.random.uniform(
            self.heading_min, self.heading_min, num_test_samples
        )

        test_features = np.column_stack((x2_test, y2_test, heading2_test))
        test_labels = np.zeros(num_test_samples)

        for i in range(num_test_samples):
            x2, y2, heading2 = x2_test[i], y2_test[i], heading2_test[i]
            rect1_vertices = torch.tensor(
                self.get_rectangle_vertices(self.x1, self.y1, self.heading1)
            )
            rect2_vertices = torch.tensor(self.get_rectangle_vertices(x2, y2, heading2))
            vertices = torch.stack([rect1_vertices, rect2_vertices], dim=0).unsqueeze(0)
            distance = get_distances_between_agents(
                vertices, distance_type="mtv", is_set_diagonal=False
            )[0, 0, 1]
            test_labels[i] = distance

        test_features = (
            torch.tensor(test_features, dtype=torch.float32) / self.feature_normalizer
        )
        test_labels = torch.tensor(test_labels, dtype=torch.float32).unsqueeze(1)

        net.eval()
        criterion = nn.MSELoss()
        with torch.no_grad():
            predicted_labels = (
                net(test_features) * self.label_normalizer
            )  # De-normalizer
            test_loss = criterion(predicted_labels, test_labels)
            print(f"Test Loss: {test_loss.item():.6f}")

        differences = (predicted_labels - test_labels).numpy()
        absolute_errors = np.abs(differences)
        mean_error = np.mean(absolute_errors)
        max_error = np.max(absolute_errors)
        print(f"Mean Absolute Error: {mean_error:.6f}")
        print(f"Max Absolute Error: {max_error:.6f}")

    def visualize_rectangles(
        self, rect1_vertices, rect2_vertices, actual_distance, predicted_distance
    ):
        """
        Visualize two rectangles and display the actual and predicted distances.

        Args:
            rect1_vertices (np.ndarray): Vertices of the first rectangle.
            rect2_vertices (np.ndarray): Vertices of the second rectangle.
            actual_distance (float): Actual distance between rectangles.
            predicted_distance (float): Predicted distance between rectangles.
        """
        fig, ax = plt.subplots()
        rect1_polygon = plt.Polygon(rect1_vertices, color="blue", alpha=0.5)
        ax.add_patch(rect1_polygon)
        rect2_polygon = plt.Polygon(rect2_vertices, color="red", alpha=0.5)
        ax.add_patch(rect2_polygon)
        all_vertices = np.vstack((rect1_vertices, rect2_vertices))
        x_min, x_max = all_vertices[:, 0].min() - 1, all_vertices[:, 0].max() + 1
        y_min, y_max = all_vertices[:, 1].min() - 1, all_vertices[:, 1].max() + 1
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal")
        plt.title(
            f"Actual Distance: {actual_distance:.6f}, Predicted Distance: {predicted_distance:.6f}"
        )
        plt.show()


def main(load_model_flag=False):
    """
    Main function to train or load a model and test its performance.

    Args:
        load_model_flag (bool): Flag to indicate whether to load a pre-trained model.

    Returns:
        DistancePredictor: Trained or loaded model.
    """
    SME = SafetyMarginEstimatorModule()
    if load_model_flag:
        SME.net = SME.load_model()
    else:
        features, labels = SME.generate_training_data()
        SME.net = SME.train_model(features, labels)

    SME.test_model(SME.net)
    return SME.net


if __name__ == "__main__":
    # Execute the main function and test the model
    net = main(load_model_flag=False)
    SME = SafetyMarginEstimatorModule()
    x2, y2, heading2 = 0.1, 0.1, 0.45
    rect1_vertices = SME.get_rectangle_vertices(SME.x1, SME.y1, SME.heading1)
    rect2_vertices = SME.get_rectangle_vertices(x2, y2, heading2)
    rect1_vertices_tensor = torch.tensor(rect1_vertices)
    rect2_vertices_tensor = torch.tensor(rect2_vertices)
    vertices = torch.stack(
        [rect1_vertices_tensor, rect2_vertices_tensor], dim=0
    ).unsqueeze(0)
    actual_distance = get_distances_between_agents(
        vertices, distance_type="mtv", is_set_diagonal=False
    )[0, 0, 1].numpy()
    features = torch.tensor([x2, y2, heading2]) / SME.feature_normalizer
    predicted_distance = net(features)[0, 0].detach().numpy() * SME.label_normalizer
    print(f"actual_distance = {actual_distance}")
    print(f"predicted_distance = {predicted_distance}")
    print(f"Difference: {100*(predicted_distance - actual_distance):.3f} cm")
    SME.visualize_rectangles(
        rect1_vertices, rect2_vertices, actual_distance, predicted_distance
    )
