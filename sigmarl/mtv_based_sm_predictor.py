# Copyright (c) 2024, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from termcolor import colored
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib
import time

matplotlib.rcParams.update({"font.size": 14})  # Set global font size

from sigmarl.helper_scenario import get_distances_between_agents

import random


# Set seeds for reproducibility
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


class DistancePredictor(nn.Module):
    """Neural network model for predicting safety margin between rectangles."""

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
    """Using neural network to predict safety margin between two car-like robots modeled by rectangles."""

    def __init__(
        self,
        length=0.16,
        width=0.08,
        path_nn="checkpoints/ecc25/sm_predictor.pth",
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
            [pos_normalizer, pos_normalizer, angle_normalizer], dtype=torch.float32
        ).unsqueeze(0)
        self.label_normalizer = pos_normalizer

        # Define feature ranges
        self.radius = np.sqrt(self.length**2 + self.width**2) / 2

        offset = 0.5 * self.length  # Default: 0.5 * length

        self.x_max = 2 * self.radius + offset
        self.y_max = 2 * self.radius + offset
        self.x_min = -self.x_max
        self.y_min = -self.y_max

        # Define excluded rectangle boundaries. If the center of the second rectangle lies inside this excluded rectangle, it must overlap with the ego rectangle
        self.excl_x_max = (self.length + self.width) / 2
        self.excl_y_max = self.width
        self.excl_x_min = -self.excl_x_max
        self.excl_y_min = -self.excl_y_max

        self.heading_max = np.pi
        self.heading_min = -self.heading_max

        self.train_losses_history = None
        self.val_losses_history = None
        self.error_upper_bound = None

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

    def get_rectangle_vertices_batch(self, x, y, theta):
        """
        Compute the vertices of rectangles given their centers, orientations, length, and width.

        Args:
            x (np.ndarray): Array of X-coordinates of rectangle centers.
            y (np.ndarray): Array of Y-coordinates of rectangle centers.
            theta (np.ndarray): Array of orientation angles of rectangles in radians.

        Returns:
            torch.Tensor: Tensor of rectangle vertices, shape [num_samples, 4, 2]
        """
        hl = self.length / 2.0
        hw = self.width / 2.0
        local_vertices = np.array(
            [[-hl, -hw], [-hl, hw], [hl, hw], [hl, -hw]]
        )  # Shape: [4, 2]

        # Compute rotation matrices
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        R = np.stack(
            [
                np.stack([cos_theta, -sin_theta], axis=1),
                np.stack([sin_theta, cos_theta], axis=1),
            ],
            axis=1,
        )  # Shape: [num_samples, 2, 2]

        # Apply rotation to local vertices
        global_vertices = np.einsum(
            "nij,kj->nik", R, local_vertices
        )  # Shape: [num_samples, 4, 2]
        global_vertices = np.transpose(global_vertices, (0, 2, 1))

        # Add translation by expanding dimensions for broadcasting
        translation = np.stack([x, y], axis=1)[
            :, np.newaxis, :
        ]  # Shape: [num_samples, 1, 2]
        global_vertices += translation  # Shape: [num_samples, 4, 2]

        return torch.tensor(global_vertices, dtype=torch.float32)

    def generate_training_data(self):
        """
        Generate training data for the distance prediction model.

        Returns:
            tuple: Features and labels for training.
        """
        num_values = 41  # Adjust as needed
        x_values = np.linspace(self.x_min, self.x_max, num_values)
        y_values = np.linspace(self.y_min, self.y_max, num_values)
        heading_values = np.linspace(self.heading_min, self.heading_max, num_values)
        X2, Y2, H2 = np.meshgrid(x_values, y_values, heading_values, indexing="ij")
        X2, Y2, H2 = X2.flatten(), Y2.flatten(), H2.flatten()
        num_samples = len(X2)
        print(f"Number of training samples: {num_samples}")

        # Combine features
        features = np.column_stack((X2, Y2, H2))

        # Vectorized computation of rectangle vertices
        rect1_vertices = torch.tensor(
            self.get_rectangle_vertices_batch(
                np.array(self.x1).repeat(num_samples),
                np.array(self.y1).repeat(num_samples),
                np.array(self.heading1).repeat(num_samples),
            )
        )  # Shape: [num_samples, 4, 2]

        rect2_vertices = self.get_rectangle_vertices_batch(
            X2, Y2, H2
        )  # Shape: [num_samples, 4, 2]

        # Stack rectangles together
        vertices = torch.stack(
            [rect1_vertices, rect2_vertices], dim=1
        )  # Shape: [num_samples, 2, 4, 2]

        # Compute distances in a vectorized manner
        # get_distances_between_agents expects [batch_size, n_agents, 4, 2]
        distances = get_distances_between_agents(
            vertices, distance_type="mtv", is_set_diagonal=False
        )[
            :, 0, 1
        ]  # Shape: [num_samples]

        labels = distances.numpy()  # Convert to NumPy array

        # Convert features and labels to tensors and normalizue
        features = torch.tensor(features, dtype=torch.float32) / self.feature_normalizer
        labels = (
            torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
            / self.label_normalizer
        )

        return features, labels

    def generate_position_samples(self, num_samples: int) -> tuple:
        """
        Generate x and y position samples within the square excluding the inner rectangle.

        Args:
            num_samples (int): Number of samples to generate.

        Returns:
            tuple: Arrays of x and y positions.
        """
        # Define the four regions outside the excluded rectangle
        # Region 1: Left of the excluded rectangle
        x1_start, x1_end = self.x_min, self.excl_x_min
        y1_start, y1_end = self.y_min, self.y_max

        # Region 2: Right of the excluded rectangle
        x2_start, x2_end = self.excl_x_max, self.x_max
        y2_start, y2_end = self.y_min, self.y_max

        # Region 3: Above the excluded rectangle
        x3_start, x3_end = self.excl_x_min, self.excl_x_max
        y3_start, y3_end = self.excl_y_max, self.y_max

        # Region 4: Below the excluded rectangle
        x4_start, x4_end = self.excl_x_min, self.excl_x_max
        y4_start, y4_end = self.y_min, self.excl_y_min

        # Calculate areas of each region
        area1 = (x1_end - x1_start) * (y1_end - y1_start)
        area2 = (x2_end - x2_start) * (y2_end - y2_start)
        area3 = (x3_end - x3_start) * (y3_end - y3_start)
        area4 = (x4_end - x4_start) * (y4_end - y4_start)
        total_allowed_area = area1 + area2 + area3 + area4

        # Proportion of each region
        prop1 = area1 / total_allowed_area
        prop2 = area2 / total_allowed_area
        prop3 = area3 / total_allowed_area
        prop4 = area4 / total_allowed_area

        # Number of samples per region
        n1 = int(round(num_samples * prop1))
        n2 = int(round(num_samples * prop2))
        n3 = int(round(num_samples * prop3))
        n4 = num_samples - (n1 + n2 + n3)  # Ensure total samples

        # Sample uniformly from each region
        x1 = np.random.uniform(x1_start, x1_end, n1)
        y1 = np.random.uniform(y1_start, y1_end, n1)

        x2 = np.random.uniform(x2_start, x2_end, n2)
        y2 = np.random.uniform(y2_start, y2_end, n2)

        x3 = np.random.uniform(x3_start, x3_end, n3)
        y3 = np.random.uniform(y3_start, y3_end, n3)

        x4 = np.random.uniform(x4_start, x4_end, n4)
        y4 = np.random.uniform(y4_start, y4_end, n4)

        # Concatenate all samples
        x = np.concatenate([x1, x2, x3, x4])

        y = np.concatenate([y1, y2, y3, y4])

        return x, y

    def train_model(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        k_folds: int = 5,
        patience: int = 20,
        factor: float = 0.5,
        min_lr: float = 1e-6,
    ) -> nn.Module:
        """
        Train the distance prediction model using k-fold cross-validation with early stopping and learning rate scheduling.

        Args:
            features (torch.Tensor): Training features.
            labels (torch.Tensor): Training labels.
            k_folds (int, optional): Number of folds for cross-validation. Defaults to 5.
            patience (int, optional): Number of epochs with no improvement for early stopping. Defaults to 10.
            factor (float, optional): Factor by which the learning rate will be reduced. Defaults to 0.5.
            min_lr (float, optional): Minimum learning rate. Defaults to 1e-6.

        Returns:
            nn.Module: Trained model.
        """
        # Check if the folder to save the model exists (note that `self.path_nn` is the full path to the model file)
        if not os.path.exists(os.path.dirname(self.path_nn)):
            print(f"Creating folder {os.path.dirname(self.path_nn)}")
            os.makedirs(os.path.dirname(self.path_nn))

        self.net = DistancePredictor(self.n_features)
        optimizer = optim.Adam(self.net.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=factor,
            patience=patience,
            min_lr=min_lr,
            verbose=True,
        )
        criterion = nn.MSELoss()

        # Lists to store training and validation losses
        self.train_losses_history = []
        self.val_losses_history = []

        training_start_time = time.time()
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

            best_val_loss = float("inf")
            epochs_no_improve = 0
            num_epochs = 200
            best_model_state = None  # Initialize best_model_state

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
                epoch_train_loss = running_loss / len(train_loader.dataset)
                self.train_losses_history.append(epoch_train_loss)

                self.net.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for inputs, targets in val_loader:
                        outputs = self.net(inputs)
                        loss = criterion(outputs, targets)
                        val_loss += loss.item() * inputs.size(0)
                epoch_val_loss = val_loss / len(val_loader.dataset)
                self.val_losses_history.append(epoch_val_loss)

                print(
                    f"Epoch [{epoch+1}/{num_epochs}], "
                    f"Train Loss: {epoch_train_loss:.6f}, Val Loss: {epoch_val_loss:.6f}"
                )

                # Step the scheduler based on validation loss
                scheduler.step(epoch_val_loss)

                # Check for improvement
                if epoch_val_loss < best_val_loss:
                    best_val_loss = epoch_val_loss
                    epochs_no_improve = 0
                    # Save the best model state
                    best_model_state = self.net.state_dict()
                else:
                    epochs_no_improve += 1

                # Early stopping
                if epochs_no_improve >= patience:
                    print(f"Early stopping triggered after {epoch+1} epochs.")
                    break

            if best_model_state is not None:
                # Load the best model state for the current fold
                self.net.load_state_dict(best_model_state)

        training_end_time = time.time()
        print(f"Training time: {training_end_time - training_start_time:.2f} seconds")

        # Save the best model checkpoint
        torch.save(
            {
                "model_state_dict": self.net.state_dict(),
                "train_losses_history": self.train_losses_history,
                "val_losses_history": self.val_losses_history,
            },
            self.path_nn,
        )
        print(f"Model saved to {self.path_nn}")

        # Plot training and validation losses
        self.plot_training_curve()

        return self.net

    def plot_training_curve(self) -> None:
        """
        Plot the training and validation loss curves.

        Args:
            train_losses (list): List of training losses per epoch.
            val_losses (list): List of validation losses per epoch.
            k_folds (int): Number of folds used in training.
        """
        epochs = range(1, len(self.train_losses_history) + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, self.train_losses_history, label="Training Loss")
        plt.plot(epochs, self.val_losses_history, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Training and Validation Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Save the plot
        plt.savefig("mtv_training_losses.png", bbox_inches="tight", dpi=600)
        plt.savefig(
            "mtv_training_losses.pdf", bbox_inches="tight", dpi=300, pad_inches=0
        )
        print(
            colored(f"[INFO] A fig has been saved under", "black"),
            colored(f"mtv_training_losses.png", "blue"),
        )
        plt.show()

    def load_model(self):
        """
        Load a pre-trained model from a file.

        Args:
            load_path (str): Path to the model file.

        Returns:
            DistancePredictor: Loaded model.
        """
        # Load the best model
        checkpoint = torch.load(self.path_nn, weights_only=False)
        self.net = DistancePredictor(self.n_features)
        self.net.load_state_dict(checkpoint["model_state_dict"])
        self.val_losses_history = checkpoint["val_losses_history"]
        self.train_losses_history = checkpoint["train_losses_history"]
        self.error_upper_bound = checkpoint["error_upper_bound"]

        self.net.eval()
        print(
            f"Model loaded from {self.path_nn}, with an error upper bound of {self.error_upper_bound:.6f}"
        )
        return self.net

    def test_model(self, net):
        """
        Test the trained model on random samples and evaluate its performance.

        Args:
            net (DistancePredictor): Trained model.
        """
        num_test_samples = 100000

        # Generate position samples within the allowed regions
        x2_test, y2_test = self.generate_position_samples(num_test_samples)

        heading2_test = np.random.uniform(
            self.heading_min, self.heading_max, num_test_samples
        )

        # Combine features
        test_features = np.column_stack((x2_test, y2_test, heading2_test))

        # Vectorized computation of rectangle vertices
        test_rect1_vertices = self.get_rectangle_vertices_batch(
            np.array(self.x1).repeat(num_test_samples),
            np.array(self.y1).repeat(num_test_samples),
            np.array(self.heading1).repeat(num_test_samples),
        )  # Shape: [num_samples, 4, 2]

        test_rect2_vertices = self.get_rectangle_vertices_batch(
            x2_test, y2_test, heading2_test
        )  # Shape: [num_samples, 4, 2]

        # Stack rectangles together
        test_vertices = torch.stack(
            [test_rect1_vertices, test_rect2_vertices], dim=1
        )  # Shape: [num_samples, 2, 4, 2]

        # Compute distances in a vectorized manner
        # get_distances_between_agents expects [batch_size, n_agents, 4, 2]
        test_labels = get_distances_between_agents(
            test_vertices, distance_type="mtv", is_set_diagonal=False
        )[:, 0, 1].unsqueeze(
            1
        )  # Shape: [num_samples]

        # Convert features and labels to tensors and normalizue
        test_features = torch.tensor(test_features, dtype=torch.float32)
        test_features_normalized = test_features / self.feature_normalizer

        net.eval()
        criterion = nn.MSELoss()
        with torch.no_grad():
            predicted_labels = (
                net(test_features_normalized) * self.label_normalizer
            )  # De-normalizer
            test_loss = criterion(predicted_labels, test_labels)
            print(f"Test Loss: {test_loss.item():.6f}")

        differences = (predicted_labels - test_labels).numpy()
        absolute_errors = np.abs(differences)
        mean_error = np.mean(absolute_errors)

        # Define the number of top errors to find
        num_top_errors = 10

        # Find the top max errors
        top_indices = np.argsort(absolute_errors, axis=0)[-num_top_errors:][::-1]
        top_errors = absolute_errors[top_indices]

        print(f"Top {num_top_errors} Max Errors:")
        for i, idx in enumerate(top_indices):
            print(
                f"Top error {i+1}: {top_errors[i][0, 0]:.6f}, at feature {test_features[idx][0]}. Relative to the width of the vehicle: {top_errors[i][0, 0] / self.width * 100:.2f}%"
            )

        print(
            f"Mean Absolute Error: {mean_error:.6f}. Relative to the width of the vehicle: {mean_error / self.width * 100:.2f}%"
        )

        if self.error_upper_bound is None:
            torch.save(
                {
                    "model_state_dict": self.net.state_dict(),
                    "train_losses_history": self.train_losses_history,
                    "val_losses_history": self.val_losses_history,
                    "error_upper_bound": top_errors[0][0, 0],
                },
                self.path_nn,
            )
            print(f"Model re-saved to {self.path_nn}")

        # Visualize errors in a 3D map
        self.plot_errors_3d(test_features, absolute_errors.flatten())

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
        x_min, x_max = (
            all_vertices[:, 0].min() - self.width,
            all_vertices[:, 0].max() + self.width,
        )
        y_min, y_max = (
            all_vertices[:, 1].min() - self.width,
            all_vertices[:, 1].max() + self.width,
        )
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal")
        plt.title(
            f"Actual Distance: {actual_distance:.6f}, Predicted Distance: {predicted_distance:.6f}"
        )
        plt.show()

    def plot_errors_3d(self, features: np.ndarray, errors: np.ndarray) -> None:
        """
        Plot the errors in a 3D scatter plot with x, y, and heading as axes.

        Args:
            features (np.ndarray): Array of input features (x, y, heading).
            errors (np.ndarray): Array of absolute errors corresponding to the features.
        """
        x = features[:, 0]
        y = features[:, 1]
        heading = features[:, 2]

        x_relative = x / self.length
        y_relative = y / self.length

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection="3d")

        relative_errors = errors / self.length * 100

        scatter = ax.scatter(
            x_relative,
            y_relative,
            heading,
            c=relative_errors,
            cmap="viridis",
            marker="o",
            alpha=0.6,
            s=20,
        )

        ax.set_xlabel(r"Relative $x$ position (%)")
        ax.set_ylabel(r"Relative $y$ position (%)")
        ax.set_zlabel(r"Heading (rad)")
        ax.set_title("Prediction Errors (all relative data based on vehicle length)")
        fig.colorbar(scatter, ax=ax, label="Relative error (%)")
        plt.tight_layout()
        # Save the plot
        plt.savefig("mtv_errors_3d.png", bbox_inches="tight", dpi=600)
        plt.savefig("mtv_errors_3d.pdf", bbox_inches="tight", dpi=300, pad_inches=0)
        print(
            colored(f"[INFO] A fig has been saved under", "black"),
            colored(f"mtv_errors_3d.png", "blue"),
        )
        plt.show()


def main(load_model_flag, is_run_testing, path_nn):
    """
    Main function to train or load a model and test its performance.

    Args:
        load_model_flag (bool): Flag to indicate whether to load a pre-trained model.

    Returns:
        DistancePredictor: Trained or loaded model.
    """
    SME = SafetyMarginEstimatorModule(path_nn=path_nn)
    if os.path.exists(path_nn):
        if load_model_flag:
            SME.net = SME.load_model()
        else:
            input(
                f"Model file already exists at {path_nn}. Press Enter if you want to train a new one and rewrite the existing one..."
            )
            features, labels = SME.generate_training_data()
            SME.net = SME.train_model(features, labels)

    else:
        if load_model_flag:
            print(f"Model file at {path_nn} not found. Training a new model instead.")
        features, labels = SME.generate_training_data()
        SME.net = SME.train_model(features, labels)

    if is_run_testing:
        SME.test_model(SME.net)

    return SME.net


if __name__ == "__main__":
    # Execute the main function and test the model
    net = main(
        load_model_flag=False,
        is_run_testing=True,
        path_nn="checkpoints/ecc25/sm_predictor.pth",
    )
    SME = SafetyMarginEstimatorModule()
