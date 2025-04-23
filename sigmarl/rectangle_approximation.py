# Copyright (c) 2024, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Tuple, Optional


class RectangleCircleApproximation:
    """
    Compute and display circles that cover a rectangle.
    The rectangle origin is at its center (0, 0).
    Circles are placed along the length axis at y = 0.
    """

    def __init__(self, length: float, width: float, n_circles: int):
        """
        Set rectangle size and number of circles.

        Args:
            length: Rectangle length (must be >= 0).
            width: Rectangle width (must be >= 0).
            n_circles: Number of circles (must be >= 1).

        Raises:
            ValueError: If any dimension is negative or n_circles < 1.
        """
        if length < 0 or width < 0:
            raise ValueError("length and width must be non-negative.")
        if n_circles < 1:
            raise ValueError("n_circles must be at least 1.")

        self.l = length
        self.w = width
        self.n_circles = n_circles
        self.radius: Optional[float] = None
        self.centers: Optional[List[Tuple[float, float]]] = None

        self.radius = self.calculate_min_radius()
        self.centers = self.get_circle_centers()

    def calculate_min_radius(self) -> float:
        """
        Compute the minimal radius so circles cover the rectangle.
        Each circle must reach the far corners of its segment.

        Returns:
            The minimal radius.
        """
        self.segment = self.l / self.n_circles
        radius = np.hypot(self.segment / 2, self.w / 2)
        return radius

    def get_circle_centers(self) -> List[Tuple[float, float]]:
        """
        Determine the circle center coordinates along the length axis.

        Returns:
            List of (x, y) pairs for circle centers.
        """
        step = self.l / self.n_circles
        start = -self.l / 2 + step / 2
        centers: List[Tuple[float, float]] = []
        for i in range(self.n_circles):
            x = start + i * step
            centers.append((x, 0.0))
        return centers

    def compute_additional_area(self) -> float:
        """
        Compute extra area of circles beyond the rectangle area.
        Follows the original segment-based method.

        Returns:
            Total extra area outside the rectangle.
        """
        if self.radius is None:
            self.radius = self.calculate_min_radius()

        r = self.radius
        w = self.w
        n = self.n_circles
        circle_area = np.pi * r**2

        # Vertical segment
        d = 2 * np.sqrt(max(0.0, r**2 - (w / 2) ** 2))
        theta1 = 2 * np.arccos((w / 2) / r)
        area_sector1 = (theta1 / (2 * np.pi)) * circle_area
        area_triangle1 = 0.5 * d * (w / 2)
        extra1 = area_sector1 - area_triangle1

        # Horizontal segment
        theta2 = np.pi - theta1
        area_sector2 = (theta2 / (2 * np.pi)) * circle_area
        area_triangle2 = w * (d / 2) / 2
        extra2 = area_sector2 - area_triangle2

        extra_area = 2 * n * extra1 + 2 * extra2
        return extra_area

    def plot_coverage(self, filename: Optional[str] = None) -> None:
        """
        Draw the rectangle and covering circles.
        Save to file if filename is given, else display.

        Args:
            filename: Path to save figure (PDF). Display if None.
        """
        if self.radius is None or self.centers is None:
            raise RuntimeError("radius or centers not set.")

        fig, ax = plt.subplots(figsize=(self.l * 5, self.w * 5))

        # Draw rectangle centered at origin
        rect = patches.Rectangle(
            (-self.l / 2, -self.w / 2),
            self.l,
            self.w,
            linewidth=0.2,
            edgecolor="black",
            facecolor="none",
        )
        ax.add_patch(rect)

        # Draw circles
        for x, y in self.centers:
            circle = patches.Circle(
                (x, y),
                self.radius,
                linewidth=0.2,
                edgecolor="black",
                facecolor="tab:blue",
                alpha=0.3,
            )
            ax.add_patch(circle)

        ax.set_aspect("equal", "box")

        # Add margin around shapes
        margin = max(self.l, self.w) * 0.01

        xlim_max = self.radius - self.segment / 2 + self.l / 2 + margin
        xlim_min = -xlim_max
        ylim_max = self.radius + margin
        ylim_min = -ylim_max
        ax.set_xlim(xlim_min, xlim_max)
        ax.set_ylim(ylim_min, ylim_max)

        ax.axis("off")

        if filename:
            plt.savefig(filename, bbox_inches="tight", pad_inches=0)
        else:
            plt.show()

        plt.close(fig)


if __name__ == "__main__":
    length = 0.16
    width = 0.08
    for n in [1, 2, 3, 4, 5]:
        print(f"\nConfiguration: n_circles = {n}")
        try:
            model = RectangleCircleApproximation(length, width, n)
            print(f"  radius = {model.radius:.4f}")
            print(f"  diameter-to-width = {2 * model.radius / width:.2f}")
            extra = model.compute_additional_area()
            print(f"  extra area = {extra:.5f}")
            model.plot_coverage(filename=f"coverage_{n}.pdf")
        except Exception as error:
            print(f"Error for n = {n}: {error}")
