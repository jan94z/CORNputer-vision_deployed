import os
import json
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tqdm import tqdm
from pathlib import Path

class SizeEstimator:
    def __init__(self, source, scale, method="pca", preprocessing_steps={"open": 0, "close": 0}, scaling_factor=None):
        """
        :param source: Path to images or list of image paths.
        :param scale: Scale factor for real-world measurements.
        :param method: "pca", "convex_hull", or "mbr" (min bounding rectangle).
        :param preprocessing_steps: List of preprocessing steps to apply (e.g., ["open", "close", "blur", "fill_holes"])
        """
        self.scale = scale
        self.method = method.lower()
        self.preprocessing_steps = preprocessing_steps or []

        self.results = {
            "path": [],
            "center_x": [],
            "center_y": [],
            "length": [],
            "width": []
        }
        self.real_world_size = {
            "path": [],
            "length": [],
            "width": []
        }

        source = Path(source)
        self.fps = self._parse_source(source)

        if scaling_factor:
            self.scaling_factor = 1 - scaling_factor / 100
        else:
            self.scaling_factor = None

    def predict(self, batch_size=8, save_img=None):
        """Processes images in batches, applies selected method, and saves results."""
        if save_img:
            os.makedirs(save_img / "images_pixel_size", exist_ok=True)
            os.makedirs(save_img / "images_real_size", exist_ok=True)
            if self.preprocessing_steps["open"] > 0 or self.preprocessing_steps["close"] > 0:
                os.makedirs(save_img / "binary_mask_corrected", exist_ok=True)

        for fp in tqdm(range(0, len(self.fps), batch_size)):
            batch = self.fps[fp:fp + batch_size]
            masks = {f: self._load_img(f) for f in batch}

            for f, (rgb_mask, binary_mask) in masks.items():
                # Choose the method
                if self.method == "pca":
                    center_x, center_y, length, width, p1, p2, s1, s2 = self._pca_analysis(binary_mask)
                    box = None
                elif self.method == "convex_hull":
                    center_x, center_y, length, width, p1, p2, s1, s2 = self._convex_hull_analysis(binary_mask)
                    box = None
                elif self.method == "mbr":
                    center_x, center_y, length, width, p1, p2, s1, s2, box = self._mbr_analysis(binary_mask)
                else:
                    raise ValueError(f"Invalid method: {self.method}")

                # Store results
                self.results["path"].append(f)
                self.results["center_x"].append(center_x)
                self.results["center_y"].append(center_y)
                self.results["length"].append(length)
                self.results["width"].append(width)

                self.real_world_size["path"].append(f)

                if self.scaling_factor:
                    self.real_world_size["length"].append(length * self.scale * self.scaling_factor)
                    self.real_world_size["width"].append(width * self.scale * self.scaling_factor)
                else:
                    self.real_world_size["length"].append(length * self.scale)
                    self.real_world_size["width"].append(width * self.scale)

                # Save annotated images
                if save_img:
                    img_pixel = self._plot_img_pixel(rgb_mask, center_x, center_y, p1, p2, s1, s2, length, width, box)
                    cv2.imwrite(save_img / "images_pixel_size" / os.path.basename(f), img_pixel)
                    img_real_world = self._plot_img_real_world(rgb_mask, center_x, center_y, p1, p2, s1, s2, length * self.scale, width * self.scale, box)
                    cv2.imwrite(save_img / "images_real_size" / os.path.basename(f), img_real_world)
                    if self.preprocessing_steps["open"] > 0 or self.preprocessing_steps["close"] > 0:
                        binary_mask_corrected = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
                        cv2.imwrite(save_img / "binary_mask_corrected" / os.path.basename(f), binary_mask_corrected)

    def export(self, output_path):
        """Exports results to CSV."""
        output_path = Path(output_path)
        os.makedirs(output_path, exist_ok=True)
        df = pd.DataFrame(self.results)
        df.to_csv(output_path / "size_estimation.csv", index=False)
        df_real_world = pd.DataFrame(self.real_world_size)
        df_real_world.to_csv(output_path / "real_world_size.csv", index=False)

    def _pca_analysis(self, binary_mask):
        """Performs PCA-based size estimation."""
        y, x = np.nonzero(binary_mask)
        points = np.column_stack((x, y))
        mean = np.mean(points, axis=0)
        centered_points = points - mean

        # PCA computation
        pca = PCA(n_components=2)
        pca.fit(centered_points)
        principal_axis = pca.components_[0]
        secondary_axis = pca.components_[1]

        # Project points onto principal components
        projected_length = centered_points @ principal_axis
        projected_width = centered_points @ secondary_axis

        length = projected_length.max() - projected_length.min()
        width = projected_width.max() - projected_width.min()

        # Compute endpoints for visualization
        p1 = (int(mean[0] + principal_axis[0] * length / 2), int(mean[1] + principal_axis[1] * length / 2))
        p2 = (int(mean[0] - principal_axis[0] * length / 2), int(mean[1] - principal_axis[1] * length / 2))
        s1 = (int(mean[0] + secondary_axis[0] * width / 2), int(mean[1] + secondary_axis[1] * width / 2))
        s2 = (int(mean[0] - secondary_axis[0] * width / 2), int(mean[1] - secondary_axis[1] * width / 2))

        return mean[0], mean[1], length, width, p1, p2, s1, s2

    def _convex_hull_analysis(self, binary_mask):
        """Performs Convex Hull-based size estimation with correct integer formatting."""
        y, x = np.nonzero(binary_mask)
        points = np.column_stack((x, y))

        # Compute convex hull
        hull = cv2.convexHull(points)  # This might return floats
        hull = hull.squeeze().astype(int)  # Ensure integer coordinates

        # Compute center of hull
        center_x, center_y = np.mean(hull, axis=0).astype(int)

        # Compute bounding box dimensions
        x_min, x_max = hull[:, 0].min(), hull[:, 0].max()
        y_min, y_max = hull[:, 1].min(), hull[:, 1].max()
        length = x_max - x_min
        width = y_max - y_min

        # Compute principal and secondary axis endpoints
        p1, p2 = (x_min, center_y), (x_max, center_y)  # Length axis
        s1, s2 = (center_x, y_min), (center_x, y_max)  # Width axis

        return center_x, center_y, length, width, p1, p2, s1, s2

    def _mbr_analysis(self, binary_mask):
        """Performs Minimum Bounding Rectangle-based size estimation."""
        y, x = np.nonzero(binary_mask)
        points = np.column_stack((x, y))

        # Compute MBR
        rect = cv2.minAreaRect(points)
        (center_x, center_y), (w, h), angle = rect
        box = cv2.boxPoints(rect)
        box = np.array(box, dtype=np.int32)  # Convert to integer coordinates

        # Ensure correct length and width assignment based on the box points
        edge_lengths = [np.linalg.norm(box[i] - box[(i + 1) % 4]) for i in range(4)]
        length_idx = np.argmax(edge_lengths)  # Find the longest edge
        width_idx = (length_idx + 1) % 4  # Width is perpendicular to the length

        length = edge_lengths[length_idx]
        width = edge_lengths[width_idx]

        # Normalize the angle (OpenCV returns -90 to 0 degrees)
        if w < h:
            angle += 90  # Adjust orientation if OpenCV flipped width/height

        # Compute the correct major and minor axes **directly from the box points**
        p1, p2, s1, s2 = self._get_mbr_axes_from_box(center_x, center_y, box, length_idx)

        return int(center_x), int(center_y), length, width, p1, p2, s1, s2, box

    def _get_mbr_axes_from_box(self, center_x, center_y, box, length_idx):
        """
        Computes the principal (length) and secondary (width) axes from the rotated MBR box.
        """
        # Assign endpoints based on box structure
        p1, p2 = box[length_idx], box[(length_idx + 1) % 4]
        s1, s2 = box[(length_idx + 2) % 4], box[(length_idx + 3) % 4]

        # Adjust so axes pass through the center
        p1 = (int(2 * center_x - p1[0]), int(2 * center_y - p1[1]))
        p2 = (int(2 * center_x - p2[0]), int(2 * center_y - p2[1]))
        s1 = (int(2 * center_x - s1[0]), int(2 * center_y - s1[1]))
        s2 = (int(2 * center_x - s2[0]), int(2 * center_y - s2[1]))

        return p1, p2, s1, s2

    def _plot_img_pixel(self, img, center_x, center_y, p1, p2, s1, s2, pixel_length, pixel_width, box=None):
        """
        Annotates and visualizes pixel-based size estimation, showing full bounding box if available.
        """
        plot = img.copy()

        # Draw principal axis (length) in red
        cv2.line(plot, p1, p2, (0, 0, 255), 2)

        # Draw secondary axis (width) in blue
        cv2.line(plot, s1, s2, (255, 0, 0), 2)

        # Draw center point
        cv2.circle(plot, (int(center_x), int(center_y)), 4, (0, 255, 0), -1)

        # Draw bounding box if available
        if box is not None:
            cv2.polylines(plot, [box], isClosed=True, color=(0, 255, 255), thickness=2)

        # Display pixel-based length and width
        cv2.putText(plot, f"Length: {pixel_length:.2f} px", 
                    (int(center_x + 10), int(center_y - 40)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.putText(plot, f"Width: {pixel_width:.2f} px", 
                    (int(center_x + 10), int(center_y + 40)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        return plot

    def _plot_img_real_world(self, img, center_x, center_y, p1, p2, s1, s2, real_world_length, real_world_width, box=None):
        """
        Annotates and visualizes real-world size estimation, showing full bounding box if available.
        """
        plot = img.copy()
        
        # Draw principal axis (length) in red
        cv2.line(plot, p1, p2, (0, 0, 255), 2)
        
        # Draw secondary axis (width) in blue
        cv2.line(plot, s1, s2, (255, 0, 0), 2)
        
        # Draw center point
        cv2.circle(plot, (int(center_x), int(center_y)), 4, (0, 255, 0), -1)

        # Draw bounding box if available
        if box is not None:
            cv2.polylines(plot, [box], isClosed=True, color=(0, 255, 255), thickness=2)

        # Display real-world length and width
        cv2.putText(plot, f"Length: {real_world_length:.2f} mm", 
                    (int(center_x + 10), int(center_y - 40)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.putText(plot, f"Width: {real_world_width:.2f} mm", 
                    (int(center_x + 10), int(center_y + 40)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        return plot
  
    def _load_img(self, fp):
        """Loads and preprocesses an image based on the defined preprocessing steps."""
        rgb_mask = cv2.imread(fp)
        binary_mask = np.where(np.any(rgb_mask != [0, 0, 0], axis=-1), 255, 0).astype(np.uint8)

        kernel = np.ones((3, 3), np.uint8)

        if self.preprocessing_steps["open"] > 0:
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=self.preprocessing_steps["open"])
        if self.preprocessing_steps["close"] > 0:
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=self.preprocessing_steps["close"])

        return rgb_mask, binary_mask
    
    def _parse_source(self, source):
        """Parses the input source (file path, list, or directory)."""
        fps = []
        if source.is_dir():
            fps.extend(source / f for f in os.listdir(source) if f.endswith('.png'))
        elif source.is_file():
            fps.append(source)
        return fps
