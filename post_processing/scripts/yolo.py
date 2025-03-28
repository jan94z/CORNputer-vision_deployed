import cv2
import numpy as np
import os
import pandas as pd
from ultralytics import YOLO
from tqdm import tqdm
from pathlib import Path

#### MODELS ####

class YoloBaseModel():
    def __init__(self):
        self.standard_args = {
            "verbose": False,
            "half": True,
            "max_det": 300,
            "vid_stride": 1,
            "stream_buffer": False,
            "visualize": False,
            "augment": False,
            "agnostic_nms": False,
            "classes": None,
            "retina_masks": False,
            "embed": None,
            "show": False,
            "save": False,
            "save_txt": False,
            "save_conf": False,
            "save_crop": False,
            "show_labels": True,
            "show_conf": True,
            "show_boxes": True,
            "line_width": None
        }

    def _process_result(self, result, postprocessor, output_path):
        postprocessor.process(result, output_path)

    def _parse_source(self, source):
        """Parses the input source (file path, list, or directory)."""
        fps = []
        if source.is_dir():
            fps.extend(source / f for f in os.listdir(source) if f.endswith('.png'))
        elif source.is_file():
            fps.append(source)
        return fps
    
class YoloClassificationModel(YoloBaseModel):
    def __init__(self):
        super().__init__()

    def predict(self, images, model_path, output_path, name, postprocessor, batch_size:int=8, device=None, **inferArgs):
        images = self._parse_source(images)
        # load model
        model_path = Path(model_path)
        model = YOLO(model_path)
        output_path = Path(output_path)
        complete_output_path = output_path / name

        images = sorted(images)
        for i in tqdm(range(0, len(images), batch_size), desc="Classifying"):
            batch = images[i:i + batch_size]
            results = model.predict(
                batch = batch_size,
                source = batch,
                project = output_path,
                name = name,
                device = device,
                **inferArgs,
                **self.standard_args
            )
            for r in results:
                self._process_result(r, postprocessor, complete_output_path)

class YoloTrackingModel(YoloBaseModel):
    def __init__(self):
        super().__init__()
    
    def predict(self, images, model_path, output_path, name, postprocessor, batch_size:int=8, device=None, **inferArgs):
        images = self._parse_source(images)
        # load model
        model_path = Path(model_path)
        model = YOLO(model_path)
        output_path = Path(output_path)
        complete_output_path = output_path / name 

        images = sorted(images)
        for i in tqdm(range(0, len(images), batch_size), desc="Tracking"):
            batch = images[i:i + batch_size]
            results = model.track(
                persist = True,
                batch = batch_size,
                source = batch,
                project = output_path,
                name = name,
                device = device,
                **inferArgs,
                **self.standard_args
            )
            for r in results:
                self._process_result(r, postprocessor, complete_output_path)

#### PROCESSORS ####

class YoloBaseProcessor():
    def __init__(self):
        pass

    def process(self):
        pass

    def _parseYoloObject(self, obj):
        img = cv2.cvtColor(obj.orig_img, cv2.COLOR_BGR2RGB)
        xyxy = obj.boxes.xyxy.cpu().numpy()
        xywh = obj.boxes.xywh.cpu().numpy()
        cls = obj.boxes.cls.cpu().numpy()
        conf = obj.boxes.conf.cpu().numpy()
        masks = obj.masks.data.cpu().numpy() if obj.masks is not None else None
        names = obj.names 
        orig_path = obj.path
        return img, xyxy, xywh, cls, conf, masks, names, orig_path

class YoloClassificationProcessor(YoloBaseProcessor):
    def __init__(self):
        self.probs = []

    def process(self, r, output_path):
        output_path = Path(output_path)
        os.makedirs(output_path, exist_ok = True)
        filename = os.path.basename(r.path)
        filename = os.path.splitext(filename)[0]
        r.save(output_path / filename + ".png")

        for cls, prob in zip(r.probs.cpu().numpy().top5, r.probs.cpu().numpy().top5conf):
            self.probs.append({'path': r.path, 'cls': cls, 'conf': prob})

    def export(self, output_path):
        output_path = Path(output_path)
        export_df = pd.DataFrame(self.probs)
        export_df.to_csv(output_path / "probs.csv", index=False)

class YoloTrackingProcessor(YoloBaseProcessor):
    def __init__(self, saveOverlay:bool = False):
        self.saveOverlay = saveOverlay
        self.metrics = {
            "id": [],
            "orig_path": [],
            "center": [],
            "score": [],
            "conf": [],
            "distance_to_center": [],
            "norm_center_proximity": [],
            "sharpness": [],
            "norm_sharpness": []
        }

        self.runningMaxSharpness = 0
        self.idCounts = {}
        self.bestInstances = {}
        self.bestInstanceMasks = {"rgb": {}, "binary": {}}
        

        self._setOverlayOptions()
        self._setScoreOptions()
        self._setThresholdFactor()

    def process(self, r, output_path):
        output_path = Path(output_path)
        img, xyxy, xywh, cls, conf, masks, names, orig_path, ids = self._parseYoloObject(r)
        filename = os.path.basename(orig_path)
        filename = os.path.splitext(filename)[0]

        self._extractMetrics(img, xywh, conf, masks, orig_path, ids)

        if self.saveOverlay:
            os.makedirs(output_path / "overlay", exist_ok = True)
            overlay = self._overlay(img, xyxy, cls, conf, masks, names, ids)
            overlay = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path / "overlay" / f"{filename}.png", overlay)

    def export(self, output_path, method):
        output_path = Path(output_path)
        os.makedirs(output_path / "single_seed_masks_before" / "rgb", exist_ok = True)
        os.makedirs(output_path / "single_seed_masks_before" / "binary", exist_ok = True)
        os.makedirs(output_path / "single_seed_masks" / "rgb", exist_ok = True)
        os.makedirs(output_path / "single_seed_masks" / "binary", exist_ok = True)

        self.metrics_df = pd.DataFrame(self.metrics)
        self.metrics_df.to_csv(output_path / "metrics.csv", index=False)

        for id_, instance in self.bestInstances.items():
            instance["count"] = self.idCounts.get(id_, 0)  # Add final count
        
        self.best_instances_df = pd.DataFrame.from_dict(self.bestInstances, orient="index").reset_index(drop=True)
        self.best_instances_df.to_csv(output_path / "best_instances.csv", index=False)

        # filter out outliers
        if method == "mean_std":
            mean_count, std_count = self.best_instances_df['count'].mean(), self.best_instances_df['count'].std()
            thresh_count = mean_count - self.factor_std * std_count
            thresh_count = abs(thresh_count) if thresh_count < 0 else thresh_count # safeguard against negative threshold
            self.best_instances_cleaned_df = self.best_instances_df[self.best_instances_df['count'] > thresh_count]
            self.best_instances_cleaned_df.to_csv(os.path.join(output_path, "best_instances_cleaned.csv"), index=False)
            self.count_stats = pd.DataFrame({"mean": [mean_count], "std": [std_count], "thresh": [thresh_count]})
        elif method == "mad":
            # Compute Median Absolute Deviation (MAD)
            median_count = self.best_instances_df['count'].median()
            mad_count = (self.best_instances_df['count'] - median_count).abs().median()
            # Define threshold using MAD (typically 2.5 to 3 times MAD)
            thresh_count = median_count - self.factor_std * mad_count 
            # Filter out outliers
            self.best_instances_cleaned_df = self.best_instances_df[self.best_instances_df['count'] > thresh_count]
            # Save cleaned data
            self.best_instances_cleaned_df.to_csv(os.path.join(output_path, "best_instances_cleaned.csv"), index=False)
            # Save threshold statistics
            self.count_stats = pd.DataFrame({"median": [median_count], "mad": [mad_count], "thresh": [thresh_count]})
        self.count_stats.to_csv(output_path / "count_stats.csv")

        # Save masks
        for id_ in self.best_instances_df['id']:
            cv2.imwrite(output_path / "single_seed_masks_before" / "binary" / f"{id_}.png", self.bestInstanceMasks["binary"][id_])
            cv2.imwrite(output_path / "single_seed_masks_before" / "rgb" / f"{id_}.png", cv2.cvtColor(self.bestInstanceMasks["rgb"][id_], cv2.COLOR_RGB2BGR))

        for id_ in self.best_instances_cleaned_df['id']:
            cv2.imwrite(output_path / "single_seed_masks" / "binary" / f"{id_}.png", self.bestInstanceMasks["binary"][id_])
            cv2.imwrite(output_path / "single_seed_masks" / "rgb" / f"{id_}.png", cv2.cvtColor(self.bestInstanceMasks["rgb"][id_], cv2.COLOR_RGB2BGR))

    def _extractMetrics(self, img, xywh, conf, masks, orig_path, ids):
        if ids is not None:
            # Prepare masks as binary images
            masks = (masks * 255).astype(np.uint8)

            # Calculate sharpness for all masks
            sharpness_vals = np.array([cv2.Laplacian(mask, cv2.CV_64F).var() for mask in masks])
            self.runningMaxSharpness = max(self.runningMaxSharpness, sharpness_vals.max())
            norm_sharpness_vals = sharpness_vals / self.runningMaxSharpness if self.runningMaxSharpness > 0 else np.zeros_like(sharpness_vals)

            # Extract bounding box center coordinates
            bbox_centers = np.array([[x, y] for x, y, _, _ in xywh])
            img_center = np.array([img.shape[1] / 2, img.shape[0] / 2])  # (width/2, height/2)

            # Calculate distances to image center 
            distances = np.linalg.norm(bbox_centers - img_center, axis=1)
            max_distance = np.sqrt((img_center[0] ** 2) + (img_center[1] ** 2))
            norm_proximities = 1 - (distances / max_distance)

            # Calculate scores for all instances
            scores = (
                self.alpha * np.array(conf) +
                self.beta *  norm_proximities +
                self.gamma * norm_sharpness_vals
            )

            # Update best instances dynamically
            for i, id_ in enumerate(ids):
                id_ = int(id_)
                if id_ not in self.idCounts:
                    self.idCounts[id_] = 0
                self.idCounts[id_] += 1

                if id_ not in self.bestInstances or self.bestInstances[id_]["score"] < scores[i]:
                    self.bestInstances[id_] = {
                        "id": id_, 
                        "conf": conf[i],
                        "distance_to_center": distances[i],
                        "norm_center_proximity": norm_proximities[i],
                        "sharpness": sharpness_vals[i],
                        "norm_sharpness": norm_sharpness_vals[i],
                        "score": scores[i],
                        "orig_path": orig_path,
                    }

                    self.bestInstanceMasks["binary"][id_] = masks[i]
                    rgb_mask = np.dstack([masks[i]] * 3)
                    rgb_mask = np.where(rgb_mask == 255, img, 0)
                    self.bestInstanceMasks["rgb"][id_] = rgb_mask

            # Update metrics dictionary
            self.metrics["id"].extend(ids)
            self.metrics["orig_path"].extend([orig_path] * len(ids))
            self.metrics["center"].extend(bbox_centers)
            self.metrics["score"].extend(scores)
            self.metrics["conf"].extend(conf)
            self.metrics["distance_to_center"].extend(distances)
            self.metrics["norm_center_proximity"].extend(norm_proximities)
            self.metrics["sharpness"].extend(sharpness_vals)
            self.metrics["norm_sharpness"].extend(norm_sharpness_vals)

    def _overlay(self, img, xyxy, cls, conf, masks, names, ids):
        # Apply class name overrides if provided
        if self.overwrite_class:
            for k, v in names.items():
                if v in self.overwrite_class:
                    names[k] = self.overwrite_class[v]

        # Filter classes if specified
        class2idx = {list(names.keys()).index(c): i for i, c in enumerate(self.filter_class)} if self.filter_class else {i: i for i in range(len(names))}

        # assign random color if class_colors is None
        if self.class_colors is None:
            num_classes = len(names)
            self.class_colors = [tuple(np.random.randint(0, 256, 3).tolist()) for _ in range(num_classes)]

        overlay = img.copy()

        for box, cl, con, mask, id in zip(xyxy, cls, conf, masks if masks is not None else [None]*len(xyxy), ids if ids is not None else [None]*len(xyxy)):
            if self.filter_class is None or cl in class2idx:
                color_index = class2idx[cl]
                x1, y1, x2, y2 = np.round(box).astype(int)
                id = id if id is not None else None

                if self.show_segmentation and mask is not None:
                    colored_mask = np.dstack([mask] * 3)
                    colored_mask = np.uint8(np.where(colored_mask == 1, self.class_colors[color_index], 0))
                    overlay = cv2.addWeighted(overlay, 1, colored_mask, 0.5, 0)

                if self.show_bbox:
                    overlay = cv2.rectangle(overlay, (x1, y1), (x2, y2), self.class_colors[color_index], 2)

                label = f"{names[cl]}"
                if self.show_conf:
                    label += f": {con:.2f}"

                overlay = cv2.putText(overlay, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, self.class_colors[color_index], 2)

                if id:
                    overlay = cv2.putText(overlay, f"ID: {id}", (x2, y2-10), cv2.FONT_HERSHEY_SIMPLEX, 1, self.class_colors[color_index], 2)

        return overlay

    def _parseYoloObject(self, obj):
        img, xyxy, xywh, cls, conf, masks, names, orig_path = super()._parseYoloObject(obj)
        ids = obj.boxes.id.cpu().numpy().astype(int) if obj.boxes.is_track else None
        return img, xyxy, xywh, cls, conf, masks, names, orig_path, ids
   
    def _setScoreOptions(self, alpha=1/3, beta=1/3, gamma=1/3):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
    
    def _setThresholdFactor(self, factor_std=2):
        self.factor_std = factor_std

    def _setOverlayOptions(self, show_segmentation:bool=True, show_bbox:bool=True, class_colors:list=None, 
                        overwrite_class:dict=None, filter_class:list=None, show_conf:bool=True):
        
        self.show_segmentation = show_segmentation
        self.show_bbox = show_bbox
        self.class_colors = class_colors
        self.overwrite_class = overwrite_class
        self.filter_class = filter_class
        self.show_conf = show_conf




