import click
import time
import yaml
from pathlib import Path
from scripts.size_analysis import SizeEstimator
from scripts.yolo import *

def tracking(config, data, name):
    processor = YoloTrackingProcessor(saveOverlay = config["save_seg_masks"])
    visArgs = config["save_masks_visualization"]
    processor._setOverlayOptions(show_segmentation = visArgs["show_pixels"], show_bbox = visArgs["show_bboxs"],
                                 show_conf = visArgs["show_conf"], class_colors = visArgs["color"], overwrite_class = visArgs["overwrite_class"])
    processor._setScoreOptions(alpha = config["alpha"], beta = config["beta"], gamma = config["gamma"])
    processor._setThresholdFactor(threshold = config["factor_std"])
    
    model = YoloTrackingModel()
    output_path = Path(config["output_path"]) / "tracking"
    YoloTrackingModel.predict(
        images = data, 
        model_path = Path(config["seg_model"]),
        output_path = output_path,
        name = name,
        postprocessor = processor,
        batch_size = config["batch"],
        device = config["device"],
        conf = config["conf"],
        iou = config["iou"],
        tracker = Path(config["tracker"])
    )
    processor.export(Path(config["output_path"]) / name, method = config["filter_method"])

def classification_broken(config, data, name):
    processor = YoloClassificationProcessor()
    model = YoloClassificationModel()
    output_path = Path(config["output_path"]) / "classification_broken"
    YoloClassificationModel.predict(
        images = data, 
        model_path = Path(config["cls_broken_model"]),
        output_path = output_path,
        name = name / "classification_broken",
        postprocessor = processor,
        batch_size = config["batch"],
        device = config["device"]
    )
    processor.export(Path(config["output_path"]) / name, method = config["filter_method"])

def classification_tip(config, data, name):
    processor = YoloClassificationProcessor()
    model = YoloClassificationModel()
    output_path = Path(config["output_path"]) / "classification_tip"
    YoloClassificationModel.predict(
        images = data, 
        model_path = Path(config["cls_broken_model"]),
        output_path = output_path,
        name = name,
        postprocessor = processor,
        batch_size = config["batch"],
        device = config["device"]
    )
    processor.export(Path(config["output_path"]) / name, method = config["filter_method"])

def size_estimation(config, data, name):
    if config["scaling_factor"] == True:
        if config["size_method"] == "pca":
            scaling_factor = 8.59
        elif config["size_method"] == "mbr":
            scaling_factor = 5.24
    else:
        scaling_factor = None
    output_path = Path(config["output_path"]) / "size_estimation" / name
    estimator = SizeEstimator(
        source = data,
        scale = config["scale"],
        method = config["size_method"],
        preprocessing_steps = {"open": config["open"], "close": config["close"]},
        scaling_factor = scaling_factor
    )
    estimator.predict(save_img = output_path if config["save_size_masks"] else None)
    estimator.export(output_path = output_path)

@click.command()
@click.option("--config", "-c", prompt="Enter the path to the config file.")
@click.option("--data", "-d", prompt="Enter the path to the data that you want to process.")
@click.option("--name", "-n", prompt="Enter the name of the folder you want to save the processed data in.")
@click.option("--whatrun", "-w", prompt="Choose the program you want to run: \n1. Tracking\n2. Classification (broken/intact)\n3. Classification (tip/no tip)\n4. Size estimation\n5. All programs\nEnter the number of the mode you want to run.")
def main(config, data, name, whatrun):
    config = Path(config)
    data = Path(data)
    name = Path(name)

    with open(config, "r") as file:
        cfg = yaml.safe_load(file)

    if whatrun == "1":
        start = time.time()
        tracking(cfg, data, name)
    elif whatrun == "2":
        start = time.time()
        classification_broken(cfg, data, name)
    elif whatrun == "3":
        start = time.time()
        classification_tip(cfg, data, name)
    elif whatrun == "4":
        start = time.time()
        size_estimation(cfg, data, name)
    elif whatrun == "5":
        start = time.time()
        tracking(cfg, data, name)
        single_seed_masks = data / name / "single_seed_masks"
        classification_broken(cfg, single_seed_masks, name)
        classification_tip(cfg, single_seed_masks, name)
        size_estimation(cfg, single_seed_masks, name)
    else:
        click.echo("Invalid choice. Please enter a valid number.")

    end = time.time()
    click.echo(f"Time taken: {end - start} seconds.")

if __name__ == "__main__":
    click.echo("Welcome to the post-processing program!")
    main()