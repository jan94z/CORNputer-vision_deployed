import yaml
import os
from tqdm import tqdm
from yolo.models import *
from yolo.processors import *
from yolo.tracking_postprocessing import *

def train(config_path, train=True, test = True, model_path=None):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    if config['task'] == 'track':
        model = YoloTrackingModel(config)
    elif config['task'] == 'segment':
        model = YoloSegmentationModel(config)
    elif config['task'] == 'classify':
        model = YoloClassificationModel(config)
    
    if train:
        model.train()
    if test:
        model.val(trainedModel=config["customArgs"]['trained_model'] if model_path is None else model_path)

def predict(config_path, images, output_name, model_path=None):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    customArgs = config['customArgs']
    
    if config['task'] == 'track':
        model = YoloTrackingModel(config)
        processor = YoloTrackingProcessor(calcMetrics = customArgs['calc_metrics'], saveOverlay = customArgs['save_overlay'])
        processor._setOverlayOptions(show_segmentation = customArgs['show_segmentation'],
                                    show_bbox = customArgs['show_bbox'],
                                    class_colors = customArgs['class_colors'],
                                    overwrite_class = customArgs['overwrite_class'],
                                    filter_class = customArgs['filter_class'],
                                    show_conf = customArgs['show_conf'])
        processor._setScoreOptions(alpha = customArgs['alpha'], beta = customArgs['beta'], gamma = customArgs['gamma'])
        processor._setThresholdFactor(factor_std = customArgs['factor_std'])
        export_path = model.predict(images, output_name, processor, trainedModel=customArgs['trained_model'] if model_path is None else model_path)
        processor.export(export_path)
    
    elif config['task'] == 'segment':
        model = YoloSegmentationModel(config)
        processor = YoloSegmentationProcessor(save_overlay = customArgs['save_overlay'])
        processor._setOverlayOptions(show_segmentation = customArgs['show_segmentation'],
                                    show_bbox = customArgs['show_bbox'],
                                    class_colors = customArgs['class_colors'],
                                    overwrite_class = customArgs['overwrite_class'],
                                    filter_class = customArgs['filter_class'],
                                    show_conf = customArgs['show_conf'])
        export_path = model.predict(images, output_name, processor)
        # processor.export(export_path)

    elif config['task'] == 'classify':
        model = YoloClassificationModel(config)
        processor = YoloClassificationProcessor()
        ...
        export_path = model.predict(images, output_name, processor, trainedModel=customArgs['trained_model'] if model_path is None else model_path)
        processor.export(export_path)

def tracking1909():
    cfg_path = "/home/jan/CORNputer-vision2/yolo/configs/params/tracking1909.yaml"
    # train(cfg_path, train=True, test=True
    pred_path = "/home/jan/datasets/1909/"
    for path in tqdm(os.listdir(pred_path), desc="Tracking"):
        if path == "ref":
            continue
        
        class_path = os.path.join(pred_path, path, "rgb")
        if os.path.isdir(class_path):
            data = [os.path.join(class_path, f) for f in os.listdir(class_path) if f.endswith('.png')]
            print(path, ": ")
            predict(config_path = cfg_path,
                    images = data,
                    output_name = path,
                    model_path="/home/jan/output/1909/train/tracking/weights/best.pt")

def classification1909_broken():
    cfg_path = "/home/jan/CORNputer-vision2/yolo/configs/params/classification1909_broken.yaml"
    train(cfg_path, train=True, test=True)
    pred_path = "/home/jan/datasets/1909_classification_broken/val"
    broken_data = [os.path.join(pred_path, "broken", f) for f in os.listdir(os.path.join(pred_path, "broken")) if f.endswith('.png')]
    intact_data = [os.path.join(pred_path, "intact", f) for f in os.listdir(os.path.join(pred_path, "intact")) if f.endswith('.png')]
    predict(config_path = cfg_path,
            images = broken_data,
            output_name = "broken")
    predict(config_path = cfg_path,
            images = intact_data,
            output_name = "intact")
    
def classification1909_embryo():
    cfg_path = "/home/jan/CORNputer-vision2/yolo/configs/params/classification1909_embryo.yaml"
    train(cfg_path, train=True, test=True)
    pred_path = "/home/jan/datasets/1909_classification_embryo/val"
    embryo_data = [os.path.join(pred_path, "embryo", f) for f in os.listdir(os.path.join(pred_path, "embryo")) if f.endswith('.png')]
    no_embryo_data = [os.path.join(pred_path, "no_embryo", f) for f in os.listdir(os.path.join(pred_path, "no_embryo")) if f.endswith('.png')]
    predict(config_path = cfg_path,
            images = embryo_data,
            output_name = "embryo")
    predict(config_path = cfg_path,
            images = no_embryo_data,
            output_name = "no_embryo")

def tracking1712():
    pred_path = "/home/jan/datasets/1712/rgb"
    images = [os.path.join(pred_path, f) for f in os.listdir(pred_path) if f.endswith('.png')]
    cfg_path = "/home/jan/CORNputer-vision2/yolo/configs/params/tracking1712.yaml"
    predict(config_path = cfg_path,
            images = images,
            output_name = "1712",
            model_path = "/home/jan/output/1909/train/tracking/weights/best.pt")

def tracking0502_D3():
    cfg_path = "/home/jan/CORNputer-vision2/yolo/configs/params/tracking0502_D3.yaml"
    pred_path = "/home/jan/datasets/0502/D3"
    for path in tqdm(os.listdir(pred_path), desc="Tracking"):
        if path == "ref":
            continue
        
        class_path = os.path.join(pred_path, path, "rgb")
        if os.path.isdir(class_path):
            data = [os.path.join(class_path, f) for f in os.listdir(class_path) if f.endswith('.png')]
            print(path, ": ")
            predict(config_path = cfg_path,
                    images = data,
                    output_name = path,
                    model_path = "/home/jan/output/1909/train/tracking/weights/best.pt")

def classification0502_broken():
    cfg_path = "/home/jan/CORNputer-vision2/yolo/configs/params/classification0502_broken.yaml"
    train(cfg_path, train=False, test=True, model_path="/home/jan/output/1909/train/classification_broken/weights/best.pt")
    pred_path = "/home/jan/datasets/0502_evaluation_broken/val"
    broken_data = [os.path.join(pred_path, "broken", f) for f in os.listdir(os.path.join(pred_path, "broken")) if f.endswith('.png')]
    intact_data = [os.path.join(pred_path, "intact", f) for f in os.listdir(os.path.join(pred_path, "intact")) if f.endswith('.png')]
    predict(config_path = cfg_path,
            images = broken_data,
            output_name = "broken",
            model_path = "/home/jan/output/1909/train/classification_broken/weights/best.pt")
    predict(config_path = cfg_path,
            images = intact_data,
            output_name = "intact",
            model_path = "/home/jan/output/1909/train/classification_broken/weights/best.pt")

def classification0502_embryo():
    cfg_path = "/home/jan/CORNputer-vision2/yolo/configs/params/classification0502_embryo.yaml"
    train(cfg_path, train=False, test=True, model_path="/home/jan/output/1909/train/classification_embryo/weights/best.pt")
    pred_path = "/home/jan/datasets/0502_evaluation_embryo/val"
    embryo_data = [os.path.join(pred_path, "embryo", f) for f in os.listdir(os.path.join(pred_path, "embryo")) if f.endswith('.png')]
    no_embryo_data = [os.path.join(pred_path, "no_embryo", f) for f in os.listdir(os.path.join(pred_path, "no_embryo")) if f.endswith('.png')]
    predict(config_path = cfg_path,
            images = embryo_data,
            output_name = "embryo",
            model_path = "/home/jan/output/1909/train/classification_embryo/weights/best.pt")
    predict(config_path = cfg_path,
            images = no_embryo_data,
            output_name = "no_embryo",
            model_path = "/home/jan/output/1909/train/classification_embryo/weights/best.pt")

def tracking0502_D4():
    cfg_path = "/home/jan/CORNputer-vision2/yolo/configs/params/tracking0502_D4.yaml"
    pred_path = "/home/jan/datasets/0502/D4"
    for path in tqdm(os.listdir(pred_path), desc="Tracking"):
        if path == "ref":
            continue
        
        class_path = os.path.join(pred_path, path, "rgb")
        if os.path.isdir(class_path):
            data = [os.path.join(class_path, f) for f in os.listdir(class_path) if f.endswith('.png')]
            print(path, ": ")
            predict(config_path = cfg_path,
                    images = data,
                    output_name = path,
                    model_path = "/home/jan/output/1909/train/tracking/weights/best.pt")

def tracking1102_D5():
    cfg_path = "/home/jan/CORNputer-vision2/yolo/configs/params/tracking1102.yaml"
    pred_path = "/home/jan/datasets/1102"
    for path in tqdm(os.listdir(pred_path), desc="Tracking"):
        
        class_path = os.path.join(pred_path, path, "rgb")
        if os.path.isdir(class_path):
            data = [os.path.join(class_path, f) for f in os.listdir(class_path) if f.endswith('.png')]
            print(path, ": ")
            predict(config_path = cfg_path,
                    images = data,
                    output_name = path,
                    model_path = "/home/jan/output/1909/train/tracking/weights/best.pt")

def tracking2502_D6():
    cfg_path = "/home/jan/CORNputer-vision2/yolo/configs/params/tracking2502.yaml"
    pred_path = "/home/jan/datasets/2502"
    for path in tqdm(os.listdir(pred_path), desc="Tracking"):
        if path == "ref":
            continue
        
        class_path = os.path.join(pred_path, path, "rgb")
        if os.path.isdir(class_path):
            data = [os.path.join(class_path, f) for f in os.listdir(class_path) if f.endswith('.png')]
            print(path, ": ")
            predict(config_path = cfg_path,
                    images = data,
                    output_name = path,
                    model_path = "/home/jan/output/1909/train/tracking/weights/best.pt")





def main():
    pass


if __name__ == "__main__":
    main()