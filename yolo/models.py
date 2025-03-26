from ultralytics import YOLO
import os
from tqdm import tqdm

class YoloBaseModel():
    def __init__(self, config):
        self._parse_config(config)

    def train(self):
        # load model
        if self.pretrained:
            model = YOLO(f"{self.model}.pt")
        else:
            model = YOLO(f"{self.model}.yaml")

        # train model
        train_results = model.train(
            data = self.data,
            batch = self.batchSize,
            project = os.path.join(self.outputPath, "train"),
            name = self.name,
            device = self.device,
            **self.trainArgs,
            **self.augmentArgs
        )
        
    def val(self, trainedModel:str='best'):
        # 'best' or 'last'
        # load model
        if trainedModel == 'best' or trainedModel == 'last':
            model_path = os.path.join(self.outputPath, "train", self.name, "weights", f"{trainedModel}.pt")
        else:
            model_path = trainedModel
        model = YOLO(model_path)

        # validate model
        val_results = model.val(
            data = self.data,
            batch = self.batchSize,
            project = os.path.join(self.outputPath, self.valArgs['split']),
            name = self.name,
            device = self.device,
            **self.valArgs
        )

        return val_results

    def predict(self, images:list, output_name, postprocessor, trainedModel:str='best'):
        # load model
        if trainedModel == 'best' or trainedModel == 'last':
            model_path = os.path.join(self.outputPath, "train", self.name, "weights", f"{trainedModel}.pt")
        else:
            model_path = trainedModel
        model = YOLO(model_path)

        if output_name is None:
            project = os.path.join(self.outputPath, "predict", self.name)
        else:
            project = os.path.join(self.outputPath, "predict", self.name, output_name)

        images = sorted(images)
        for i in tqdm(range(0, len(images), self.batchSize), desc="Predicting"):
            batch = images[i:i + self.batchSize]
            results = model.predict(
                batch = self.batchSize,
                source = batch,
                project = project,
                name = self.name,
                device = self.device,
                **self.inferArgs
            )
            for r in results:
                self._process_result(r, postprocessor, project)

        return project

    def _parse_config(self, config):
        self.data = config['data']
        self.task = config['task']
        self.model = config['model']
        self.pretrained = config['pretrained']
        self.outputPath = config['output_path']
        self.name = config['name']
        self.device = config['device']
        self.batchSize = config['batch']
        self.trainArgs = config['train']
        self.augmentArgs = config['augment']
        self.valArgs = config['val']
        self.inferArgs = config['infer']
        self.customArgs = config['customArgs']

    def _process_result(self, result, postprocessor, output_path):
        postprocessor.process(result, output_path)

class YoloClassificationModel(YoloBaseModel):
    def __init__(self, config):
        super().__init__(config)

class YoloSegmentationModel(YoloBaseModel):
    def __init__(self, config):
        super().__init__(config)
        # for safety, remove tracker from config -> segmentation and tracking configs are identical apart from tracker
        if 'tracker' in config['infer'].keys():
            del config['infer']['tracker']

class YoloTrackingModel(YoloBaseModel):
    def __init__(self, config):
        super().__init__(config)
    
    def predict(self, images:list, output_name, postprocessor, trainedModel:str='best'):
        # load model
        if trainedModel == 'best' or trainedModel == 'last':
            model_path = os.path.join(self.outputPath, "train", self.name, "weights", f"{trainedModel}.pt")
        else:
            model_path = trainedModel

        model = YOLO(model_path)

        if output_name is None:
            project = os.path.join(self.outputPath, "predict", self.name)
        else:
            project = os.path.join(self.outputPath, "predict", self.name, output_name)

        images = sorted(images)
        for i in tqdm(range(0, len(images), self.batchSize), desc="Predicting"):
            batch = images[i:i + self.batchSize]
            results = model.track(
                persist = True,
                batch = self.batchSize,
                source = batch,
                project = project,
                name = self.name,
                device = self.device,
                **self.inferArgs
            )
            for r in results:
                self._process_result(r, postprocessor, project)

        return project


