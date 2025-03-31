
import yaml
from pathlib import Path
from ultralytics import YOLO
import json
import numpy as np
import os
import shutil

### YOLO TRAINING ###
class YoloModel():
    def __init__(self, config):
        self._parse_config(config)

    def train(self):
        if self.pretrained:
            model = YOLO(f"{self.model}.pt")
        else:
            model = YOLO(f"{self.model}.yaml")
        
        train_results = model.train(
            data = self.data,
            batch = self.batchSize,
            project = self.outputPath / "train",
            name = self.name,
            device = self.device,
            **self.trainArgs,
            **self.augmentArgs
        )

    def val(self, model2validate:str="best"): # last
        # load model
        if model2validate == 'best' or model2validate == 'last':
            model_path = self.outputPath / "train" / self.name / "weights" / f"{model2validate}.pt"
        else: # if path to a custom model is given
            model_path = model2validate
        model = YOLO(model_path)

        val_results = model.val(
            data = self.data,
            batch = self.batchSize,
            project = self.outputPath / self.valArgs['split'],
            name = self.name,
            device = self.device,
            **self.valArgs
        )

    def _parse_config(self, config):
        self.model = config['model']
        self.data = Path(config['data'])
        self.pretrained = config['pretrained']
        self.outputPath = Path(config['output_path'])
        self.name = Path(config['name'])
        self.device = config['device']
        self.batchSize = config['batch']
        self.trainArgs = config['train']
        self.augmentArgs = config['augment']
        self.valArgs = config['val']

### CALCULATE SCALE FROM REFERENCE OBJECTS ###
""" 
This function was used to calculate the scale of the reference object, using multiple images of the same object.
There are probably a lot of ways to calculate the scale, but this is the one I used. It requires an annotation file in COCO format.
"""
def calc_reference_scale_from_json(json_path, ref_dict):
    json_path = Path(json_path)
    with open(json_path, 'r') as file:
        data = json.load(file)

    bboxs = {}
    scales = []
    segmasks = {}
    for path, real_world_size in ref_dict.items():
        for image in data['images']:
            if path == image['path']:
                id = image['id']
                for ann in data['annotations']:
                    if ann['image_id'] == id:
                        bbox = ann['bbox']
                        scale1 = real_world_size / bbox[2] # width (xywh format)
                        scale2 = real_world_size / bbox[3] # height (xywh format)
                        scale = (scale1 + scale2) / 2
                        scales.append(scale)
                        x, y, w, h = map(int, bbox)
                        bboxs[path] = (x, y, w, h)
                        segmasks[path] = ann['segmentation']
                        break

                        

    scale = np.mean(scales)
    return scale, bboxs, segmasks

### FUNCTIONS THAT MAY BE USEFUL FOR MANIPULATING THE DATA ###
""" 
These functions are not used in the training process itself, but they are useful for manipulating the datasets. 
YOLO requires a certain format for the datasets. The functions are used to convert the COCO format to the YOLO format. 
There is no example use case for these functions, because it is highly dependent on the dataset. They rather give an idea of 
how you could get from the COCO format to the YOLO format.

"""
class DatasetEditor(): 
    """
    For inheriting the _datasplit_from_list method.
    """
    def _datasplit_from_list(self, data:list, train_ratio:float, valid_ratio:float, eval_ratio:float, seed:int=None):
        """
        Splits the given data list into train, validation, and evaluation sets based on the provided ratios.
        Args:
            data (list): The list of data to be split.
            train_ratio (float): The ratio of data to be allocated for training.
            valid_ratio (float): The ratio of data to be allocated for validation.
            eval_ratio (float): The ratio of data to be allocated for evaluation.
            seed (int, optional): The seed value for random shuffling. Defaults to None.
        Returns:
            tuple: A tuple containing the train data, validation data, and evaluation data (if eval_ratio is not 0).
        """
        
        assert train_ratio + valid_ratio + eval_ratio == 1, "Ratios must sum up to 1"

        if seed:
            np.random.seed(seed)

        # Shuffle the data
        np.random.shuffle(data)
        # Split the data
        num_data = len(data)

        if eval_ratio != 0:
            num_train = int(train_ratio * num_data)
            num_valid = int(valid_ratio * num_data)
            train_data = data[:num_train]
            valid_data = data[num_train:num_train + num_valid]
            eval_data = data[num_train + num_valid:] 
        
        else:
            num_train = int(train_ratio * num_data)
            train_data = data[:num_train]
            valid_data = data[num_train:]
            eval_data = None

        return train_data, valid_data, eval_data

class SegmentationDatasetEditor(DatasetEditor):
    """ 
    A class for editing and manipulating datasets in COCO format.
    Methods:
    - __init__(self, path_json:str): Initializes the DatasetEditor object.
    - modify_coco_annotation(self, replacement_path:str, output_path:str=None): Changes the path of the images in the COCO annotation file and changes the category_id to an index.
    - datasplit_from_coco_annotation(self, train_ratio:float, valid_ratio:float, eval_ratio:float, output_path:str=None, seed=None): Splits the dataset based on COCO annotations.
    - coco2yolo_dataset(self, yolo_data_path:str, yolo_data_yaml_path:str=None, datasplit_output_path:str=None): Converts COCO dataset format to YOLO dataset format.
    """
    
    def __init__(self, path_json:str):
        """
        Initializes the EditDataset object.
        Parameters:
        - path_json (str): The path to the JSON file.
        Returns:
        - None
        """
        with open(path_json, 'r') as file:
            self.data = json.load(file)
    
    def modify_coco_ann(self, replacement_path:str, output_path:str=None, delete_cat:list=[]):
        """_Change the path of the images in the COCO annotation file and change the category_id to an index.
        
        The path of the images in the annotation json file downloaded from our coco annotator tool is the 
        uploading path of the image. All datasets are uploaded from the same machine.
        This function changes the path to the actual local path of the used machine. 
        Furthermore, the coco annotator tool does assign arbitrary category_id values. 
        This function changes the category_id to an index starting from 0._

        Args:
            replacement_path (str): _Location of the dataset. Replaces the uploading path_
            output_path (str, optional): _Filepath of the output annotation file_. Defaults to None.
        """
        # Change the path of the images
        for image in self.data['images']:
            old_path = image['path']
            parts = old_path.split('/')
            replacement = f"{replacement_path}/{parts[-3]}/{parts[-2]}/{parts[-1]}"
            if "ß" in replacement:
                replacement = replacement.replace("ß", "ss")
            image['path'] = replacement

        # Change the category_id to an index
        self.data['categories'] = [cat for cat in self.data['categories'] if cat['name'] not in delete_cat]
        cat_ids = [cat['id'] for cat in self.data['categories']]
        cat_id_to_idx = {cat_id: idx for idx, cat_id in enumerate(sorted(cat_ids))} # mapping

        # Change the category_id in the categories
        for cat in self.data['categories']:
            cat['id'] = cat_id_to_idx[cat['id']]

        # Change the category_id in the annotations
        new_annotations = []
        for ann in self.data['annotations']:
            if ann['category_id'] in cat_id_to_idx:
                ann['category_id'] = cat_id_to_idx[ann['category_id']]
                new_annotations.append(ann)

        # Update annotations after filtering
        self.data['annotations'] = new_annotations

        # Save the modified annotation
        if output_path:
            with open(output_path, 'w') as file:
                json.dump(self.data, file, indent=4)

    def random_datasplit_from_coco_ann(self, train_ratio:float, valid_ratio:float, eval_ratio:float, output_path:str=None, seed=None):
        """
        Split the dataset based on COCO annotations.
        Args:
            train_ratio (float): The ratio of training data.
            valid_ratio (float): The ratio of validation data.
            eval_ratio (float): The ratio of evaluation data.
            output_path (str, optional): The path to save the split dataset. Defaults to None.
            seed (optional): The seed value for randomization. Defaults to None.
        Returns:
            None
        Raises:
            None
        This function takes the COCO annotations of the dataset and splits it into training, validation, and evaluation sets based on the provided ratios. It creates separate dictionaries for each set, mapping image IDs to their respective paths. If an output path is provided, the split dataset is saved as a YAML file.
        The function first creates a dictionary `cornshape2id` to map cornshape names to image IDs. It iterates over the images in the dataset and extracts the cornshape name from the image path. If the cornshape is not "reference", it adds the image ID to the corresponding cornshape key in `cornshape2id`.
        Next, it initializes empty lists `train_ids`, `valid_ids`, and `eval_ids` to store the IDs of images in each set.
        Then, it iterates over the cornshape keys in `cornshape2id` and calls the `_datasplit_from_list` function to split the image IDs into train, valid, and eval sets based on the provided ratios. The resulting IDs are appended to the respective lists.
        After that, it initializes empty dictionaries `train_id2path`, `valid_id2path`, and `eval_id2path` to map image IDs to their paths in each set.
        It then iterates over the images in the dataset again and checks if the image ID is present in the train, valid, or eval sets. If so, it adds the image ID and path to the corresponding dictionary.
        Finally, if an output path is provided, it creates a `dump_data` dictionary containing the train, valid, and eval dictionaries. It saves the `dump_data` as a YAML file at the specified output path.
        Note: The function requires the PyYAML library to be installed.
        """

        cornshape2id = {}
        # Create a dictionary mapping cornshape names to image IDs
        for image in self.data['images']:
            path = image['path']
            id = image['id']
            parts = path.split('/')
            cornshape = parts[-3]
            if cornshape != "reference":
                if cornshape not in cornshape2id:
                    cornshape2id[cornshape] = [id]
                else:
                    cornshape2id[cornshape].append(id)

        train_ids, valid_ids, eval_ids = [], [], []

        # Split the dataset based on cornshape
        for cornshape, ids in cornshape2id.items():
            train, valid, eval = self._datasplit_from_list(ids, train_ratio, valid_ratio, eval_ratio, seed)
            train_ids.extend(train)
            valid_ids.extend(valid)
            if eval:
                eval_ids.extend(eval)

        # Create dictionaries mapping image IDs to their paths
        self.train_id2path, self.valid_id2path, self.eval_id2path = {}, {}, {}
        for image in self.data['images']:
            id = image['id']
            path = image['path']
            if id in train_ids:
                self.train_id2path[id] = path
            elif id in valid_ids:
                self.valid_id2path[id] = path
            elif eval_ids and id in eval_ids:
                self.eval_id2path[id] = path

        # Save the split dataset
        if output_path:
            dump_data = dict(
                train=self.train_id2path,
                valid=self.valid_id2path
            )
            if eval_ids:
                dump_data['eval'] = self.eval_id2path
            with open(output_path, 'w') as file:
                yaml.dump(dump_data, file, default_flow_style=False)

    def datasplit_from_custom_coco_ids(self, train_ids:list, valid_ids:list, eval_ids:list, output_path:str=None):
        """
        Split the dataset based on custom image IDs.
        Args:
            train_ids (list): The list of image IDs for the training set.
            valid_ids (list): The list of image IDs for the validation set.
            eval_ids (list): The list of image IDs for the evaluation set.
            output_path (str, optional): The path to save the split dataset. Defaults to None.
        Returns:
            None
        Raises:
            None
        This function takes custom image IDs and splits the dataset into training, validation, and evaluation sets based on the provided IDs. It creates separate dictionaries for each set, mapping image IDs to their respective paths. If an output path is provided, the split dataset is saved as a YAML file.
        The function initializes empty dictionaries `train_id2path`, `valid_id2path`, and `eval_id2path` to map image IDs to their paths in each set.
        It then iterates over the images in the dataset and checks if the image ID is present in the train, valid, or eval sets. If so, it adds the image ID and path to the corresponding dictionary.
        Finally, if an output path is provided, it creates a `dump_data` dictionary containing the train, valid, and eval dictionaries. It saves the `dump_data` as a YAML file at the specified output path.
        Note: The function requires the PyYAML library to be installed.
        """
        # Create dictionaries mapping image IDs to their paths
        self.train_id2path, self.valid_id2path, self.eval_id2path = {}, {}, {}
        for image in self.data['images']:
            id = image['id']
            path = image['path']
            if id in train_ids:
                self.train_id2path[id] = path
            elif id in valid_ids:
                self.valid_id2path[id] = path
            elif id in eval_ids:
                self.eval_id2path[id] = path

        # Save the split dataset
        if output_path:
            dump_data = dict(
                train=self.train_id2path,
                valid=self.valid_id2path,
                eval=self.eval_id2path
            )
            with open(output_path, 'w') as file:
                yaml.dump(dump_data, file, default_flow_style=False)

    def coco2yolo_dataset(self, yolo_data_path:str, yolo_data_yaml_path:str=None, datasplit_output_path:str=None):   
        """
        Converts COCO dataset format to YOLO dataset format.
        Args:
            self: The instance of the class.
            yolo_data_path (str): The path to the YOLO dataset directory.
            yolo_data_yaml_path (str, optional): The path to the YOLO dataset YAML file. Defaults to None.
            datasplit_output_path (str, optional): The path to the output YAML file containing the data split information. Defaults to None.
        """
        os.makedirs((f"{yolo_data_path}/train/images"), exist_ok = True)
        os.makedirs((f"{yolo_data_path}/train/labels"), exist_ok = True)
        os.makedirs((f"{yolo_data_path}/val/images"), exist_ok = True)
        os.makedirs((f"{yolo_data_path}/val/labels"), exist_ok = True)

        if self.eval_id2path:
            os.makedirs((f"{yolo_data_path}/test/images"), exist_ok = True)
            os.makedirs((f"{yolo_data_path}/test/labels"), exist_ok = True)

        # Convert COCO annotations to YOLO format
        for ann in self.data['annotations']:
            image_id = ann['image_id']
            category_id = ann['category_id']
            segmentation = ann['segmentation']

            # Get image information
            for image in self.data['images']:
                if image['id'] == image_id:
                    image_path = image['path']
                    image_filename = os.path.basename(image['file_name'])
                    image_filename = os.path.splitext(image_filename)[0] # Removing the extension (png)
                    image_width = image['width']
                    image_height = image['height']
                    break

            # Convert segmentation to YOLO format
            yolo_segmentation = [f"{(x) / image_width:.5f} {(y) / image_height:.5f}" for x, y in zip(segmentation[0][::2], segmentation[0][1::2])]
            yolo_segmentation = ' '.join(yolo_segmentation)
            yolo_annotation = f"{category_id} {yolo_segmentation}"

            # Save the image and annotation to the YOLO dataset directory
            if image_id in self.train_id2path:
                shutil.copyfile(image_path, f"{yolo_data_path}/train/images/{image_filename}.png")
                with open(f"{yolo_data_path}/train/labels/{image_filename}.txt", 'a+') as file:
                    file.write(yolo_annotation + '\n')
            elif image_id in self.valid_id2path:
                shutil.copyfile(image_path, f"{yolo_data_path}/val/images/{image_filename}.png")
                with open(f"{yolo_data_path}/val/labels/{image_filename}.txt", 'a+') as file:
                    file.write(yolo_annotation + '\n')
            elif image_id in self.eval_id2path:
                shutil.copyfile(image_path, f"{yolo_data_path}/test/images/{image_filename}.png")
                with open(f"{yolo_data_path}/test/labels/{image_filename}.txt", 'a+') as file:
                    file.write(yolo_annotation + '\n')
        
        # Save the YOLO dataset YAML file
        if yolo_data_yaml_path:
            id2name = {cat['id']: cat['name'] for cat in self.data['categories']}
            id2name = dict(sorted(id2name.items()))

            dump_data = dict(
                path=yolo_data_path,
                train="train",
                val="val",
                names=id2name
            )
            if self.eval_id2path:
                dump_data['test'] = "test"
            else:
                dump_data['test'] = ''
            
            with open(yolo_data_yaml_path, 'w') as file:
                yaml.dump(dump_data, file, default_flow_style=False)
        
        # Save the data split information
        if datasplit_output_path:
            train_dump = {id: {"original_path": path, "yolo_path": f"{yolo_data_path}/train/images/{os.path.basename(path)}"} for id, path in self.train_id2path.items()}
            valid_dump = {id: {"original_path": path, "yolo_path": f"{yolo_data_path}/val/images/{os.path.basename(path)}"} for id, path in self.valid_id2path.items()}
            if self.eval_id2path:
                eval_dump = {id: {"original_path": path, "yolo_path": f"{yolo_data_path}/test/images/{os.path.basename(path)}"} for id, path in self.eval_id2path.items()}

            dump_data = dict(
                train=train_dump,
                valid=valid_dump
            )
            if self.eval_id2path:
                dump_data['eval'] = eval_dump
            
            with open(datasplit_output_path, 'w') as file:
                yaml.dump(dump_data, file, default_flow_style=False)

class ClassificationDatasetEditor(DatasetEditor):
    def __init__(self, class2paths:dict, train_ratio:float, valid_ratio:float, seed:int=None):
        self.class2data = class2paths
        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio
        self.seed = seed

        self.datasplit = {
            "train": {},
            "val": {},
        }

        self._datasplit()
    
    def _datasplit(self):
        for class_name, paths in self.class2data.items():
            train, valid, eval = self._datasplit_from_list(paths, self.train_ratio, self.valid_ratio, 0, self.seed)
            for path in train:
                self.datasplit["train"][path] = class_name
            for path in valid:
                self.datasplit["val"][path] = class_name
        
            if eval:
                print(f"Classname: {class_name}, train: {len(train)}, valid: {len(valid)}, eval: {len(eval)}")
            else:
                print(f"Classname: {class_name}, train: {len(train)}, valid: {len(valid)}")

    def toYolo(self, output_path:str):
        export_dict = {
            "train": {},
            "val": {},
        }

        for split, fp2class in self.datasplit.items():
            for class_name in self.class2data.keys():
                if class_name not in export_dict[split]:
                    export_dict[split][class_name] = {}
                folder_path = os.path.join(output_path, split, class_name)
                os.makedirs(folder_path, exist_ok = True)
        
            for idx, (fp, class_name) in enumerate(fp2class.items()):
                new_fp = os.path.join(output_path, split, class_name, f"{idx}.png")
                export_dict[split][class_name][fp] = new_fp
                shutil.copy(fp, new_fp)
    
        self.old2new = export_dict

    def exportDatasplitYaml(self, output_path:str):
        # os.makedirs(output_path, exist_ok = True)
        with open(output_path, 'w') as file:
            yaml.dump(self.old2new, file, default_flow_style=False)

class ClassificationDatasetEditor_including_eval(DatasetEditor):
    """ 
    no functionality if any of the ratios is 0 -> okay for now bc we are using val and eval anyways
    """
    def __init__(self, class2paths:dict, train_ratio:float, valid_ratio:float, eval_ratio:float, seed:int=None):
        self.class2data = class2paths
        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio
        self.eval_ratio = eval_ratio
        self.seed = seed

        self.datasplit = {
            "train": {},
            "val": {},
            "test": {}
        }

        self._datasplit()
    
    def _datasplit(self):
        for class_name, paths in self.class2data.items():
            train, valid, eval = self._datasplit_from_list(paths, self.train_ratio, self.valid_ratio, self.eval_ratio, self.seed)
            for path in train:
                self.datasplit["train"][path] = class_name
            for path in valid:
                self.datasplit["val"][path] = class_name
            for path in eval:
                self.datasplit["test"][path] = class_name
            # if eval:
            #     for path in eval:
            #         self.datasplit["eval"][path] = class_name

    def toYolo(self, output_path:str):
        export_dict = {
            "train": {},
            "val": {},
            "test": {}
        }

        for split, fp2class in self.datasplit.items():
            for class_name in self.class2data.keys():
                if class_name not in export_dict[split]:
                    export_dict[split][class_name] = {}
                folder_path = os.path.join(output_path, split, class_name)
                os.makedirs(folder_path, exist_ok = True)
        
            for idx, (fp, class_name) in enumerate(fp2class.items()):
                new_fp = os.path.join(output_path, split, class_name, f"{idx}.png")
                export_dict[split][class_name][fp] = new_fp
                shutil.copy(fp, new_fp)
    
        self.old2new = export_dict

    def exportDatasplitYaml(self, output_path:str):
        # os.makedirs(output_path, exist_ok = True)
        with open(output_path, 'w') as file:
            yaml.dump(self.old2new, file, default_flow_style=False)
