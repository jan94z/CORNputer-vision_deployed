import os
from size_analysis.size_analysis import * 
from tqdm import tqdm
import yaml
import click
from pathlib import Path

@click.command()
@click.option("--config", "-c", prompt="Enter path to config file")
def main(config):
    # load config
    config = Path(config)
    with open(config, "r") as fp:
        cfg = yaml.safe_load(fp)
    
    # parse config
    input_path = Path(cfg["input_path"])
    output_path = Path(cfg["output_path"])
    scale = cfg["scale"]
    method = cfg["method"]
    open = cfg["open"]
    close = cfg["close"]
    if cfg["scaling_factor"] == True:
        if method == "pca":
            scaling_factor = 8.59
        elif method == "mbr":
            scaling_factor = 5.24
    else:
        scaling_factor = None
    save_img = cfg["save_img"]

    # init estimator
    estimator = SizeEstimator(source = input_path,
                                scale = scale,
                                method = method,
                                preprocessing_steps = {"open": open, "close": close},
                                scaling_factor = scaling_factor)

    # predict and export
    estimator.predict(save_img = output_path if save_img else None)
    estimator.export(output_path = output_path)
        
if __name__ == "__main__":
    main()