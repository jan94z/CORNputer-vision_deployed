from model_development.mods import YoloModel
import yaml
import click
from pathlib import Path

@click.command()
@click.option('--config', '--c', prompt='Enter the path to the config file.')
@click.option('--train', '--t', prompt='Train? (y/n)')
@click.option('--val', '--v', prompt='Validate? (y/n)')
def main(config, train, val):
    config = Path(config)
    with open(config, 'r') as file:
        cfg = yaml.safe_load(file)
    
    model = YoloModel(cfg)
    if train == 'y':
        model.train()
    elif train == 'n':
        pass
    if val == 'y':
        model.val()
    elif val == 'n':
        pass
    else:
        print("Please specify --train or --val.")

if __name__ == "__main__":
    main()

