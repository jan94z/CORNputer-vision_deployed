import click
import time

def tracking():
    pass

def classification_broken():
    pass

def classification_tip():
    pass

def size_estimation():
    pass

@click.command()
@click.option("--config", "-c", prompt="Enter the path to the config file.")
@click.option("--data", "-d", prompt="Enter the path to the data that you want to process.")
@click.option("--name", "-n", prompt="Enter the name of the folder you want to save the processed data in.")
@click.option("--whatrun", "-w", prompt="Choose the program you want to run: \n1. Tracking\n2. Classification (broken/intact)\n3. Classification (tip/no tip)\n4. Size estimation\n5. All programs\nEnter the number of the mode you want to run.")
def main(config, data, name, whatrun):
    start = time.time()
    if whatrun == "1":
        tracking()
    elif whatrun == "2":
        classification_broken()
    elif whatrun == "3":
        classification_tip()
    elif whatrun == "4":
        size_estimation()
    elif whatrun == "5":
        tracking()
        classification_broken()
        classification_tip()
        size_estimation()
    else:
        click.echo("Invalid choice. Please enter a valid number.")



    end = time.time()
    click.echo(f"Time taken: {end - start} seconds.")

if __name__ == "__main__":
    click.echo("Welcome to the post-processing program!")
    main()