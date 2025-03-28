import click
import time
import data_capture.run as run_data_capture
import post_processing.run as run_post_processing

def main():
    click.echo("Choose the program you want to run:")
    click.echo("1. Data Capture")
    click.echo("2. Postprocessing")

    choice = input("Enter the number of the program you want to run: ")

    if choice == "1":
        start = time.time()
        run_data_capture.main()
    elif choice == "2":
        start = time.time()
        run_post_processing.main()
    else:
        click.echo("Invalid choice. Please enter a valid number.")
    end = time.time()
    click.echo(f"Time taken: {end - start} seconds.")

if __name__ == "__main__":
    main()