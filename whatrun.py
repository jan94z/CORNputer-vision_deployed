import click
import data_capture.run as run_data_capture
import model_development.run as run_training
import predict.run as run_predict

def main():
    click.echo("Choose the program you want to run:")
    click.echo("1. Data Capture")
    click.echo("2. Training")
    click.echo("3. Prediction")

    choice = input("Enter the number of the program you want to run: ")

    if choice == "1":
        run_data_capture.main()
    elif choice == "2":
        run_training.main()
    elif choice == "3":
        run_predict.main()
    else:
        click.echo("Invalid choice. Please enter a valid number.")

if __name__ == "__main__":
    main()