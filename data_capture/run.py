import cv2
import yaml
import click
import datetime
import time
import os
from data_capture.realsense import realsense
from pathlib import Path

def show_stream(rs_class):
    """
    Displays the RGB, depth, and background removed streams from the given rs_class object.
    Parameters:
    rs_class (object): The object of the rs_class class.
    Returns:
    None
    """

    rgb = input("Show RGB? (y/n) : ")
    depth = input("Show Depth? (y/n) : ")
    bg_removed = input("Show Background Removed? (y/n) : ")

    rs_class.print_current_params()
    rs_class.start()

    print("Starting stream, press 'q' to quit.")
    while True:
        depth_image, color_image = rs_class.get_frame()

        if rgb == "y":
            cv2.imshow('Color', color_image)

        if depth == "y":
            depth_cm = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            cv2.imshow('Depth', depth_cm)

        if bg_removed == "y":
            background_removed = rs_class.remove_background(depth_image, color_image)
            cv2.imshow('Background Removed', background_removed)        

        if cv2.waitKey(1) == ord('q'):
            break

    rs_class.pipe.stop()

def on_click(rs_class):
    """
    Function that handles the on-click event for the RealSense camera.
    Args:
        rs_class: An instance of the RealSense class.
    Returns:
        None
    """

    rgb = input("Save RGB? (y/n) : ")
    depth = input("Save Depth? (y/n) : ")
    # bg_removed = input("Save Background Removed? (y/n) : ")
    folder_name = input("Enter the folder name to save the images: ")

    # save
    if rgb == "y":
        # os.makedirs(f"{rs_class.save_path}/rgb", exist_ok = True)
        os.makedirs(rs_class.save_path / folder_name / rgb, exist_ok = True)
    if depth == "y":
        # os.makedirs(f"{rs_class.save_path}/depth", exist_ok = True)
        os.makedirs(rs_class.save_path / folder_name / depth, exist_ok = True)

    with open(f"{rs_class.save_path}/rs_config.yaml", "w") as fp:
        yaml.dump(rs_class.config, fp)
    # rs_class.save_params2config(f"{rs_class.save_path}/camera_config.json")
    rs_class.save_params2config(rs_class.save_path / folder_name / "camera_config.json")

    rs_class.print_current_params()
    rs_class.start()

    # cv2.namedWindow("Stream", cv2.WINDOW_AUTOSIZE)

    print("Starting stream, press 'q' to quit.")
    print("Press 'c' to take a picture.")
    while True:
        depth_image, color_image = rs_class.get_frame()

        cv2.imshow("Color", color_image)

        # if bg_removed == "y":
        #     background_removed = rs_class.remove_background(depth_image, color_image)

        key = cv2.waitKey(1) & 0xFF
        timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        if key == ord('c'):
            if rgb == "y":
                # cv2.imwrite(f"{rs_class.save_path}/rgb/{timestamp}.png", color_image)
                cv2.imwrite(rs_class.save_path / folder_name / rgb / f"{timestamp}.png", color_image)

            if depth == "y":
                # cv2.imwrite(f"{rs_class.save_path}/depth/{timestamp}.png", depth_image)
                cv2.imwrite(rs_class.save_path / folder_name / depth / f"{timestamp}.png", depth_image)

            # if bg_removed == "y":
            #     cv2.imwrite(f"{rs_class.save_path}/{timestamp}_bg_removed.png", background_removed)
            print("Image(s) saved.")  

        if key == ord('q'):
            break

    rs_class.pipe.stop()    

def all_frames(rs_class):
    """
    Function to save RGB and depth frames from a RealSense camera.
    Args:
        rs_class: An instance of the RealSense class.
    Returns:
        None
    """

    rgb = input("Save RGB? (y/n) : ")
    depth = input("Save Depth? (y/n) : ")
    # bg_removed = input("Save Background Removed? (y/n) : ")
    folder_name = input("Enter the folder name to save the images: ")

    # save
    if rgb == "y":
        # os.makedirs(f"{rs_class.save_path}/rgb", exist_ok = True)
        os.makedirs(rs_class.save_path / folder_name / rgb, exist_ok = True)
    if depth == "y":
        # os.makedirs(f"{rs_class.save_path}/depth", exist_ok = True)
        os.makedirs(rs_class.save_path / folder_name / depth, exist_ok = True)
    with open(f"{rs_class.save_path}/rs_config.yaml", "w") as fp:
        yaml.dump(rs_class.config, fp)
    rs_class.save_params2config(f"{rs_class.save_path}/camera_config.json")
    
    # start
    rs_class.print_current_params()
    rs_class.start()

    time.sleep(5)
    print("Starting stream, press 'q' to quit.")
    frame_count = 1
    time_start = time.time()
    while True:
        depth_image, color_image = rs_class.get_frame()

        cv2.imshow("Color", color_image)

        timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

        # if bg_removed == "y":
        #     background_removed = rs_class.remove_background(depth_image, color_image)
            # cv2.imwrite(f"{rs_class.save_path}/{timestamp}_{frame_count}_bg_removed.png", background_removed) 
        
        if rgb == "y":
            # cv2.imwrite(f"{rs_class.save_path}/rgb/{timestamp}_{frame_count}.png", color_image)
            cv2.imwrite(rs_class.save_path / folder_name / rgb / f"{timestamp}_{frame_count}.png", color_image)

        if depth == "y":
            # cv2.imwrite(f"{rs_class.save_path}/depth/{timestamp}_{frame_count}.png", depth_image)
            cv2.imwrite(rs_class.save_path / folder_name / depth / f"{timestamp}_{frame_count}.png", depth_image)

        print("Image(s) saved.")  

        frame_count += 1
        # if frame_count == rs_class.fps_rgb:
        #     frame_count = 1


        if cv2.waitKey(1) == ord('q'):
            break

    rs_class.pipe.stop()
    time_end = time.time()
    print(f"Time taken: {time_end - time_start} seconds.")
    
@click.command()
@click.option("--config", "-c", prompt="Enter path to config file")
@click.option("--whatrun", "-w", prompt="Choose the mode you want to run: \n1. Display the stream captured by the camera\n2. Take an image when button is pressed\n3. Capture all frames\nEnter the number of the mode you want to run")
def main(config, whatrun):
    # init rs class
    config = Path(config)
    rs = realsense(config=config)
    
    # what to run
    if whatrun == "1":
        show_stream(rs)

    elif whatrun == "2":
        on_click(rs)

    elif whatrun == "3":
        all_frames(rs)

if __name__ == "__main__":
    main()
