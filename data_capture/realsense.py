import pyrealsense2 as rs
import time
import numpy as np
import json
import yaml
from pathlib import Path

class realsense():
    """ 
    The realsense class represents a RealSense camera and provides methods for configuring and interacting with the camera.
    Attributes:
        advnc_mode (rs.rs400_advanced_mode): The advanced mode object for the RealSense device.
        config (dict): The configuration settings for the camera.
        camera_config (str): The path to the camera configuration file.
        save_path (str): The path to save the captured images.
        rgb_w (int): The width of the RGB stream.
        rgb_h (int): The height of the RGB stream.
        depth_w (int): The width of the depth stream.
        depth_h (int): The height of the depth stream.
        fps_rgb (int): The frame rate of the RGB stream.
        fps_depth (int): The frame rate of the depth stream.
        align_rgb2depth (bool): Flag indicating whether to align the RGB and depth streams.
        clip_dist (float): The clipping distance for background removal.
        background_color (str): The background color for background removal.
        post_process_depth (bool): Flag indicating whether to post-process the depth frames.
        decimation (bool): Flag indicating whether to apply decimation filter.
        decimation_filter_magnitude (int): The magnitude of the decimation filter.
        spatial (bool): Flag indicating whether to apply spatial filter.
        spatial_filter_magnitude (int): The magnitude of the spatial filter.
        spatial_filter_smooth_alpha (float): The alpha value for smoothing in the spatial filter.
        spatial_filter_smooth_delta (float): The delta value for smoothing in the spatial filter.
        spatial_filter_hole_filling (int): The hole filling mode for the spatial filter.
        hole_filling (bool): Flag indicating whether to apply hole filling filter.
        hole_filling_mode (int): The hole filling mode.
        temporal (bool): Flag indicating whether to apply temporal filter.
        temporal_smooth_alpha (float): The alpha value for smoothing in the temporal filter.
        temporal_smooth_delta (float): The delta value for smoothing in the temporal filter.
        temporal_persistency (float): The persistency value for the temporal filter.
        pipe (rs.pipeline): The RealSense pipeline object.
        cfg (rs.config): The RealSense configuration object.
        profile (rs.pipeline_profile): The RealSense pipeline profile object.
        align (rs.align): The RealSense align object.
    Methods:
        __init__(self, config): Initializes the RS class.
        _find_device_that_supports_advanced_mode(self): Finds and returns a device that supports advanced mode from the D400 product line.
        _enable_advanced_mode(self): Enables the advanced mode for the RealSense device.
        _load_json(self, path): Load a JSON file and pass its contents to the advanced mode of the RealSense camera.
        _load_yaml(self, path): Load YAML configuration file and assign values to class attributes.
        _post_process_depth(self, frames): Post-processes the depth frames using various filters.
        save_params2config(self, path): Save the advanced mode parameters to a configuration file.
        print_current_params(self): Print the current parameters of the camera.
        start(self): Starts the RealSense pipeline and configures the streams.
        get_frame(self): Retrieves a depth image and a color image from the RealSense camera.
        remove_background(self, depth_image, color_image): Removes the background from the color image based on the depth image.
    """

    def __init__(self, config):
        """
        Initializes the RS class.
        Args:
            config (str): The path to the configuration file.
        Returns:
            None
        """

        self.advnc_mode = self._enable_advanced_mode()
        config = Path(config)
        self._load_yaml(config)
        if self.camera_config:
            self._load_json(self.camera_config)

    def _find_device_that_supports_advanced_mode(self):
        """
        Finds and returns a device that supports advanced mode from the D400 product line.
        Returns:
            rs.device: The device that supports advanced mode.
        Raises:
            Exception: If no D400 product line device that supports advanced mode was found.
        """

        # https://github.com/IntelRealSense/librealsense/blob/master/wrappers/python/examples/python-rs400-advanced-mode-example.py
        DS5_product_ids = ["0AD1", "0AD2", "0AD3", "0AD4", "0AD5", "0AF6", "0AFE", "0AFF", "0B00", "0B01", "0B03", "0B07", "0B3A", "0B5C", "0B5B"]
        ctx = rs.context()
        ds5_dev = rs.device()
        devices = ctx.query_devices();
        for dev in devices:
            if dev.supports(rs.camera_info.product_id) and str(dev.get_info(rs.camera_info.product_id)) in DS5_product_ids:
                if dev.supports(rs.camera_info.name):
                    print("Found device that supports advanced mode:", dev.get_info(rs.camera_info.name))
                return dev
        raise Exception("No D400 product line device that supports advanced mode was found")
    
    def _enable_advanced_mode(self):
        """
        Enables the advanced mode for the RealSense device.
        Returns:
            rs.rs400_advanced_mode: The advanced mode object if successfully enabled, None otherwise.
        """

        # https://github.com/IntelRealSense/librealsense/blob/master/wrappers/python/examples/python-rs400-advanced-mode-example.py
        try:
            dev = self._find_device_that_supports_advanced_mode()
            advnc_mode = rs.rs400_advanced_mode(dev)
            print("Advanced mode is", "enabled" if advnc_mode.is_enabled() else "disabled")

            # Loop until we successfully enable advanced mode
            while not advnc_mode.is_enabled():
                print("Trying to enable advanced mode...")
                advnc_mode.toggle_advanced_mode(True)
                # At this point the device will disconnect and re-connect.
                print("Sleeping for 5 seconds...")
                time.sleep(5)
                # The 'dev' object will become invalid and we need to initialize it again
                dev = self._find_device_that_supports_advanced_mode()
                advnc_mode = rs.rs400_advanced_mode(dev)
                print("Advanced mode is", "enabled \n" if advnc_mode.is_enabled() else "disabled \n")

            return advnc_mode
        
        except Exception as e:
            print(e)
            pass

    def _load_json(self, path):
        """
        Load a JSON file and pass its contents to the advanced mode of the RealSense camera.
        Args:
            path (str): The path to the JSON file.
        Returns:
            None
        """
        path = Path(path)
        with open(path, "r") as f:
            config = json.load(f)
        json_string = str(config).replace("'", '\"')
        self.advnc_mode.load_json(json_string)
    
    def _load_yaml(self, path):
        """
        Load YAML configuration file and assign values to class attributes.
        Parameters:
        - path (str): The path to the YAML configuration file.
        Returns:
        None
        """

        path = Path(path)
        with open(path) as fp:
            config = yaml.load(fp, Loader=yaml.FullLoader)
        self.config = config
        self.camera_config = Path(config["camera_config"])
        self.save_path = Path(config["save_path"])
        self.rgb_w = config["rgb_width"]
        self.rgb_h = config["rgb_height"]
        self.depth_w = config["depth_width"]
        self.depth_h = config["depth_height"]
        self.fps_rgb = config["fps_rgb"]
        self.fps_depth = config["fps_depth"]
        self.align_rgb2depth = config["align"]
        self.clip_dist = config["clip_dist"]
        self.background_color = config["background_color"]
        self.post_process_depth = config["post_process_depth"]
        self.decimation = config["decimation"]["enabled"]
        self.decimation_filter_magnitude = config["decimation"]["magnitude"]
        self.spatial = config["spatial"]["enabled"]
        self.spatial_filter_magnitude = config["spatial"]["magnitude"]
        self.spatial_filter_smooth_alpha = config["spatial"]["alpha"]
        self.spatial_filter_smooth_delta = config["spatial"]["delta"]
        self.spatial_filter_hole_filling = config["spatial"]["hole_filling"]
        self.hole_filling = config["hole_filling"]["enabled"]
        self.hole_filling_mode = config["hole_filling"]["mode"]
        self.temporal = config["temporal"]["enabled"]
        self.temporal_smooth_alpha = config["temporal"]["alpha"]
        self.temporal_smooth_delta = config["temporal"]["delta"]
        self.temporal_persistency = config["temporal"]["persistency"]

    def _post_process_depth(self, frames):
        """
        Post-processes the depth frames using various filters.
        Args:
            frames (rs.frames): The depth frames to be processed.
        Returns:
            rs.frameset: The processed depth frames as a frameset.
        """

        if self.decimation:
            decimation = rs.decimation_filter()
            decimation.set_option(rs.option.filter_magnitude, self.decimation_filter_magnitude)
            frames = decimation.process(frames)
        
        if self.spatial:
            spatial = rs.spatial_filter()
            spatial.set_option(rs.option.filter_magnitude, self.spatial_filter_magnitude)
            spatial.set_option(rs.option.filter_smooth_alpha, self.spatial_filter_smooth_alpha)
            spatial.set_option(rs.option.filter_smooth_delta, self.spatial_filter_smooth_delta)
            spatial.set_option(rs.option.holes_fill, self.spatial_filter_hole_filling)
            frames = spatial.process(frames)

        if self.temporal:
            temporal = rs.temporal_filter()
            temporal.set_option(rs.option.filter_smooth_alpha, self.temporal_smooth_alpha)
            temporal.set_option(rs.option.filter_smooth_delta, self.temporal_smooth_delta)
            temporal.set_option(rs.option.holes_fill, self.temporal_persistency)
            frames = temporal.process(frames)

        if self.hole_filling:
            hole_filling = rs.hole_filling_filter()
            hole_filling.set_option(rs.option.holes_fill, self.hole_filling_mode)
            frames = hole_filling.process(frames)

        return frames.as_frameset()
        
    def save_params2config(self, path):
        """
        Save the advanced mode parameters to a configuration file.
        Parameters:
        - path (str): The path to the configuration file.
        Returns:
        None
        """

        path = Path(path)
        serialized_string = self.advnc_mode.serialize_json()
        as_json_object = json.loads(serialized_string)
        with open(path, "w") as f:
            json.dump(as_json_object, f, indent=4)

    def print_current_params(self):
        """
        Print the current parameters of the camera.
        Returns:
            None
        """

        # https://github.com/IntelRealSense/librealsense/blob/master/wrappers/python/examples/python-rs400-advanced-mode-example.py
        
        print("Current Parameters: \n")
        print("Depth Control: \n", self.advnc_mode.get_depth_control(), "\n")
        print("RSM: \n", self.advnc_mode.get_rsm(), "\n")
        print("RAU Support Vector Control: \n", self.advnc_mode.get_rau_support_vector_control(), "\n")
        print("Color Control: \n", self.advnc_mode.get_color_control(), "\n")
        print("RAU Thresholds Control: \n", self.advnc_mode.get_rau_thresholds_control(), "\n")
        print("SLO Color Thresholds Control: \n", self.advnc_mode.get_slo_color_thresholds_control(), "\n")
        print("SLO Penalty Control: \n", self.advnc_mode.get_slo_penalty_control(), "\n")
        print("HDAD: \n", self.advnc_mode.get_hdad(), "\n")
        print("Color Correction: \n", self.advnc_mode.get_color_correction(), "\n")
        print("Depth Table: \n", self.advnc_mode.get_depth_table(), "\n")
        print("Auto Exposure Control: \n", self.advnc_mode.get_ae_control(), "\n")
        print("Census: \n", self.advnc_mode.get_census(), "\n")

    def start(self):
        """
        Starts the RealSense pipeline and configures the streams.
        Returns:
            None
        """

        self.pipe = rs.pipeline()
        self.cfg = rs.config()
        self.cfg.enable_stream(rs.stream.color, self.rgb_w, self.rgb_h, rs.format.bgr8, self.fps_rgb)
        self.cfg.enable_stream(rs.stream.depth, self.depth_w, self.depth_h, rs.format.z16, self.fps_depth)
        self.profile = self.pipe.start(self.cfg)
        # depth_sensor, color_sensor, *_ = self.profile.get_device().query_sensors()
        # color_sensor.set_option(rs.option.enable_auto_white_balance, 1)
        # color_sensor.set_option(rs.option.auto_exposure_priority, 1)

        if self.align_rgb2depth:
            align_to = rs.stream.color
            self.align = rs.align(align_to)

    def get_frame(self):
        """
        Retrieves a depth image and a color image from the RealSense camera.
        Returns:
            depth_image (numpy.ndarray): The depth image as a numpy array.
            color_image (numpy.ndarray): The color image as a numpy array.
        """

        frame = self.pipe.wait_for_frames()

        if self.post_process_depth:
            frame = self._post_process_depth(frame)
        
        if self.align_rgb2depth:
            frame = self.align.process(frame)

        depth_frame = frame.get_depth_frame()
        color_frame = frame.get_color_frame()

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        return depth_image, color_image

    def remove_background(self, depth_image, color_image):
        """
        Removes the background from the color image based on the depth image.
        Args:
            depth_image (numpy.ndarray): The depth image.
            color_image (numpy.ndarray): The color image.
        Returns:
            numpy.ndarray: The color image with the background removed.
        """

        if self.align_rgb2depth == False:
            self.align_rgb2depth = True
            align_to = rs.stream.color
            self.align = rs.align(align_to)
            print("align_rgb2depth has been set to True, because it is required for background removal.")

        depth_sensor = self.profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        clipping_distance = self.clip_dist / depth_scale
        depth_image_3d = np.dstack((depth_image, depth_image, depth_image))
        bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), self.background_color, color_image)
        return bg_removed

    def stop(self):
        """
        Stops the pipeline.
        """
    
        self.pipe.stop()

