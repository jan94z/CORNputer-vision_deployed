# CORNputer Vision Repository 🌽

Welcome to the **CORNputer Vision** repository! This project forms the backbone of my master's thesis, where I developed a camera- and computer vision-powered system for maize seed analysis.

The required Python Version is **Python 3.11**.

# UBUNTU GUIDE
## 🚀 Installation
Follow these steps to set up the repository on **Ubuntu 20.04**.

### 🔧 Realsense Library
To capture images using a Realsense camera, install the Realsense library following the [Realsense Installation Guide](https://dev.intelrealsense.com/docs/compiling-librealsense-for-linux-ubuntu-guide).

### 📦 Install Pre-requisites
Run the following commands in the root directory of this repository to install necessary pre-requisites:

```bash
sudo apt-get update
sudo apt-get -y upgrade
sudo apt-get install -y python3-pip
sudo apt-get install build-essential libssl-dev libffi-dev python-dev
sudo apt-get install -y python3-venv
```

### 🐍 Python Virtual Environment Setup
Execute the following commands in the root directory of this repository to set up the Python virtual environment:

```bash
python3 -m venv ".venv"
source .venv/bin/activate
pip install --upgrade pip
```

To deactivate the environment, simply run:

```bash
deactivate
```

### 📥 Install required python packages
Execute the following command in the root directory of this repository to install the required Python packages:

```bash
pip install -r requirements.txt
```

## 🛠️ Usage

# WINDOWS GUIDE
## 🚀 Installation on Windows

Follow these steps to set up the repository on **Windows 10/11** using **Python 3.11**.

### 🔧 Realsense Library

To use a Realsense camera on Windows, install the **Intel RealSense SDK** from the official site:  
👉 [Intel Realsense SDK Releases](https://github.com/IntelRealSense/librealsense/releases)

1. Download the latest `.exe` installer under **Assets**
2. Install the SDK and plug in your Realsense camera

### 🐍 Python Virtual Environment Setup

Open PowerShell or Command Prompt in the root directory of the repository and run:

```bash
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
```

To deactivate the environment:

```bash
deactivate
```

### 📥 Install required Python packages

Once the virtual environment is activated, install the required packages with:

```bash
pip install -r requirements.txt
```


