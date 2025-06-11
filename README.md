# Computer-Vision
## Description
This is the repository github of the Computer Vision part of the TRAVELERS projet (2024-25 edition).
The implementation was done with a Raspberry Pi 5 branched to a PiCamera 3. 
The Goal here was to implement an optimised YOLO model on the raspberry Pi while permitting a live-transmission of the model with an acceptable number of FPS (>10 FPS) 
The objectives : 
- High resolution
- Optimized format of output
- Optimized format of YOLO model
- Optimized fine-tuning of the model (through the roboflow plateform) implemented correctly on the rasp.

## Instructions
Supposedly you have the computer_vision/requirements.txt file which gives the full set of instructions to implement a usable python venv on your rasp with ultralytics in it.
Ultralytics is the library where you will find the YOLO library, opencv for live-reading of input of the picamera for example.
<pre lang="markdown">pip install -r requirements.txt </pre>
This install the to-get-started dependencies in a venv that you will use for computer vision/

However here are each instructions in this file as well as their explanations.

I advise you to work on the cmd of an IDE such as VS-code. 
On it you will now put the following commands

To use without problems libraries such as ultralytics without taking out the restrains, you have to create a virtual python virtual environment : 
<pre lang="markdown"> bash python3 -m venv --system-site-packages yolo_object </pre>

You can enter into it by using the bin/activate of this venv at anytime. after making sure you are in the same file as this virtual env, because it is a folder that you installed.
_Command to enter the venv_
<pre lang="markdown"> source yolo_object/bin/activate </pre>

To make sure that you can use pip in a non-obsolete version of python
<pre lang="markdown">sudo apt update
sudo apt install python3-pip -y
pip install -U pip </pre>

You will now install the ultralytics 

<pre lang="markdown"> pip install ultralytics[export] </pre>

### Use of the code already implemented

To understand how to get started in computer vision you can go and see the core-electronics tutorial
https://core-electronics.com.au/guides/raspberry-pi/getting-started-with-yolo-object-and-animal-recognition-on-the-raspberry-pi/

