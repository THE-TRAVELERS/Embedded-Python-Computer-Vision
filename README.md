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
<pre lang="markdown"> python3 -m venv --system-site-packages yolo_object </pre>

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

You can test the app.py to see how the yolov8n model works, you will see a very small amount of fps.
put "yolov8n.pt" in the argument of the model in the code, then do (in the folder of the code)
<pre lang="markdown"> python app.py </pre>

Then, you will see that even the yolov8n doesn't work well
I advise you to keep using a spark server as I do to print the output, to avoid unneccesary complication with the output of the picamera and disfonctionalities with certain libraries.

you will now export a ncnn model version of your yolov8n.pt that was installed during the execution of the python script.
<pre lang="markdown"> python ncnn_export.py </pre>
I have put the .pt of yolov8n as the argument and the resolution of 320 for the model, but you can change it with 640 for example.

This works with any .pt of yolo model
you simply need to change the argument in the app.py with "yolov8n_ncnn_model" so that it uses the folder with the weights, args of your ncnn model

You can also install the yolov11 and yolov12 directly with git.
use this on a notebook and you will get the versions you want
<pre lang="markdown"> !wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt -O yolov8n.pt
!wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-v11.pt -O yolov8n-v11.pt
!wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-v12.pt -O yolov8n-v12.pt </pre>

or with code such as 
<pre lang="markdown">
import requests

url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"
output_path = "yolov8n.pt"

response = requests.get(url)
with open(output_path, "wb") as f:
    f.write(response.content)

print("Téléchargement terminé.")
</pre>


check the existance of your .pt with : 
<pre lang="markdown"> !ls -lh yolov8n.pt </pre>

