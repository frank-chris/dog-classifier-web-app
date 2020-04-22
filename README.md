# Dog Breed Classification using CNN  

## Description of files/folders  

### 1. dog_app.ipynb    
Jupyter notebook with code showing the creation, training, validation and testing of the CNN models and the dog and human face detectors. Also has the function for applying the code on images. 

### 2. report.html    
HTML export of the Jupyter notebook.

### 3. report.pdf   
PDF export of the Jupyter notebook.  

### 4. proposal.pdf   
PDF file of the Capstone Proposal.   

### 5. project_report.pdf   
PDF file of the Capstone Project Report.   

### 6. Test_Images   
The folder containing images that I used at the end of the Jupyter notebook for trying the model on user-supplied images.  

### 7. myapp   
The folder containing the code for the web app made using Flask. It couldn't be done on the notebook as the connection to the webpage was not working on the notebook. 


## Instructions for CNN   

The instructions for the creation, training, validation and testing of the CNN models and the dog and human face detectors and the function for trying the trained model on images are present in the Jupyter Notebook named dog_app.ipynb

## Instructions for running the web app   

### 1. Download the folder named myapp

### 2. Install the following Python libraries:   

1. Flask

  `pip install Flask`

2. Werkzeug

  `pip install Werkzeug`

3. PyTorch

  `pip install torch torchvision`

4. OpenCV

  `pip install opencv-python`

5. Matplotlib

  `pip install matplotlib`

6. Pillow

  `pip install Pillow`

Note: The web app also uses the following preinstalled libraries:   
  1. os  
  2. shutil   
  3. atexit   


### 3. Open a terminal in the folder myapp

### 4. Run the python script named routes.py

   `python3 routes.py`  

### 5. Open a web browser and navigate to the specified URL.


## Video  
[Here is a link to a Demo video of the web app](https://youtu.be/duUZCrp7msc)


## Description of files/folders in myapp folder 


### 1. \_\_pycache\_\_    
Folder containing cache.

### 2. haarcascades  
Folder containing OpenCV's Haar cascades based classifier.  


### 3. static  
Folder required by the Flask web app to store static content like images. 

### 4. templates  
Folder required by the Flask web app to store the HTML files of the webpages.  

### 5. model_transfer.pt  
Trained PyTorch model(transfer learning).

### 6. routes.py  
Python script with code for the web app. The code is mostly the same as the Jupyter Notebook, except for a few Flask functions at the end. 

### 7. routes.pyc  
Just a compiled file of routes.py


## Important Note

These instructions were written for a Linux OS. 

## Datasets   

[Dog Images Dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip)     
[Human Images Dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip)
