from myapp import app
from flask import render_template, flash, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
import shutil
import torchvision.models as models
import torch.nn as nn
import cv2  
import torch              
import matplotlib.pyplot as plt                        
from PIL import Image
import torchvision.transforms as transforms



# extract pre-trained face detector
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')


# returns "True" if face is detected in image stored at img_path
def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

# define VGG16 model
VGG16 = models.vgg16(pretrained=True)

# check if CUDA is available
use_cuda = torch.cuda.is_available()

# move model to GPU if CUDA is available
if use_cuda:
    VGG16 = VGG16.cuda()

# Set PIL to be tolerant of image files that are truncated.
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def VGG16_predict(img_path):
    '''
    Use pre-trained VGG-16 model to obtain index corresponding to 
    predicted ImageNet class for image at specified path
    
    Args:
        img_path: path to an image
        
    Returns:
        Index corresponding to VGG-16 model's prediction
    '''
    
    ## TODO: Complete the function.
    ## Load and pre-process an image from the given img_path
    ## Return the *index* of the predicted class for that image
    image = Image.open(img_path).convert('RGB')
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    
    transformations = transforms.Compose([transforms.Resize(size=(224, 224)),
                                         transforms.ToTensor(),
                                         normalize])
    
    transformed_image = transformations(image)[:3,:,:].unsqueeze(0)
    
    if use_cuda:
        transformed_image = transformed_image.cuda()
    
    output = VGG16(transformed_image)
    
    return torch.max(output,1)[1].item() # predicted class index

### returns "True" if a dog is detected in the image stored at img_path
def dog_detector(img_path):
    ## TODO: Complete the function.
    
    return (VGG16_predict(img_path)>=151 and  VGG16_predict(img_path)<= 268) # true/false



# list of class names by index, i.e. a name can be accessed like class_names[0]
class_names = ['Affenpinscher', 'Afghan hound', 'Airedale terrier', 'Akita', \
    'Alaskan malamute', 'American eskimo dog', 'American foxhound', \
        'American staffordshire terrier', 'American water spaniel', 'Anatolian shepherd dog', \
            'Australian cattle dog', 'Australian shepherd', 'Australian terrier', 'Basenji',\
                 'Basset hound', 'Beagle', 'Bearded collie', 'Beauceron', 'Bedlington terrier',\
                      'Belgian malinois', 'Belgian sheepdog', 'Belgian tervuren',\
                           'Bernese mountain dog', 'Bichon frise', 'Black and tan coonhound', \
                               'Black russian terrier', 'Bloodhound', 'Bluetick coonhound', \
                                   'Border collie', 'Border terrier', 'Borzoi', 'Boston terrier',\
                                        'Bouvier des flandres', 'Boxer', 'Boykin spaniel', 'Briard', 'Brittany', \
                                            'Brussels griffon', 'Bull terrier', 'Bulldog', 'Bullmastiff', 'Cairn terrier', 'Canaan dog', 'Cane corso', 'Cardigan welsh corgi', 'Cavalier king charles spaniel', 'Chesapeake bay retriever', 'Chihuahua', 'Chinese crested', 'Chinese shar-pei', 'Chow chow', 'Clumber spaniel', 'Cocker spaniel', 'Collie', 'Curly-coated retriever', 'Dachshund', 'Dalmatian', 'Dandie dinmont terrier', 'Doberman pinscher', 'Dogue de bordeaux', 'English cocker spaniel', 'English setter', 'English springer spaniel', 'English toy spaniel', 'Entlebucher mountain dog', 'Field spaniel', 'Finnish spitz', 'Flat-coated retriever', 'French bulldog', 'German pinscher', 'German shepherd dog', 'German shorthaired pointer', 'German wirehaired pointer', 'Giant schnauzer', 'Glen of imaal terrier', 'Golden retriever', 'Gordon setter', 'Great dane', 'Great pyrenees', 'Greater swiss mountain dog', 'Greyhound', 'Havanese', 'Ibizan hound', 'Icelandic sheepdog', 'Irish red and white setter', 'Irish setter', 'Irish terrier', 'Irish water spaniel', 'Irish wolfhound', 'Italian greyhound', 'Japanese chin', 'Keeshond', 'Kerry blue terrier', 'Komondor', 'Kuvasz', 'Labrador retriever', 'Lakeland terrier', 'Leonberger', 'Lhasa apso', 'Lowchen', 'Maltese', 'Manchester terrier', 'Mastiff', 'Miniature schnauzer', 'Neapolitan mastiff', 'Newfoundland', 'Norfolk terrier', 'Norwegian buhund', 'Norwegian elkhound', 'Norwegian lundehund', 'Norwich terrier', 'Nova scotia duck tolling retriever',\
                                             'Old english sheepdog', 'Otterhound', 'Papillon', 'Parson russell terrier', 'Pekingese', 'Pembroke welsh corgi', 'Petit basset griffon vendeen', 'Pharaoh hound', 'Plott', 'Pointer', 'Pomeranian', 'Poodle', 'Portuguese water dog', 'Saint bernard', 'Silky terrier', 'Smooth fox terrier', 'Tibetan mastiff', 'Welsh springer spaniel', 'Wirehaired pointing griffon', 'Xoloitzcuintli', 'Yorkshire terrier']

model_transfer = models.resnet101(pretrained=True)

if use_cuda:
    model_transfer = model_transfer.cuda()



for param in model_transfer.parameters():
    param.requires_grad = False
    
model_transfer.fc = nn.Linear(2048, 133, bias=True)

if use_cuda:
    model_transfer = model_transfer.cuda()



model_transfer.load_state_dict(torch.load('model_transfer.pt', map_location=torch.device('cpu')))

def predict_breed_transfer(img_path):
    # load the image and return the predicted breed
    image = Image.open(img_path).convert('RGB')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    transformations = transforms.Compose([transforms.Resize(size=(224, 224)), transforms.ToTensor(), normalize])
    
    transformed_image = transformations(image)[:3,:,:].unsqueeze(0)
    
    if use_cuda:
        transformed_image = transformed_image.cuda()
    
    output = model_transfer(transformed_image)
    
    pred_index = torch.max(output,1)[1].item()
    
#     return class_names[pred_index]
    return class_names[pred_index]

### TODO: Write your algorithm.
### Feel free to use as many code cells as needed.

def show_image(img_path):
    img = Image.open(img_path)
    plt.imshow(img)
    plt.show()
    
def run_app(img_path):
    ## handle cases for a human face, dog, and neither
   
    if face_detector(img_path):
        
        predicted_breed = predict_breed_transfer(img_path)
        
        return ["Hi Human!", img_path, "You look like a: "+ predicted_breed + "\n"]

    elif dog_detector(img_path):
       
        predicted_breed = predict_breed_transfer(img_path)
        
        return ["Hi Dog!", img_path, "You look like a: "+ predicted_breed + "\n"]
       
    else:
        return ["Image is invalid", img_path, " "]
















@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/predict_from_location', methods=['POST'])
def predict_from_location():
    return render_template('prediction.html', prediction = run_app(request.form['projectFilePath']))


@app.route('/predict_from_file', methods=['POST'])
def predict_from_file():
   if request.method == 'POST':
      f = request.files['file']
      f.save(secure_filename(f.filename))
      prediction = run_app(f.filename)
      prediction[1] = f.filename
      shutil.copy(f.filename, 'myapp/static')
      return render_template('prediction.html', prediction = prediction)

