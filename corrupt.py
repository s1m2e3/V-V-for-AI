import torch
from IPython.display import Image  # for displaying images
import os 
import random
import time
import shutil
import sys


from itertools import product
 
import pkg_resources
from sklearn.model_selection import train_test_split
import xml.etree.ElementTree as ET
from xml.dom import minidom
from tqdm import tqdm
from PIL import Image, ImageDraw
import numpy as np

import argparse
import shutil
import yaml
import copy
import subprocess

random.seed(108)


import imagecorruptions
#https://github.com/bethgelab/imagecorruptions



# Function to get the data from XML Annotation
def extract_info_from_xml(xml_file):
    root = ET.parse(xml_file).getroot()
    
    # Initialise the info dict 
    info_dict = {}
    info_dict['bboxes'] = []

    # Parse the XML Tree
    for elem in root:
        # Get the file name 
        if elem.tag == "filename":
            info_dict['filename'] = elem.text
            
        # Get the image size
        elif elem.tag == "size":
            image_size = []
            for subelem in elem:
                image_size.append(int(subelem.text))
            
            info_dict['image_size'] = tuple(image_size)
        
        # Get details of the bounding box 
        elif elem.tag == "object":
            bbox = {}
            for subelem in elem:
                if subelem.tag == "name":
                    bbox["class"] = subelem.text
                    
                elif subelem.tag == "bndbox":
                    for subsubelem in subelem:
                        bbox[subsubelem.tag] = int(subsubelem.text)            
            info_dict['bboxes'].append(bbox)
    
    return info_dict

def update_xml(xml_file,new_image):
    
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    # Parse the XML Tree
    for elem in root:
       
        # Get the image size
        if elem.tag == "size":
            elem.set("width",str(new_image.shape[0]))
            elem.set("height",str(new_image.shape[1]))
    tree.write(xml_file)  
   

# Convert the info dict to the required yolo format and write it to disk
def convert_to_yolov5(info_dict):
    print_buffer = []
    
    # For each bounding box
    for b in info_dict["bboxes"]:
        try:
            class_id = class_name_to_id_mapping[b["class"]]
        except KeyError:
            print("Invalid Class. Must be one from ", class_name_to_id_mapping.keys())
        
        # Transform the bbox co-ordinates as per the format required by YOLO v5
        b_center_x = (b["xmin"] + b["xmax"]) / 2 
        b_center_y = (b["ymin"] + b["ymax"]) / 2
        b_width    = (b["xmax"] - b["xmin"])
        b_height   = (b["ymax"] - b["ymin"])
        
        # Normalise the co-ordinates by the dimensions of the image
        image_w, image_h, image_c = info_dict["image_size"]  
        b_center_x /= image_w 
        b_center_y /= image_h 
        b_width    /= image_w 
        b_height   /= image_h 
        
        #Write the bbox details to the file 
        print_buffer.append("{} {:.3f} {:.3f} {:.3f} {:.3f}".format(class_id, b_center_x, b_center_y, b_width, b_height))
        
    # Name of the file which we have to save 
    
    save_file_name = os.path.join("annotations", info_dict["filename"].replace("png", "txt"))
    
    # Save the annotation to disk
    print("\n".join(print_buffer), file= open(save_file_name, "w"))
#Utility function to move images 
def move_files_to_folder(list_of_files, destination_folder):
    os.mkdir(destination_folder)
    for f in list_of_files:
        try:
            shutil.move(f, destination_folder)
        except:
            print(f)
            assert False

def resize(image_array,ratio):
        
        image = Image.fromarray(image_array, 'RGB')
        width = image.size[0]
        height = image.size[1]
        newWidth = int(round(width * ratio))
        newHeight = int(round(height * ratio))
        newImage = image.resize((newWidth, newHeight), Image.Resampling.LANCZOS)
        newImage.format = image.format
        
        return np.asarray(newImage)

def extra_corruption(image, corruption_name,annotation):
    
    if "resolution_change" in corruption_name:
        if "2x" in corruption_name:
            ratio = 2.0
        elif "x_2" in corruption_name:
            ratio=0.5
        new_image = resize(image,ratio)
        update_xml(annotation,new_image)
        return new_image
        

    elif "lense_crush" in corruption_name:
        
        width = np.arange(image.shape[0])
        height = np.arange(image.shape[1])
        depth = np.arange(image.shape[2])
        indices = list(product(width,height,depth))
        size = int(0.1*image.shape[0]*image.shape[1]*image.shape[2])
        
        if "gaussian" in corruption_name:

            indices = random.choices(indices,k=size)
            new_image = image
            for index in indices:
                new_image[index]=0
            return new_image

        if "directed" in corruption_name:
            choose_center =  random.choices(indices,k=1)[0]
           
            up = choose_center[0]
            right = choose_center[1]
            up_max = image.shape[0]-up>image.shape[0]/4
            up_min = up>100
            right_max = image.shape[1]-right>image.shape[1]/3
            right_min = right >100
            
            
            while not up_max or not up_min or not right_max or not right_min:
                choose_center =  random.choices(indices,k=1)[0]

                up = choose_center[0]
                right = choose_center[1]
                up_max = image.shape[0]-up>image.shape[0]/4
                up_min = up>100
                right_max = image.shape[1]-right>image.shape[1]/3
                right_min = right >100
                
            indices_up = np.arange(int(up-image.shape[0]/4),int(up+image.shape[0]/4))
            indices_right = np.arange(int(right-image.shape[1]/3),int(right+image.shape[1]/3))
            indices = list(product(indices_up,indices_right,depth))
            indices = random.choices(indices,k=size)
            
            for index in indices:
                image[index] = 0
            new_image = image
            return new_image
    
    
        


def corrupt_dataset(corruption_name,train_images,val_images,test_images,train_annotations,val_annotations,test_annotations,severity=4,apply_on_train=True,apply_on_test=True):
    
    train_names = copy.deepcopy(train_images)
    val_names = copy.deepcopy(val_images)
    test_names = copy.deepcopy(test_images)
    
    
    if apply_on_train:

        train_images = [np.array(Image.open(image)) for image in train_images]
        train_images = [np.array(Image.fromarray(np.uint8(image)).convert('RGB')) for image in train_images]
        if corruption_name in imagecorruptions.get_corruption_names():
            train_images = [imagecorruptions.corrupt(image,corruption_name=corruption_name,severity=3) for image in train_images]
        else:
            print("in extra corruption")
            train_images = [extra_corruption(train_images[i],corruption_name,train_annotations[i]) for i in range(len(train_images))]
            print("finished extra corruption")
        for i in range(len(train_images)):
            image = Image.fromarray(train_images[i].astype('uint8'), 'RGB')
            image.save(train_names[i],"PNG")

        val_images = [np.array(Image.open(image)) for image in val_images]
        val_images = [np.array(Image.fromarray(np.uint8(image)).convert('RGB')) for image in val_images]
        
        if corruption_name in imagecorruptions.get_corruption_names():
            val_images = [imagecorruptions.corrupt(image,corruption_name=corruption_name,severity=3) for image in val_images]
        else:
            val_images = [extra_corruption(val_images[i],corruption_name,val_annotations[i]) for i in range(len(val_images))]
            

        for i in range(len(val_images)):
            image = Image.fromarray(val_images[i].astype('uint8'), 'RGB')
            image.save(val_names[i],"PNG")

    if apply_on_test:
        
        test_images = [np.array(Image.open(image)) for image in test_images]
        test_images = [np.array(Image.fromarray(np.uint8(image)).convert('RGB')) for image in test_images]
        
        
        if corruption_name in imagecorruptions.get_corruption_names():
            test_images = [imagecorruptions.corrupt(image,corruption_name=corruption_name,severity=severity) for image in test_images]
        else:
            test_images = [extra_corruption(test_images[i],corruption_name,test_annotations[i]) for i in range(len(test_images))]
            

        
        
        for i in range(len(test_images)):
            image = Image.fromarray(test_images[i].astype('uint8'), 'RGB')
            image.save(test_names[i],"PNG")

        if not apply_on_train:

            val_images = [np.array(Image.open(image)) for image in val_images]
            val_images = [np.array(Image.fromarray(np.uint8(image)).convert('RGB')) for image in val_images]
            
            
            if corruption_name in imagecorruptions.get_corruption_names():
                val_images = [imagecorruptions.corrupt(image,corruption_name=corruption_name,severity=severity) for image in val_images]
            else:
                val_images = [extra_corruption(val_images[i],corruption_name,val_annotations[i]) for i in range(len(val_images))]
                

            
            for i in range(len(val_images)):
                image = Image.fromarray(val_images[i].astype('uint8'), 'RGB')
                image.save(val_names[i],"PNG")
            
    
def convert_annotations(annotations):

    for ann in tqdm(annotations):
        info_dict = extract_info_from_xml(ann)
        convert_to_yolov5(info_dict)

def creater_parser():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--corruption_name', type=str, default="gaussian_blur" ,help='name of the corruption as defined by the imagecorruptions library')
    parser.add_argument('--severity', type=int, default=1, help='severity of the corruption')
    parser.add_argument('--apply_on_train', type=bool, default=False , help='dataset.yaml path')
    parser.add_argument('--apply_on_test', type=bool, default=False , help='dataset.yaml path')

    return parser.parse_args() 



if __name__ == "__main__":
    
    parser = creater_parser()
    
    # Dictionary that maps class names to IDs
    class_name_to_id_mapping = {"trafficlight": 0,
                           "stop": 1,
                           "speedlimit": 2,
                           "crosswalk": 3}    
    
    #create folder for corrupted images
    os.chdir("../Road_Sign_Dataset")
    currentdir = os.getcwd()
    
    added = "_"+parser.corruption_name
    # added = "_"+parser.corruption_name +"_train" if parser.apply_on_train else "_"+parser.corruption.name
    # added = added + str(int(parser.severity))
    # added = added + "_test" if parser.apply_on_test else added 
    path = currentdir + added
    
    os.mkdir(path)
    os.chdir("../"+"Road_Sign_Dataset"+added)
    #copy images from original folder
    shutil.copytree(currentdir+"\\images", path+"\\images")
    shutil.copytree(currentdir+"\\annotations", path+"\\annotations")
    
    
    
    time.sleep(5)
    # Read images and annotations
    images = [os.path.join('images', x) for x in os.listdir('images')]
    annotations = [os.path.join('annotations', x) for x in os.listdir('annotations') if x[-3:] == "xml"]
    annotations.sort()
    

    images.sort()
    annotations.sort()

    # Split the dataset into train-valid-test splits 
    train_images, val_images, train_annotations, val_annotations = train_test_split(images, annotations, test_size = 0.2, random_state = 1)
    val_images, test_images, val_annotations, test_annotations = train_test_split(val_images, val_annotations, test_size = 0.5, random_state = 1)
    print("start_corruption")
    corrupt_dataset(parser.corruption_name,train_images,val_images,test_images,train_annotations,val_annotations,test_annotations)
    print("finish_corruption")

    # Convert and save the annotations
    convert_annotations(annotations)
    train_annotations = map(lambda string: string.replace(".xml",".txt") ,train_annotations)
    val_annotations = map(lambda string: string.replace(".xml",".txt"),val_annotations)
    test_annotations = map(lambda string: string.replace(".xml",".txt"),test_annotations)


    move_files_to_folder(train_images, 'images/train')
    move_files_to_folder(val_images, 'images/val/')
    move_files_to_folder(test_images, 'images/test/')
    move_files_to_folder(train_annotations, 'annotations/train/')
    move_files_to_folder(val_annotations, 'annotations/val/')
    move_files_to_folder(test_annotations, 'annotations/test/')

    currentdir = os.getcwd()
    os.rename(currentdir+"/annotations",currentdir+"/labels")
    os.chdir("../yolov5/data")

    with open("coco128.yaml", "r", encoding='utf8') as stream:
        try:

            data=yaml.safe_load(stream)
            data["path"] = path
            data['train'] = path+"/images/train/" 
            data["val"] = path+"/images/val/"
            data["test"] = path+"/images/test/"
            data["nc"] = 4
            data["names"] = ["trafficlight","stop", "speedlimit","crosswalk"]
    
        except yaml.YAMLError as exc:
            print(exc)

    name= "road_sign_data" + added + ".yaml"
    stream = open(name, 'w')
    yaml.dump(data, stream)
    stream.close()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name_train = "python train.py --img 640 --cfg yolov5s.yaml --hyp hyp.no-augmentation.yaml --batch 16 --epochs 100 --data "+ name +" --weights yolov5s.pt --workers 24 --name yolo_road_det"+added
    os.chdir("../")
    
    name_infer = "python detect.py --source ../Road_Sign_Dataset"+added+"/images/test --weights runs/train/"+"yolo_road_det"+added+"/weights/best.pt --conf 0.25 --name yolo_road_det"+added
    name_val = "python val.py --data "+name +" --weights runs/train/yolo_road_det"+ added + "/weights/best.pt --img 640 --task test"
    subprocess.call(name_train)
    subprocess.call(name_infer)
    subprocess.call(name_val)
    print("finished_testing")
    