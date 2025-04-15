import torch
import torchvision
import torchvision.transforms.functional
from torchviz import make_dot
import model
from model import common
import cv2 as cv
import numpy as np
import os
from tkinter import filedialog
import random
from math import ceil, floor
import pandas as pd
from imgaug import augmenters as iaa

DEBUG=False
TEST=True
TRAIN=False
TRAIN_TYPE = 3 # 0 = weighted ROI ; 1 = black inverted ROI ; 2 = traditional MSE ; 3 - binary output
DNN_TYPE = 0 # 0 = EDSR ; 1 = UNET

IMG_WIDTH = int(1920/2)
IMG_HEIGHT = int(1080/2)
IMG_CHANNELS = 3
input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

SOURCE_FOLDER = filedialog.askdirectory(title='Choose Parent Folder',initialdir=os.getcwd())
#SOURCE_FOLDER = 'C:/Users/guilherme.franco/Documents/GAF/Estagio_Rui_Brito'
training_history_filename = SOURCE_FOLDER+'/Results/log.csv'

seed=420
SPLIT = 0.895
EPOCHS = 500
BATCH_SIZE=4
LEARNING_RATE = 10e-5

### Setup Dataset ###############################################################################################

raw_df = pd.read_csv(SOURCE_FOLDER+'/Lens_Dataset/data.csv').sample(frac=1)  #(read and shuffle)

raw_size = int(len(raw_df))

train_df = raw_df.head(floor(int(raw_size*SPLIT)))
train_size = int(len(train_df))

val_df = raw_df.tail(ceil(int(raw_size*(1-SPLIT))))
val_size = int(len(val_df))

train_df = train_df.sample(frac=1)
val_df = val_df.sample(frac=1)

#################################################################################################################

def yield_training_batch(img_file_list, corners_list):
        
    batch_size = len(img_file_list)

    input_images = []
    output_images = []
    batch_corners = []

    for f in range(0,batch_size):

        # Read image from list and convert to array
        input_image_path = SOURCE_FOLDER+'/Lens_Dataset/INPUTS/'+str(img_file_list[f])+'.png'
        output_image_path = SOURCE_FOLDER+'/Lens_Dataset/OUTPUTS/'+str(img_file_list[f])+'.png'
        
        input_image = cv.imread(input_image_path)
        output_image = cv.imread(output_image_path)

        cornerTL_x = corners_list[f][0]
        cornerTL_y = corners_list[f][1]
        cornerTR_x = corners_list[f][2]
        cornerTR_y = corners_list[f][3]
        cornerBL_x = corners_list[f][4]
        cornerBL_y = corners_list[f][5]
        cornerBR_x = corners_list[f][6]
        cornerBR_y = corners_list[f][7]


        if DEBUG:
            debug_img = output_image.copy()
            cv.circle(debug_img,(int(cornerTL_x),int(cornerTL_y)),10,(0,0,0),-1)
            cv.circle(debug_img,(int(cornerTL_x),int(cornerTL_y)),6,(255,255,255),-1)

            cv.circle(debug_img,(int(cornerTR_x),int(cornerTR_y)),10,(0,0,0),-1)
            cv.circle(debug_img,(int(cornerTR_x),int(cornerTR_y)),6,(255,255,255),-1)

            cv.circle(debug_img,(int(cornerBL_x),int(cornerBL_y)),10,(0,0,0),-1)
            cv.circle(debug_img,(int(cornerBL_x),int(cornerBL_y)),6,(255,255,255),-1)

            cv.circle(debug_img,(int(cornerBR_x),int(cornerBR_y)),10,(0,0,0),-1)
            cv.circle(debug_img,(int(cornerBR_x),int(cornerBR_y)),6,(255,255,255),-1)

            cv.imshow('PRE DEBUG OUTPT',  cv.resize(debug_img,(960,720)))


        #generate random vector to determine which augmentations to perform
        R = []
        for r in range(0,7):
            if DEBUG:
                R.append(0)
            else:
                rn = random.random()
                R.append(rn)

        #======ZOOM AND CROP OR NOT?======#

        if R[0] < 0.3:
            
            aspect_ratio = input_image.shape[1]/input_image.shape[0]
            target_ratio = IMG_WIDTH/IMG_HEIGHT
            height_center = int(input_image.shape[0]/2)
            width_center = int(input_image.shape[1]/2)

            if aspect_ratio > IMG_WIDTH/IMG_HEIGHT:
                target_width = int(input_image.shape[0]*target_ratio)
                input_image = input_image[0:input_image.shape[0], width_center-floor(target_width/2):width_center+floor(target_width/2)]
                output_image = output_image[0:input_image.shape[0], width_center-floor(target_width/2):width_center+floor(target_width/2)]
                
            elif aspect_ratio < IMG_WIDTH/IMG_HEIGHT:
                target_height = int(input_image.shape[1]*target_ratio)
                input_image = input_image[height_center-floor(target_height/2):height_center+floor(target_height/2), 0:input_image.shape[1]]
                output_image = output_image[height_center-floor(target_height/2):height_center+floor(target_height/2), 0:input_image.shape[1]]

            rn = random.uniform(1,1)    #how much to zoom in
            
            crop_w = floor(IMG_WIDTH*rn)    #how much to crop
            crop_h = floor(IMG_HEIGHT*rn)
            
            w_space = IMG_WIDTH - crop_w    #how much space is left
            h_space = IMG_HEIGHT - crop_h
            
            random_center_x = int(crop_w/2) + random.randint(0,w_space)     #center of the zoom/crop
            random_center_y = int(crop_h/2) + random.randint(0,h_space)
            
            #crops according to random parameters and resizes to desired input/output size
            input_image = input_image[random_center_y-int(crop_h/2):random_center_y+int(crop_h/2),random_center_x-int(crop_w/2):random_center_x+int(crop_w/2)]
            input_image = cv.resize(input_image, (IMG_WIDTH, IMG_HEIGHT), interpolation = cv.INTER_CUBIC)
            output_image = output_image[random_center_y-int(crop_h/2):random_center_y+int(crop_h/2),random_center_x-int(crop_w/2):random_center_x+int(crop_w/2)]
            output_image = cv.resize(output_image, (IMG_WIDTH, IMG_HEIGHT), interpolation = cv.INTER_NEAREST)


            # TODO: estas contas provavelmente estão mal, pelo que 'desabilitei o random crop e zoom'
            cornerTL_x = (cornerTL_x-(random_center_x - (IMG_WIDTH/2)))*rn
            cornerTL_y = (cornerTL_y-(random_center_y - (IMG_HEIGHT/2)))*rn
            cornerTR_x = (cornerTR_x-(random_center_x - (IMG_WIDTH/2)))*rn
            cornerTR_y = (cornerTR_y-(random_center_y - (IMG_HEIGHT/2)))*rn
            cornerBL_x = (cornerBL_x-(random_center_x - (IMG_WIDTH/2)))*rn
            cornerBL_y = (cornerBL_y-(random_center_y - (IMG_HEIGHT/2)))*rn
            cornerBR_x = (cornerBR_x-(random_center_x - (IMG_WIDTH/2)))*rn
            cornerBR_y = (cornerBR_y-(random_center_y - (IMG_HEIGHT/2)))*rn

        else:

            input_image = cv.resize(input_image, (IMG_WIDTH, IMG_HEIGHT), interpolation = cv.INTER_CUBIC)
            output_image = cv.resize(output_image, (IMG_WIDTH, IMG_HEIGHT), interpolation = cv.INTER_CUBIC)
            
            

        #======FLIP OR NOT?======#

        #Flip Horizontaly
        if R[1] < 0.5:
            input_image = cv.flip(input_image, 1)
            output_image = cv.flip(output_image, 1)
            cornerTL_x = (IMG_WIDTH) - cornerTL_x
            cornerTR_x = (IMG_WIDTH) - cornerTR_x
            cornerBL_x = (IMG_WIDTH) - cornerBL_x
            cornerBR_x = (IMG_WIDTH) - cornerBR_x
        
        #Flip Verticaly
        if R[2] < 0.5:
            input_image = cv.flip(input_image, 0)
            output_image = cv.flip(output_image, 0)
            cornerTL_y = (IMG_HEIGHT) - cornerTL_y
            cornerTR_y = (IMG_HEIGHT) - cornerTR_y
            cornerBL_y = (IMG_HEIGHT) - cornerBL_y
            cornerBR_y = (IMG_HEIGHT) - cornerBR_y
        
        #Color Shift
        if R[3] < 0.3:
            seq = iaa.Sequential([iaa.MultiplyHueAndSaturation((0.8, 1.2), per_channel=True)])
            input_image = seq.augment_image(input_image)
            output_image = seq.augment_image(output_image)

        #Random Simplex Noise Blobs
        if R[4] < 0.3:
            seq = iaa.SimplexNoiseAlpha( first=iaa.Multiply(mul = (0.6,1.4),per_channel=True),per_channel=True)
            input_image = seq.augment_image(input_image)
            output_image = seq.augment_image(output_image)
        
        #Now we have to work in [0,1] instead of [0,255]
        input_image = input_image/255.0
        output_image = output_image/255.0

        # Gaussian Blur
        if R[5] < 0.3:
            input_image = cv.GaussianBlur(input_image,(5,5),0)

        #Gaussian Noise
        if R[6] < 0:#0.2:
            mean = 0
            var = 0.001
            sigma = var**0.5
            gauss = np.random.normal(mean,sigma,(IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS))
            gauss = gauss.reshape(IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS)
            input_image = input_image + gauss

        if DEBUG:
            debug_img = output_image.copy()
            cv.circle(debug_img,(int(cornerTL_x),int(cornerTL_y)),10,(0,0,0),-1)
            cv.circle(debug_img,(int(cornerTL_x),int(cornerTL_y)),6,(255,255,255),-1)

            cv.circle(debug_img,(int(cornerTR_x),int(cornerTR_y)),10,(0,0,0),-1)
            cv.circle(debug_img,(int(cornerTR_x),int(cornerTR_y)),6,(255,255,255),-1)

            cv.circle(debug_img,(int(cornerBL_x),int(cornerBL_y)),10,(0,0,0),-1)
            cv.circle(debug_img,(int(cornerBL_x),int(cornerBL_y)),6,(255,255,255),-1)

            cv.circle(debug_img,(int(cornerBR_x),int(cornerBR_y)),10,(0,0,0),-1)
            cv.circle(debug_img,(int(cornerBR_x),int(cornerBR_y)),6,(255,255,255),-1)

            cv.imshow('DEBUG OUTPT',  cv.resize(debug_img,(960,720)))
            cv.waitKey()

        input_images.append(input_image)
        output_images.append(output_image)
        batch_corners.append((cornerTL_x,cornerTL_y,cornerTR_x,cornerTR_y,cornerBL_x,cornerBL_y,cornerBR_x,cornerBR_y))
        
    input_images = torch.Tensor(np.array(input_images)).permute(0,3,1,2)
    output_images = torch.Tensor(np.array(output_images)).permute(0,3,1,2)

    

    return input_images.cuda(), output_images.cuda(), batch_corners

def yield_training_batch_black_BG(img_file_list, corners_list):
        
    batch_size = len(img_file_list)

    input_images = []
    output_images = []
    batch_corners = []

    for f in range(0,batch_size):

        # Read image from list and convert to array
        input_image_path = SOURCE_FOLDER+'/Lens_Dataset/INPUTS/'+str(img_file_list[f])+'.png'
        output_image_path = SOURCE_FOLDER+'/Lens_Dataset/OUTPUTS/'+str(img_file_list[f])+'.png'
        
        input_image = cv.imread(input_image_path)
        output_image = cv.imread(output_image_path)

        cornerTL_x = corners_list[f][0]
        cornerTL_y = corners_list[f][1]
        cornerTR_x = corners_list[f][2]
        cornerTR_y = corners_list[f][3]
        cornerBL_x = corners_list[f][4]
        cornerBL_y = corners_list[f][5]
        cornerBR_x = corners_list[f][6]
        cornerBR_y = corners_list[f][7]


        if DEBUG:
            debug_img = output_image.copy()
            cv.circle(debug_img,(int(cornerTL_x),int(cornerTL_y)),10,(0,0,0),-1)
            cv.circle(debug_img,(int(cornerTL_x),int(cornerTL_y)),6,(255,255,255),-1)

            cv.circle(debug_img,(int(cornerTR_x),int(cornerTR_y)),10,(0,0,0),-1)
            cv.circle(debug_img,(int(cornerTR_x),int(cornerTR_y)),6,(255,255,255),-1)

            cv.circle(debug_img,(int(cornerBL_x),int(cornerBL_y)),10,(0,0,0),-1)
            cv.circle(debug_img,(int(cornerBL_x),int(cornerBL_y)),6,(255,255,255),-1)

            cv.circle(debug_img,(int(cornerBR_x),int(cornerBR_y)),10,(0,0,0),-1)
            cv.circle(debug_img,(int(cornerBR_x),int(cornerBR_y)),6,(255,255,255),-1)

            cv.imshow('PRE DEBUG OUTPT',  cv.resize(debug_img,(960,720)))


        #generate random vector to determine which augmentations to perform
        R = []
        for r in range(0,7):
            if DEBUG:
                R.append(0)
            else:
                rn = random.random()
                R.append(rn)

        #======ZOOM AND CROP OR NOT?======#

        if R[0] < 0.3:
            
            aspect_ratio = input_image.shape[1]/input_image.shape[0]
            target_ratio = IMG_WIDTH/IMG_HEIGHT
            height_center = int(input_image.shape[0]/2)
            width_center = int(input_image.shape[1]/2)

            if aspect_ratio > IMG_WIDTH/IMG_HEIGHT:
                target_width = int(input_image.shape[0]*target_ratio)
                input_image = input_image[0:input_image.shape[0], width_center-floor(target_width/2):width_center+floor(target_width/2)]
                output_image = output_image[0:input_image.shape[0], width_center-floor(target_width/2):width_center+floor(target_width/2)]
                
            elif aspect_ratio < IMG_WIDTH/IMG_HEIGHT:
                target_height = int(input_image.shape[1]*target_ratio)
                input_image = input_image[height_center-floor(target_height/2):height_center+floor(target_height/2), 0:input_image.shape[1]]
                output_image = output_image[height_center-floor(target_height/2):height_center+floor(target_height/2), 0:input_image.shape[1]]

            rn = random.uniform(1,1)    #how much to zoom in
            
            crop_w = floor(IMG_WIDTH*rn)    #how much to crop
            crop_h = floor(IMG_HEIGHT*rn)
            
            w_space = IMG_WIDTH - crop_w    #how much space is left
            h_space = IMG_HEIGHT - crop_h
            
            random_center_x = int(crop_w/2) + random.randint(0,w_space)     #center of the zoom/crop
            random_center_y = int(crop_h/2) + random.randint(0,h_space)
            
            #crops according to random parameters and resizes to desired input/output size
            input_image = input_image[random_center_y-int(crop_h/2):random_center_y+int(crop_h/2),random_center_x-int(crop_w/2):random_center_x+int(crop_w/2)]
            input_image = cv.resize(input_image, (IMG_WIDTH, IMG_HEIGHT), interpolation = cv.INTER_CUBIC)
            output_image = output_image[random_center_y-int(crop_h/2):random_center_y+int(crop_h/2),random_center_x-int(crop_w/2):random_center_x+int(crop_w/2)]
            output_image = cv.resize(output_image, (IMG_WIDTH, IMG_HEIGHT), interpolation = cv.INTER_NEAREST)


            # TODO: estas contas provavelmente estão mal, pelo que 'desabilitei o random crop e zoom'
            cornerTL_x = (cornerTL_x-(random_center_x - (IMG_WIDTH/2)))*rn
            cornerTL_y = (cornerTL_y-(random_center_y - (IMG_HEIGHT/2)))*rn
            cornerTR_x = (cornerTR_x-(random_center_x - (IMG_WIDTH/2)))*rn
            cornerTR_y = (cornerTR_y-(random_center_y - (IMG_HEIGHT/2)))*rn
            cornerBL_x = (cornerBL_x-(random_center_x - (IMG_WIDTH/2)))*rn
            cornerBL_y = (cornerBL_y-(random_center_y - (IMG_HEIGHT/2)))*rn
            cornerBR_x = (cornerBR_x-(random_center_x - (IMG_WIDTH/2)))*rn
            cornerBR_y = (cornerBR_y-(random_center_y - (IMG_HEIGHT/2)))*rn

        else:

            input_image = cv.resize(input_image, (IMG_WIDTH, IMG_HEIGHT), interpolation = cv.INTER_CUBIC)
            output_image = cv.resize(output_image, (IMG_WIDTH, IMG_HEIGHT), interpolation = cv.INTER_CUBIC)
            
            

        #======FLIP OR NOT?======#

        #Flip Horizontaly
        if R[1] < 0.5:
            input_image = cv.flip(input_image, 1)
            output_image = cv.flip(output_image, 1)
            cornerTL_x = (IMG_WIDTH) - cornerTL_x
            cornerTR_x = (IMG_WIDTH) - cornerTR_x
            cornerBL_x = (IMG_WIDTH) - cornerBL_x
            cornerBR_x = (IMG_WIDTH) - cornerBR_x
        
        #Flip Verticaly
        if R[2] < 0.5:
            input_image = cv.flip(input_image, 0)
            output_image = cv.flip(output_image, 0)
            cornerTL_y = (IMG_HEIGHT) - cornerTL_y
            cornerTR_y = (IMG_HEIGHT) - cornerTR_y
            cornerBL_y = (IMG_HEIGHT) - cornerBL_y
            cornerBR_y = (IMG_HEIGHT) - cornerBR_y
        
        #Color Shift
        if R[3] < 0.3:
            seq = iaa.Sequential([iaa.MultiplyHueAndSaturation((0.8, 1.2), per_channel=True)])
            input_image = seq.augment_image(input_image)
            output_image = seq.augment_image(output_image)

        #Random Simplex Noise Blobs
        if R[4] < 0.3:
            seq = iaa.SimplexNoiseAlpha( first=iaa.Multiply(mul = (0.6,1.4),per_channel=True),per_channel=True)
            input_image = seq.augment_image(input_image)
            output_image = seq.augment_image(output_image)
        
        #Now we have to work in [0,1] instead of [0,255]
        input_image = input_image/255.0
        output_image = output_image/255.0

        # Gaussian Blur
        if R[5] < 0.3:
            input_image = cv.GaussianBlur(input_image,(5,5),0)

        #Gaussian Noise
        if R[6] < 0:#0.2:
            mean = 0
            var = 0.001
            sigma = var**0.5
            gauss = np.random.normal(mean,sigma,(IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS))
            gauss = gauss.reshape(IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS)
            input_image = input_image + gauss

        sorted_x = np.sort((cornerTL_x,cornerBL_x,cornerTR_x,cornerBR_x))
        sorted_y = np.sort((cornerTL_y,cornerBL_y,cornerTR_y,cornerBR_y))

        # find the second largest/smallest of the x and y values, and give some manouvering margin to ensure the are no edges
        min_x = int(sorted_x[1])  + 10
        max_x = int(sorted_x[-2]) - 10
        min_y = int(sorted_y[1])  + 10
        max_y = int(sorted_y[-2]) - 10

        #min_x = int(min(cornerTL_x,cornerBL_x,cornerTR_x,cornerBR_x))
        #max_x = int(max(cornerTL_x,cornerBL_x,cornerTR_x,cornerBR_x))
        #min_y = int(min(cornerTL_y,cornerBL_y,cornerTR_y,cornerBR_y))
        #max_y = int(max(cornerTL_y,cornerBL_y,cornerTR_y,cornerBR_y))
    
        cv.rectangle(output_image,(0,0),(IMG_WIDTH,min_y),(0,0,0),-1)
        cv.rectangle(output_image,(0,max_y),(IMG_WIDTH,IMG_HEIGHT),(0,0,0),-1)
        cv.rectangle(output_image,(0,0),(min_x,IMG_HEIGHT),(0,0,0),-1)
        cv.rectangle(output_image,(max_x,0),(IMG_WIDTH,IMG_HEIGHT),(0,0,0),-1)


        if DEBUG:
            debug_img = output_image.copy()
            cv.circle(debug_img,(int(cornerTL_x),int(cornerTL_y)),10,(0,0,0),-1)
            cv.circle(debug_img,(int(cornerTL_x),int(cornerTL_y)),6,(255,255,255),-1)

            cv.circle(debug_img,(int(cornerTR_x),int(cornerTR_y)),10,(0,0,0),-1)
            cv.circle(debug_img,(int(cornerTR_x),int(cornerTR_y)),6,(255,255,255),-1)

            cv.circle(debug_img,(int(cornerBL_x),int(cornerBL_y)),10,(0,0,0),-1)
            cv.circle(debug_img,(int(cornerBL_x),int(cornerBL_y)),6,(255,255,255),-1)

            cv.circle(debug_img,(int(cornerBR_x),int(cornerBR_y)),10,(0,0,0),-1)
            cv.circle(debug_img,(int(cornerBR_x),int(cornerBR_y)),6,(255,255,255),-1)

            cv.imshow('DEBUG OUTPT',  cv.resize(debug_img,(960,720)))
            cv.waitKey()

        input_images.append(input_image)
        output_images.append(output_image)
        batch_corners.append((cornerTL_x,cornerTL_y,cornerTR_x,cornerTR_y,cornerBL_x,cornerBL_y,cornerBR_x,cornerBR_y))
        
    input_images = torch.Tensor(np.array(input_images)).permute(0,3,1,2)
    output_images = torch.Tensor(np.array(output_images)).permute(0,3,1,2)

    

    return input_images.cuda(), output_images.cuda(), batch_corners

def yield_training_batch_binary(img_file_list, corners_list):
        
    batch_size = len(img_file_list)

    input_images = []
    output_images = []
    batch_corners = []

    for f in range(0,batch_size):

        # Read image from list and convert to array
        input_image_path = SOURCE_FOLDER+'/Lens_Dataset/INPUTS/'+str(img_file_list[f])+'.png'
        output_image_path = SOURCE_FOLDER+'/Lens_Dataset/OUTPUTS/'+str(img_file_list[f])+'.png'
        
        input_image = cv.imread(input_image_path)
        output_image = cv.imread(output_image_path)

        cornerTL_x = corners_list[f][0]
        cornerTL_y = corners_list[f][1]
        cornerTR_x = corners_list[f][2]
        cornerTR_y = corners_list[f][3]
        cornerBL_x = corners_list[f][4]
        cornerBL_y = corners_list[f][5]
        cornerBR_x = corners_list[f][6]
        cornerBR_y = corners_list[f][7]


        if DEBUG:
            debug_img = output_image.copy()
            cv.circle(debug_img,(int(cornerTL_x),int(cornerTL_y)),10,(0,0,0),-1)
            cv.circle(debug_img,(int(cornerTL_x),int(cornerTL_y)),6,(255,255,255),-1)

            cv.circle(debug_img,(int(cornerTR_x),int(cornerTR_y)),10,(0,0,0),-1)
            cv.circle(debug_img,(int(cornerTR_x),int(cornerTR_y)),6,(255,255,255),-1)

            cv.circle(debug_img,(int(cornerBL_x),int(cornerBL_y)),10,(0,0,0),-1)
            cv.circle(debug_img,(int(cornerBL_x),int(cornerBL_y)),6,(255,255,255),-1)

            cv.circle(debug_img,(int(cornerBR_x),int(cornerBR_y)),10,(0,0,0),-1)
            cv.circle(debug_img,(int(cornerBR_x),int(cornerBR_y)),6,(255,255,255),-1)

            cv.imshow('PRE DEBUG OUTPT',  cv.resize(debug_img,(960,720)))


        #generate random vector to determine which augmentations to perform
        R = []
        for r in range(0,7):
            if DEBUG:
                R.append(0)
            else:
                rn = random.random()
                R.append(rn)

        #======ZOOM AND CROP OR NOT?======#

        if R[0] < 0.3:
            
            aspect_ratio = input_image.shape[1]/input_image.shape[0]
            target_ratio = IMG_WIDTH/IMG_HEIGHT
            height_center = int(input_image.shape[0]/2)
            width_center = int(input_image.shape[1]/2)

            if aspect_ratio > IMG_WIDTH/IMG_HEIGHT:
                target_width = int(input_image.shape[0]*target_ratio)
                input_image = input_image[0:input_image.shape[0], width_center-floor(target_width/2):width_center+floor(target_width/2)]
                output_image = output_image[0:input_image.shape[0], width_center-floor(target_width/2):width_center+floor(target_width/2)]
                
            elif aspect_ratio < IMG_WIDTH/IMG_HEIGHT:
                target_height = int(input_image.shape[1]*target_ratio)
                input_image = input_image[height_center-floor(target_height/2):height_center+floor(target_height/2), 0:input_image.shape[1]]
                output_image = output_image[height_center-floor(target_height/2):height_center+floor(target_height/2), 0:input_image.shape[1]]

            rn = random.uniform(1,1)    #how much to zoom in
            
            crop_w = floor(IMG_WIDTH*rn)    #how much to crop
            crop_h = floor(IMG_HEIGHT*rn)
            
            w_space = IMG_WIDTH - crop_w    #how much space is left
            h_space = IMG_HEIGHT - crop_h
            
            random_center_x = int(crop_w/2) + random.randint(0,w_space)     #center of the zoom/crop
            random_center_y = int(crop_h/2) + random.randint(0,h_space)
            
            #crops according to random parameters and resizes to desired input/output size
            input_image = input_image[random_center_y-int(crop_h/2):random_center_y+int(crop_h/2),random_center_x-int(crop_w/2):random_center_x+int(crop_w/2)]
            input_image = cv.resize(input_image, (IMG_WIDTH, IMG_HEIGHT), interpolation = cv.INTER_CUBIC)
            output_image = output_image[random_center_y-int(crop_h/2):random_center_y+int(crop_h/2),random_center_x-int(crop_w/2):random_center_x+int(crop_w/2)]
            output_image = cv.resize(output_image, (IMG_WIDTH, IMG_HEIGHT), interpolation = cv.INTER_NEAREST)


            # TODO: estas contas provavelmente estão mal, pelo que 'desabilitei o random crop e zoom'
            cornerTL_x = (cornerTL_x-(random_center_x - (IMG_WIDTH/2)))*rn
            cornerTL_y = (cornerTL_y-(random_center_y - (IMG_HEIGHT/2)))*rn
            cornerTR_x = (cornerTR_x-(random_center_x - (IMG_WIDTH/2)))*rn
            cornerTR_y = (cornerTR_y-(random_center_y - (IMG_HEIGHT/2)))*rn
            cornerBL_x = (cornerBL_x-(random_center_x - (IMG_WIDTH/2)))*rn
            cornerBL_y = (cornerBL_y-(random_center_y - (IMG_HEIGHT/2)))*rn
            cornerBR_x = (cornerBR_x-(random_center_x - (IMG_WIDTH/2)))*rn
            cornerBR_y = (cornerBR_y-(random_center_y - (IMG_HEIGHT/2)))*rn

        else:

            input_image = cv.resize(input_image, (IMG_WIDTH, IMG_HEIGHT), interpolation = cv.INTER_CUBIC)
            output_image = cv.resize(output_image, (IMG_WIDTH, IMG_HEIGHT), interpolation = cv.INTER_CUBIC)
            
            

        #======FLIP OR NOT?======#

        #Flip Horizontaly
        if R[1] < 0.5:
            input_image = cv.flip(input_image, 1)
            output_image = cv.flip(output_image, 1)
            cornerTL_x = (IMG_WIDTH) - cornerTL_x
            cornerTR_x = (IMG_WIDTH) - cornerTR_x
            cornerBL_x = (IMG_WIDTH) - cornerBL_x
            cornerBR_x = (IMG_WIDTH) - cornerBR_x
        
        #Flip Verticaly
        if R[2] < 0.5:
            input_image = cv.flip(input_image, 0)
            output_image = cv.flip(output_image, 0)
            cornerTL_y = (IMG_HEIGHT) - cornerTL_y
            cornerTR_y = (IMG_HEIGHT) - cornerTR_y
            cornerBL_y = (IMG_HEIGHT) - cornerBL_y
            cornerBR_y = (IMG_HEIGHT) - cornerBR_y
        
        #Color Shift
        if R[3] < 0.3:
            seq = iaa.Sequential([iaa.MultiplyHueAndSaturation((0.8, 1.2), per_channel=True)])
            input_image = seq.augment_image(input_image)
            output_image = seq.augment_image(output_image)

        #Random Simplex Noise Blobs
        if R[4] < 0.3:
            seq = iaa.SimplexNoiseAlpha( first=iaa.Multiply(mul = (0.6,1.4),per_channel=True),per_channel=True)
            input_image = seq.augment_image(input_image)
            output_image = seq.augment_image(output_image)
        
        #Now we have to work in [0,1] instead of [0,255]
        input_image = input_image/255.0
        output_image = output_image/255.0

        # Gaussian Blur
        if R[5] < 0.3:
            input_image = cv.GaussianBlur(input_image,(5,5),0)

        #Gaussian Noise
        if R[6] < 0:#0.2:
            mean = 0
            var = 0.001
            sigma = var**0.5
            gauss = np.random.normal(mean,sigma,(IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS))
            gauss = gauss.reshape(IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS)
            input_image = input_image + gauss

        sorted_x = np.sort((cornerTL_x,cornerBL_x,cornerTR_x,cornerBR_x))
        sorted_y = np.sort((cornerTL_y,cornerBL_y,cornerTR_y,cornerBR_y))

        # find the second largest/smallest of the x and y values, and give some manouvering margin to ensure the are no edges
        min_x = int(sorted_x[1])  + 10
        max_x = int(sorted_x[-2]) - 10
        min_y = int(sorted_y[1])  + 10
        max_y = int(sorted_y[-2]) - 10

        #min_x = int(min(cornerTL_x,cornerBL_x,cornerTR_x,cornerBR_x))
        #max_x = int(max(cornerTL_x,cornerBL_x,cornerTR_x,cornerBR_x))
        #min_y = int(min(cornerTL_y,cornerBL_y,cornerTR_y,cornerBR_y))
        #max_y = int(max(cornerTL_y,cornerBL_y,cornerTR_y,cornerBR_y))
    
        cv.rectangle(output_image,(0,0),(IMG_WIDTH,IMG_HEIGHT),(255,255,255),-1)
        cv.rectangle(output_image,(0,0),(IMG_WIDTH,min_y),(0,0,0),-1)
        cv.rectangle(output_image,(0,max_y),(IMG_WIDTH,IMG_HEIGHT),(0,0,0),-1)
        cv.rectangle(output_image,(0,0),(min_x,IMG_HEIGHT),(0,0,0),-1)
        cv.rectangle(output_image,(max_x,0),(IMG_WIDTH,IMG_HEIGHT),(0,0,0),-1)


        if DEBUG:
            debug_img = output_image.copy()
            cv.circle(debug_img,(int(cornerTL_x),int(cornerTL_y)),10,(0,0,0),-1)
            cv.circle(debug_img,(int(cornerTL_x),int(cornerTL_y)),6,(255,255,255),-1)

            cv.circle(debug_img,(int(cornerTR_x),int(cornerTR_y)),10,(0,0,0),-1)
            cv.circle(debug_img,(int(cornerTR_x),int(cornerTR_y)),6,(255,255,255),-1)

            cv.circle(debug_img,(int(cornerBL_x),int(cornerBL_y)),10,(0,0,0),-1)
            cv.circle(debug_img,(int(cornerBL_x),int(cornerBL_y)),6,(255,255,255),-1)

            cv.circle(debug_img,(int(cornerBR_x),int(cornerBR_y)),10,(0,0,0),-1)
            cv.circle(debug_img,(int(cornerBR_x),int(cornerBR_y)),6,(255,255,255),-1)

            cv.imshow('DEBUG OUTPT',  cv.resize(debug_img,(960,720)))
            cv.waitKey()

        output_image = cv.cvtColor(output_image.astype(np.uint8),cv.COLOR_BGR2GRAY)
        output_image = np.expand_dims(output_image,axis=2)

        input_images.append(input_image)
        output_images.append(output_image)
        batch_corners.append((cornerTL_x,cornerTL_y,cornerTR_x,cornerTR_y,cornerBL_x,cornerBL_y,cornerBR_x,cornerBR_y))
        
    input_images = torch.Tensor(np.array(input_images)).permute(0,3,1,2)
    output_images = torch.Tensor(np.array(output_images)).permute(0,3,1,2)

    

    return input_images.cuda(), output_images.cuda(), batch_corners

class EDSR(torch.nn.Module):
    def __init__(self, conv=common.default_conv):
        super(EDSR, self).__init__()

        rgb_range = 1.0
        n_resblocks = 5
        if DNN_TYPE==0:
            n_feats=3
            n_output_feats=3
            self.n_colors = 3
        else:
            n_feats=1
            n_output_feats=1
            self.n_colors = 3

        kernel_size = 3
        scale = 1
        act = torch.nn.ReLU(True)
        self.url = None
        self.sub_mean = common.MeanShift(rgb_range)
        self.add_mean = common.MeanShift(rgb_range, sign=1)
        
        self.res_scale = 1

        # define head module
        m_head = [conv(self.n_colors, n_feats, kernel_size)]

        # define body module
        m_body = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=self.res_scale
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            common.Upsampler(conv, scale, n_output_feats, act=False),
            conv(n_feats, self.n_colors, kernel_size)
        ]

        self.head = torch.nn.Sequential(*m_head)
        self.body = torch.nn.Sequential(*m_body)
        self.tail = torch.nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        #x = self.tail(res)
        x = self.add_mean(res)

        return x 

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, torch.nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

class EDSR_binary(torch.nn.Module):
    def __init__(self, conv=common.default_conv):
        super(EDSR_binary, self).__init__()

        rgb_range = 1.0
        n_resblocks = 5
        if DNN_TYPE==0:
            n_feats=3
            n_output_feats=3
            self.n_colors = 3
        else:
            n_feats=1
            n_output_feats=1
            self.n_colors = 3

        kernel_size = 3
        scale = 1
        act = torch.nn.ReLU(True)
        self.url = None
        self.sub_mean = common.MeanShift(rgb_range)
        self.add_mean = common.MeanShift(rgb_range, sign=1)
        self.collapse = torch.nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1)
        
        self.res_scale = 1

        # define head module
        m_head = [conv(self.n_colors, n_feats, kernel_size)]

        # define body module
        m_body = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=self.res_scale
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            common.Upsampler(conv, scale, n_output_feats, act=False),
            conv(n_feats, self.n_colors, kernel_size)
        ]

        self.head = torch.nn.Sequential(*m_head)
        self.body = torch.nn.Sequential(*m_body)
        self.tail = torch.nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        #x = self.tail(res)
        x = self.add_mean(res)
        x=self.collapse(x)

        return x 

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, torch.nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

FILTER_SIZE = 128
PADDING = 1

class DoubleConv(torch.nn.Module):
    """(Conv => BN => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),

            torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(torch.nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=[4,8, 16, 32]):
        super(UNet, self).__init__()

        # Downsampling path
        self.downs = torch.nn.ModuleList()
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Upsampling path
        self.ups = torch.nn.ModuleList()
        self.upconvs = torch.nn.ModuleList()

        rev_features = features[::-1]
        for feature in rev_features:
            self.upconvs.append(torch.nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature * 2, feature))

        # Final output layer
        if DNN_TYPE==0:
            self.final_conv = torch.nn.Conv2d(features[0], 3, kernel_size=1)
        else:
            self.final_conv = torch.nn.Conv2d(features[0], 1, kernel_size=1)

        
    def forward(self, x):
        # Encoder / Down path
        x1 = self.downs[0](x)              # [B, 64, H, W]
        x2 = self.downs[1](self.pool(x1))  # [B, 128, H/2, W/2]
        x3 = self.downs[2](self.pool(x2))  # [B, 256, H/4, W/4]
        x4 = self.downs[3](self.pool(x3))  # [B, 512, H/8, W/8]

        # Bottleneck
        x5 = self.bottleneck(self.pool(x4))  # [B, 1024, H/16, W/16]

        # Decoder / Up path
        u4 = self.upconvs[0](x5)
        if u4.shape != x4.shape:
            u4 = torch.nn.functional.interpolate(u4, size=x4.shape[2:])
        u4 = self.ups[0](torch.cat([x4, u4], dim=1))

        u3 = self.upconvs[1](u4)
        if u3.shape != x3.shape:
            u3 = torch.nn.functional.interpolate(u3, size=x3.shape[2:])
        u3 = self.ups[1](torch.cat([x3, u3], dim=1))

        u2 = self.upconvs[2](u3)
        if u2.shape != x2.shape:
            u2 = torch.nn.functional.interpolate(u2, size=x2.shape[2:])
        u2 = self.ups[2](torch.cat([x2, u2], dim=1))

        u1 = self.upconvs[3](u2)
        if u1.shape != x1.shape:
            u1 = torch.nn.functional.interpolate(u1, size=x1.shape[2:])
        u1 = self.ups[3](torch.cat([x1, u1], dim=1))

        # Final output
        return self.final_conv(u1)


#======================================== LOSS FUNCTION ==============================================

if DNN_TYPE==0:
    if TRAIN_TYPE==3:
        model=EDSR_binary()
    else:
        model=EDSR()
else:
    model=UNet()


class MSE_Crop_Loss(torch.nn.Module):
    def __init__(self):
        super(MSE_Crop_Loss, self).__init__()
    def forward(self,pred, target, corners):
        
        crop_score = 0

        for b in range(len(corners)):
            corners_x = corners[b][0:8:2]
            corners_y = corners[b][1:8:2]
            min_x = min(corners_x)
            max_x = max(corners_x)
            min_y = min(corners_y)
            max_y = max(corners_y)

            if min_x<0:
                min_x=0
            if min_y<0:
                min_y=0
            if max_x<0:
                max_x=0
            if max_y<0:
                max_y=0
            if min_x>IMG_WIDTH:
                min_x=IMG_WIDTH
            if min_y>IMG_HEIGHT:
                min_y=IMG_HEIGHT
            if max_x>IMG_WIDTH:
                max_x=IMG_WIDTH
            if max_y>IMG_HEIGHT:
                max_y=IMG_HEIGHT

            cropped_tgt = torchvision.transforms.functional.crop(target[b],int(min_y),int(min_x),int(max_y-min_y), int(max_x-min_x))
            cropped_pred = torchvision.transforms.functional.crop(pred[b],int(min_y),int(min_x),int(max_y-min_y), int(max_x-min_x))
            #cropped_tgt = target[b][min_x:max_x,min_y:max_y]
            #cropped_pred = pred[b][min_x:max_x,min_y:max_y]
            crop_score += torch.nn.functional.mse_loss(cropped_pred,cropped_tgt)
            #crop_score = np.square(np.subtract(cropped_pred,cropped_tgt)).mean()
        
        score = torch.nn.functional.mse_loss(pred,target)

        crop_score = crop_score / len(corners)
        return 1-(0.9*crop_score + 0.1*score)

x = torch.randn(1,3,1920, 1080)
y = model(x)

aux = y[0,:,:,:].cpu().detach().numpy()*255
aux = aux.astype(np.uint8)
aux = np.transpose(aux, (1,2,0))

make_dot(y.mean(), params=dict(model.named_parameters())).render('Debulr_Model', format="png")

model_image = cv.imread('Debulr_Model.png')
cv.imwrite(SOURCE_FOLDER+'/Results/Deblur_Model.png',model_image)

model= torch.nn.DataParallel(model)
model.to('cuda')

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

if TRAIN_TYPE == 0:
    loss_function = MSE_Crop_Loss()
elif TRAIN_TYPE==3:
    loss_function = torch.nn.BCEWithLogitsLoss(reduction='mean')
else:
    loss_function = torch.nn.MSELoss()

# Create Log file
with open(training_history_filename, "w") as csv_file:
    progress_string = 'EPOCH,LOSS,TRAIN_SCORE,VAL_SCORE/n'
    csv_file.write(progress_string)


best_train_score = 0.0

if TRAIN:

    for epoch in range(EPOCHS):

        #torch.cuda.empty_cache()

        #reset running loss
        running_loss = 0.0
        running_train_score = 0.0
        running_val_score = 0.0

        #shuffle data list
        train_df = train_df.sample(frac=1)    #shuffle dataframe
        
        csv_row = train_df.iloc[:]
        train_img_list = list(csv_row['ID'])
        train_corners_list = list(csv_row[['cornerTL_x','cornerTL_y','cornerTR_x','cornerTR_y','cornerBL_x','cornerBL_y','cornerBR_x','cornerBR_y']].to_numpy())

        # train batches ---------------------------------------------
        #iterate through batches
        for batch in range(0,len(train_img_list),BATCH_SIZE):

            if (batch+BATCH_SIZE<=len(train_img_list)):
                batch_img_list = train_img_list[batch:batch+BATCH_SIZE]
                batch_corners_list = train_corners_list[batch:batch+BATCH_SIZE]
            else:
                if len(train_img_list)==1:
                    batch_img_list = [train_img_list[0]]
                    batch_corners_list = [train_corners_list[0]]
                else:
                    batch_img_list = train_img_list[batch:-1]
                    batch_corners_list = train_corners_list[batch:-1]

            if TRAIN_TYPE == 1:
                batch_inputs, batch_outputs, batch_corners = yield_training_batch_black_BG(batch_img_list, batch_corners_list)
            elif TRAIN_TYPE == 3:
                batch_inputs, batch_outputs, batch_corners = yield_training_batch_binary(batch_img_list, batch_corners_list)
            else:
                batch_inputs, batch_outputs, batch_corners = yield_training_batch(batch_img_list, batch_corners_list)

            #print(torch.cuda.memory_summary(device=None, abbreviated=False))

            #read each image and augment
            batch_preds = model(batch_inputs)

            #reset the gradients
            optimizer.zero_grad()

            #claculate loss for batch
            if TRAIN_TYPE == 0:
                loss = loss_function(batch_preds, batch_outputs.cuda(),batch_corners)
            else:
                loss = loss_function(batch_preds, batch_outputs.cuda())


            #calculate score for batch
            train_score = 1-loss
            
            #backward pass
            loss.backward()

            #step the optimizer ...?
            optimizer.step()

            running_loss += loss
            running_train_score += train_score




        # validation batches ---------------------------------------------
        #shuffle data list
        val_df = val_df.sample(frac=1)    #shuffle dataframe
        
        csv_row = val_df.iloc[:]
        val_img_list = list(csv_row['ID'])
        val_corners_list = list(csv_row[['cornerTL_x','cornerTL_y','cornerTR_x','cornerTR_y','cornerBL_x','cornerBL_y','cornerBR_x','cornerBR_y']].to_numpy())

        #iterate through batches
        for batch in range(0,len(val_img_list),BATCH_SIZE):
            if (batch+BATCH_SIZE<len(val_img_list)):
                batch_img_list = val_img_list[batch:batch+BATCH_SIZE]
                batch_corners_list = val_corners_list[batch:batch+BATCH_SIZE]
            else:
                if len(val_img_list)==1:
                    batch_img_list = [val_img_list[0]]
                    batch_corners_list = [val_corners_list[0]]
                else:
                    batch_img_list = val_img_list[batch:-1]
                    batch_corners_list = val_corners_list[batch:-1]

            if TRAIN_TYPE == 1:
                batch_inputs, batch_outputs, batch_corners = yield_training_batch_black_BG(batch_img_list, batch_corners_list)
            elif TRAIN_TYPE == 3:
                batch_inputs, batch_outputs, batch_corners = yield_training_batch_binary(batch_img_list, batch_corners_list)
            else:
                batch_inputs, batch_outputs, batch_corners = yield_training_batch(batch_img_list, batch_corners_list)

            #read each image and augment
            batch_preds = model(batch_inputs)

            #calculate score for batch
            if TRAIN_TYPE == 0:
                val_score = 1-loss_function(batch_preds, batch_outputs.cuda(), batch_corners)
            else:
                val_score = 1-loss_function(batch_preds, batch_outputs.cuda())
            
                    
            running_val_score += val_score#.item()

        # Writes progress to csv file
        with open(training_history_filename, "a") as csv_file:  # append mode
            progress_string = str(epoch)+','+str(running_loss.item())+','+str(running_train_score.item())+','+str(running_val_score.item())+'/n'
            csv_file.write(progress_string)

        print('=========================================================')
        print('Epoch '+ str(epoch))
        print('Loss ' + str(running_loss.item()))
        print('Train Score '+ str(running_train_score.item()))
        print('Validation Score '+ str(running_val_score.item()))
        print('=========================================================')

        
        if running_train_score.item()>best_train_score:
            model_name = 'Best_Deblurr_model'

            if DNN_TYPE==0:
                if TRAIN_TYPE==3:
                    model_to_save=EDSR_binary()
                else:
                    model_to_save=EDSR()
            else:
                model_to_save=UNet()

            model_to_save.load_state_dict(model.module.state_dict())
            model_to_save.to('cuda')
            model_scripted = torch.jit.script(model_to_save)
            model_scripted.save(SOURCE_FOLDER+'/Results/'+model_name+'.pt') 
            #torch.save(model.module.state_dict(),SOURCE_FOLDER+'/LineFeatures_Results/'+model_name+'.pt')
            best_train_score=running_train_score.item()

print('DONE')

if TEST:
    
    model = torch.load("C:/Users/guilherme.franco/Documents/GAF/Estagio_Rui_Brito/Results_Black_BG/Best_Deblurr_model.pt")

    test_dir = filedialog.askdirectory(title='Choose Testing Folder',initialdir=os.getcwd())

    filelist=os.listdir(test_dir)
    for fichier in filelist[:]: # filelist[:] makes a copy of filelist.
        if not(fichier.endswith(".png")):
            filelist.remove(fichier)

    while True:
        r = random.randrange(0,len(filelist))

        frame = cv.imread(test_dir+'/'+filelist[r])

        frame = cv.resize(frame, (1920, 1080))/255        #resize to your nn input dimensions

        frame_T = np.transpose(frame, (2,0,1))

        np_tensor = np.zeros((1,3,1080,1920),np.float32)
        np_tensor[:,0,:,:] = frame_T[0,:,:]
        np_tensor[:,1,:,:] = frame_T[1,:,:]
        np_tensor[:,2,:,:] = frame_T[2,:,:]

        input_tensor = torch.tensor(np_tensor)

        preds = model(input_tensor.cuda())

        aux = preds[0,:,:,:].cpu().detach().numpy()*255
        aux = aux.astype(np.uint8)
        output_image = np.transpose(aux, (1,2,0))        #Do whatever you want. Save it to a folder, disply on opencv, etc...

        cv.imshow('Prediction', cv.resize(output_image,(960,540)))
        cv.waitKey(0)