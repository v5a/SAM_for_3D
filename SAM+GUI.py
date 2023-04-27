#----------------------------------------------#
#导入所需的库
#----------------------------------------------#
import pygame
import math
from sys import exit
import cv2
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np
import torch
import matplotlib.pyplot as plt

#----------------------------------------------#
#用于展示的代码
#----------------------------------------------#
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    print("w")
    ax.imshow(mask_image)
    return mask_image

def to_show_mask(mask, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    print("w")
    # ax.imshow(mask_image)
    return mask_image   
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    
#----------------------------------------------#
#将图片格式转换为pygame的Surface格式
#----------------------------------------------#
def convert_opencv_img_to_pygame(opencv_image):

    # opencv_image = opencv_image[:,:,::-1]  # OpenCVはBGR、pygameはRGBなので変換してやる必要がある。
    shape = opencv_image.shape[1::-1]  # OpenCVは(高さ, 幅, 色数)、pygameは(幅, 高さ)なのでこれも変換。
    pygame_image = pygame.image.frombuffer(opencv_image.tostring(), shape, 'RGB')

    return pygame_image
#----------------------------------------------#
#读入SAM预训练模型
#使用GPU开始训练
#----------------------------------------------#
import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor

sam_checkpoint = "sam_vit_h_4b8939.pth"#预训练权重
model_type = "vit_h"#预训练权重的类型

device = "cuda"#用不用GPU

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)

img_src = 'images/dog.jpg'                      #自己的项目需要修改图片路径
#----------------------------------------------#
#利用opencv传入图片
#----------------------------------------------#
image = cv2.imread(img_src)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


predictor.set_image(image)#通过调用“SamPredictor.set_image”处理图像以生成图像嵌入。“SamPrejector”会记住此嵌入，并将其用于后续掩码预测。

#----------------------------------------------#
#设置屏幕大小
#----------------------------------------------#
screen = pygame.display.set_mode((image.shape[1],image.shape[0]))
background = pygame.image.load(img_src).convert()
while True:
    #----------------------------------------------#
    #读入图片并监听鼠标
    #----------------------------------------------#
    x,y = pygame.mouse.get_pos() 
    for event in pygame.event.get():
        #----------------------------------------------#
        #画正负prompt点
        #----------------------------------------------#
        if 'input_point' and 'input_label' in dir():
            n=0
            for point in input_point:
                # for x,y in point:
                if input_label[n]==1:
                    color_num = (0,255,0)
                if input_label[n]==0:
                    color_num = (255,0,0)
                n = n+1
                pygame.draw.circle(background,color_num,(point[0],point[1]),5)
        #监听键盘
        if event.type == pygame.KEYDOWN:
            #trans.write(chr(event.key).encode())
            if event.key == 13:
                print("ENTER")
            print(chr(event.key))
        #监听鼠标
        if event.type == pygame.MOUSEBUTTONDOWN:
            pressed_array = pygame.mouse.get_pressed()
            for index in range(len(pressed_array)):
                 if pressed_array[index]:
                     
                     #是否是鼠标左键
                     if index == 0:
                        print('Pressed LEFT Button!')
                        print(str(x)+' '+str(y))
                        
                        if 'input_point' and 'input_label' not in dir():       #查看变量有没有定义，没有就加一个定义
                            input_point = np.array([[x,y]])
                            input_label = np.array([1])
                            print(input_point)
                            print(input_label)
                        else:
                            input_point = np.concatenate((input_point,np.array([[x,y]])),axis=0)
                            input_label = np.concatenate((input_label,np.array([1])))
                            print(input_point)
                            print(input_point)
                        
                        masks, scores, logits = predictor.predict(
                            point_coords=input_point,
                            point_labels=input_label,
                            multimask_output=False,
                        )
                        # masks = ~masks   #反色
                        masks = np.uint8(masks*150)
                        c = np.concatenate([masks,masks,masks],axis=0)
                        c= c.transpose(1,2,0)
                        imgadd = cv2.add(image,c)
                        background = convert_opencv_img_to_pygame(imgadd)
                     #是否是鼠标中键
                     elif index == 1:
                         print('The mouse wheel Pressed!')
                         del input_point
                         del input_label
                         background = pygame.image.load(img_src).convert()
                     #是否是鼠标右键
                     elif index == 2:
                         print('Pressed RIGHT Button!')
                         print(str(x)+' '+str(y))
                         if 'input_point' and 'input_label' not in dir():       #查看变量有没有定义，没有就加一个定义
                            input_point = np.array([[x,y]])
                            input_label = np.array([0])
                            print(input_point)
                            print(input_label)
                         else:
                            input_point = np.concatenate((input_point,np.array([[x,y]])),axis=0)
                            input_label = np.concatenate((input_label,np.array([0])))
                            print(input_point)
                            print(input_label)
                        
                         masks, scores, logits = predictor.predict(
                            point_coords=input_point,
                            point_labels=input_label,
                            multimask_output=False,
                        )
                        #  masks = ~masks    #反色
                         masks = np.uint8(masks*150)
                         c = np.concatenate([masks,masks,masks],axis=0)
                         c= c.transpose(1,2,0)
                         imgadd = cv2.add(image,c)
                         background = convert_opencv_img_to_pygame(imgadd)
    
    #刷新屏幕
    screen.blit(background,(0,0))
    pygame.display.update()
