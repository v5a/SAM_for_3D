#----------------------------------------------#
#导入所需的库
#----------------------------------------------#
import pygame
import math
from sys import exit
import cv2
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import sys
sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))
# If you have multiple GPUs, you can set the GPU to use here.
# The default is to use the first GPU, which is usually GPU 0.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"



# Grounding DINO
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from GroundingDINO.groundingdino.util.inference import annotate, load_image, predict

import numpy as np
import torch
import matplotlib.pyplot as plt

#pygame初始化
pygame.init()

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
#DINO初始化
#----------------------------------------------#


from huggingface_hub import hf_hub_download

def load_model_hf(repo_id, filename, ckpt_config_filename, device='cpu'):
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)

    args = SLConfig.fromfile(cache_config_file) 
    model = build_model(args)
    args.device = device

    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location='cpu')
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print("Model loaded from {} \n => {}".format(cache_file, log))
    _ = model.eval()
    return model

# Use this command for evaluate the Grounding DINO model
# Or you can download the model by yourself
ckpt_repo_id = "ShilongLiu/GroundingDINO"
ckpt_filenmae = "groundingdino_swinb_cogcoor.pth"
ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"

groundingdino_model = load_model_hf(ckpt_repo_id, ckpt_filenmae, ckpt_config_filename)



# local_image_path = 'assets/inpaint_demo.jpg'
# TEXT_PROMPT = "dog"
BOX_TRESHOLD = 0.3
TEXT_TRESHOLD = 0.25



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
# img_src = 'assets/inpaint_demo.jpg'
# local_image_path = 'assets/inpaint_demo.jpg'
#----------------------------------------------#
#利用opencv传入图片
#----------------------------------------------#
image = cv2.imread(img_src)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


predictor.set_image(image)#通过调用“SamPredictor.set_image”处理图像以生成图像嵌入。“SamPrejector”会记住此嵌入，并将其用于后续掩码预测。

#----------------------------------------------#
#图片reshape到固定大小显示。
#----------------------------------------------#
# 输入你想要resize的图像高。
size = 640
# 获取原始图像宽高。
height, width = image.shape[0], image.shape[1]
# 等比例缩放尺度。
scale = height/size
# 获得相应等比例的图像宽度。
width_size = int(width/scale)
# resize
image_resize = cv2.resize(image, (width_size, size))
#----------------------------------------------#
#设置屏幕大小
#----------------------------------------------#
screen = pygame.display.set_mode((image_resize.shape[1]+200,image_resize.shape[0]))
# background = pygame.image.load(convert_opencv_img_to_pygame(image_resize)).convert()
background = convert_opencv_img_to_pygame(image_resize)
white = [255, 255, 255]
red = [255, 0, 0]
screen.fill(white)#加一个白底
screen.blit(background,(0,0))
#----------------------------------------------#
#加一个文本框
#----------------------------------------------#

font = pygame.font.Font(None, 32)
input_box = pygame.Rect(image_resize.shape[1]+10, 150, 140, 32)
color_inactive = pygame.Color('lightskyblue3')
color_active = pygame.Color('dodgerblue2')
color = color_inactive
active = False
text = ''
done = False

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
                pygame.draw.circle(background,color_num,(int(point[0]/scale),int(point[1]/scale)),5)
        if 'boxes_xyxy' in dir():
            pygame.draw.rect(background, (0,0,0), [int(boxes_xyxy[0]/scale),int(boxes_xyxy[1]/scale),int((boxes_xyxy[2]-boxes_xyxy[0])/scale),int((boxes_xyxy[3]-boxes_xyxy[1])/scale)], 2)
        #监听键盘
        if event.type == pygame.KEYDOWN:
            screen.fill((255, 255, 255))
            if active:
                    if event.key == pygame.K_RETURN:
                        print(text)
                        image_source, image_ = load_image(img_src)

                        boxes, logits, phrases = predict(
                            model=groundingdino_model, 
                            image=image_, 
                            caption=text, 
                            box_threshold=BOX_TRESHOLD, 
                            text_threshold=TEXT_TRESHOLD
                        )

                        annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
                        annotated_frame = annotated_frame[...,::-1] # BGR to RGB

                        # box: normalized box xywh -> unnormalized xyxy
                        H, W, _ = image_source.shape
                        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes[0]) * torch.Tensor([W, H, W, H])

                        transformed_boxes = predictor.transform.apply_boxes_torch(boxes_xyxy, image_source.shape[:2]).to("cuda")
                        # if 'input_point' and 'input_label' not in dir():       #查看变量有没有定义，没有就加一个定义
                        #     input_point = None
                        #     input_label = None
                        masks, _, _ = predictor.predict_torch(
                                    point_coords=None,
                                    point_labels=None,
                                    boxes = transformed_boxes,
                                    multimask_output = False,
                                )
                        # print(masks)
                        # masks_1 = np.uint8(masks[0].cpu()*150)
                        # c = np.concatenate([masks_1,masks_1,masks_1],axis=0)
                        # c= c.transpose(1,2,0)
                        # imgadd = cv2.add(image,c)

                        # masks_2 = np.uint8(masks[1].cpu()*150)
                        # c = np.concatenate([masks_2,masks_2,masks_2],axis=0)
                        # c= c.transpose(1,2,0)
                        # imgadd = cv2.add(imgadd,c)
                        
                        # print(masks.shape)
                        masks = np.uint8(masks[0].cpu()*150)
                        c = np.concatenate([masks,masks,masks],axis=0)
                        c= c.transpose(1,2,0)
                        imgadd = cv2.add(image,c)

                        imgadd = cv2.resize(imgadd, (width_size, size))
                        background = convert_opencv_img_to_pygame(imgadd)
                        text = ''
                    elif event.key == pygame.K_BACKSPACE:
                        text = text[:-1]
                    else:
                        text += event.unicode
            #trans.write(chr(event.key).encode())
            # if event.key == 13:
            #     print("ENTER")
            # print(chr(event.key))
        #监听鼠标
        if event.type == pygame.MOUSEBUTTONDOWN:
            #----------------------------------------------#
            #激活文本框
            #----------------------------------------------#
            # If the user clicked on the input_box rect.
            if input_box.collidepoint(event.pos):
                # Toggle the active variable.
                active = not active
            else:
                active = False
            # Change the current color of the input box.
            color = color_active if active else color_inactive
            #----------------------------------------------#
            #超过图片区域不激活prompt点
            #----------------------------------------------#
            if x>image_resize.shape[1] or y>image_resize.shape[0]:
                continue

            pressed_array = pygame.mouse.get_pressed()#获取案件情况
            for index in range(len(pressed_array)):
                 if pressed_array[index]:
                     
                     #是否是鼠标左键
                     if index == 0:
                        print('Pressed LEFT Button!')
                        print(str(x)+' '+str(y))
                        x_original = int(x*scale)
                        y_original = int(y*scale)
                        if 'input_point' and 'input_label' not in dir():       #查看变量有没有定义，没有就加一个定义
                            input_point = np.array([[x_original,y_original]])
                            input_label = np.array([1])
                            print(input_point)
                            print(input_label)
                        else:
                            input_point = np.concatenate((input_point,np.array([[x_original,y_original]])),axis=0)
                            input_label = np.concatenate((input_label,np.array([1])))
                            print(input_point)
                            print(input_label)
                        # else:
                        #     input_point = np.array([[x_original,y_original]])
                        #     input_label = np.array([1])
                        #     print(input_point)
                        #     print(input_label)
                            
                        if 'boxes_xyxy' not in dir():
                            boxes = None
                        else:
                             boxes = boxes_xyxy.numpy().round()
                            
                            #  boxes = np.array([351 ,233 ,624, 931])
                            #  boxes = np.array([233,351,931, 624])
                             boxes = boxes[None, :]
                             print("boxes:"+str(boxes))
                        masks, scores, logits = predictor.predict(
                            point_coords=input_point,
                            point_labels=input_label,
                            box = boxes,
                            multimask_output=False,
                        )
                        # masks = ~masks   #反色
                        # print(type(masks))
                        # print(masks.shape)
                        masks = np.uint8(masks*150)
                        c = np.concatenate([masks,masks,masks],axis=0)
                        print(type(c))
                        print(c.shape)
                        c= c.transpose(1,2,0)
                        imgadd = cv2.add(image,c)
                        imgadd = cv2.resize(imgadd, (width_size, size))
                        background = convert_opencv_img_to_pygame(imgadd)
                     #是否是鼠标中键
                     elif index == 1:
                         print('The mouse wheel Pressed!')
                         if 'input_point' and 'input_label' in dir(): 
                            del input_point
                            del input_label
                         if 'boxes_xyxy' in dir():
                            del boxes_xyxy
                         background = convert_opencv_img_to_pygame(image_resize)
                     #是否是鼠标右键
                     elif index == 2:
                         print('Pressed RIGHT Button!')
                         print(str(x)+' '+str(y))
                         x_original = int(x*scale)
                         y_original = int(y*scale)
                         if 'input_point' and 'input_label' not in dir():       #查看变量有没有定义，没有就加一个定义
                            input_point = np.array([[x_original,y_original]])
                            input_label = np.array([0])
                            transformed_boxes = None
                            print(input_point)
                            print(input_label)
                         else:
                            input_point = np.concatenate((input_point,np.array([[x_original,y_original]])),axis=0)
                            input_label = np.concatenate((input_label,np.array([0])))
                            print(input_point)
                            print(input_label)
                         if 'boxes_xyxy' not in dir():
                            boxes = None
                         else:
                             boxes = boxes_xyxy.numpy().round()
                            
                            #  boxes = np.array([351 ,233 ,624, 931])
                            #  boxes = np.array([233,351,931, 624])
                             boxes = boxes[None, :]
                             print("boxes:"+str(boxes))
                         masks, scores, logits = predictor.predict(
                            point_coords=input_point,
                            point_labels=input_label,
                            box = boxes,
                            multimask_output=False,
                        )
                        #  masks = ~masks    #反色
                         masks = np.uint8(masks*150)
                         c = np.concatenate([masks,masks,masks],axis=0)
                         c= c.transpose(1,2,0)
                         imgadd = cv2.add(image,c)
                         imgadd = cv2.resize(imgadd, (width_size, size))
                         background = convert_opencv_img_to_pygame(imgadd)

    # Render the current text.
    txt_surface = font.render(text, True, color)
    # Resize the box if the text is too long.
    width = max(150, txt_surface.get_width()+10)
    input_box.w = width
    # Blit the text.
    screen.blit(txt_surface, (input_box.x+5, input_box.y+5))
    # Blit the input_box rect.
    pygame.draw.rect(screen, color, input_box, 2)
    pygame.display.flip()


    #刷新屏幕
    screen.blit(background,(0,0))
    pygame.display.update()
