import tkinter
from tkinter import filedialog
from tkinter.filedialog import askdirectory
from tkinter import *

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

#----------------------------------------------#
#保存为json文件
#----------------------------------------------#
from PIL import Image
import json


#----------------------------------------------#
#DINO初始化
#----------------------------------------------#


from huggingface_hub import hf_hub_download

def load_model_hf(repo_id, filename, ckpt_config_filename, device='cpu'):
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)
    
    # cache_config_file = r'GroundingDINO_SwinB.cfg.py'
    print('cache_config_file',type(cache_config_file),cache_config_file)
    args = SLConfig.fromfile(cache_config_file) 
    print('args',type(args))
    model = build_model(args)
    args.device = device

    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    
    # cache_file = r'groundingdino_swinb_cogcoor.pth'
    checkpoint = torch.load(cache_file, map_location='cpu')
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print("Model loaded from {} \n => {}".format(cache_file, log))
    _ = model.eval()
    return model

ckpt_repo_id = "ShilongLiu/GroundingDINO"
ckpt_filenmae = "groundingdino_swinb_cogcoor.pth"
ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"

groundingdino_model = load_model_hf(ckpt_repo_id, ckpt_filenmae, ckpt_config_filename)
BOX_TRESHOLD = 0.3
TEXT_TRESHOLD = 0.6

import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor

sam_checkpoint = "sam_vit_h_4b8939.pth"#预训练权重
model_type = "vit_h"#预训练权重的类型

device = "cuda"#用不用GPU

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)
#-------------------------------------------------------------------------------------------------------------------------------------------------------#
def autolabeling(img_src,text,shape_type="polygon"):
    image = cv2.imread(img_src)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)
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
    # print("boxes",boxes)
    for box in boxes:
        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(box) * torch.Tensor([W, H, W, H])
    # boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes[0]) * torch.Tensor([W, H, W, H])
        print(boxes_xyxy,type(boxes_xyxy))
        if 'boxes_list' in dir():
            boxes_list.append(boxes_xyxy.tolist())
        else:
            boxes_list = [boxes_xyxy.tolist()]

        transformed_boxes = predictor.transform.apply_boxes_torch(boxes_xyxy, image_source.shape[:2]).to("cuda")
        masks, _, _ = predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes = transformed_boxes,
                multimask_output = False,
            )
        masks = np.uint8(masks[0].cpu()*150)
        c = np.concatenate([masks,masks,masks],axis=0)
        c= c.transpose(1,2,0)
        #------------------------------------------#
        imgray = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(imgray, 127, 255, 0)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # 这里取轮廓点数最多的（可能返回多个轮廓）
        contour = contours[0]
        for cont in contours:
            if len(cont) > len(contour):
                contour = cont
        print('contour',type(contour),contour.shape)
        if 'contour_list' in dir():
            contour_list.append(contour)
        else:
            contour_list = [contour]

    # print(boxes_list,type(boxes_list))
#UnboundLocalError: local variable 'contour_list' referenced before assignment
    __version__="5.2.0.post4"
    flags={}
    null=None
    img = Image.open(img_src)
    imgSize = img.size  #大小/尺寸
    imageHeight = img.height  
    imageWidth = img.width
    filepath, tempfilename = os.path.split(img_src)
    filename, extension = os.path.splitext(tempfilename)
    imagePath_ = tempfilename
    save_path = filepath+'\\'+filename+'.json'
    if 'boxes_list' not in dir():
        return
    if shape_type=='rectangle':
        for boxes in boxes_list:
            #-----------------shapes----------------#
            label=text#------#
            # points=contours_
            group_id = None
            description=""
            shape_type="rectangle"
            flags={}
            #-----------------shapes----------------#
            shape = {
                "label": label,
                "points":[[boxes[0],boxes[1]],[boxes[2],boxes[3]]],
                "group_id": group_id,
                "description": description,
                "shape_type": shape_type,
                "flags": flags
                }
            if 'shapes' in dir():
                shapes.append(shape)
            else:
                shapes = [shape]
    #-------------------------------------------------------------------------------------------#
    if 'contour_list' not in dir():
        return
    if shape_type=='polygon':
        for contour_ in contour_list:
            contours_ = contour_.reshape(-1,2).tolist()
            #-----------------shapes----------------#
            label=text#------#
            points=contours_
            group_id = None
            description=""
            shape_type="polygon"
            flags={}
            #-----------------shapes----------------#
            shape = {
                "label": label,
                "points":points,
                "group_id": group_id,
                "description": description,
                "shape_type": shape_type,
                "flags": flags
                }
            
            if 'shapes' in dir():
                shapes.append(shape)
            else:
                shapes = [shape]
    #-------------------------------------------------------------------------------------------#
    imagePath=imagePath_
    imageData = None
    imageHeight = imageHeight
    imageWidth = imageWidth

    data = dict(
        version=__version__,
        flags=flags,
        shapes=shapes,
        imagePath=imagePath,
        imageData=imageData,
        imageHeight=imageHeight, 
        imageWidth=imageWidth,
    )

    with open(save_path, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        print("save")
#-------------------------------------------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------------------------------------------#
def select_folder(self):
    text1.insert(tkinter.INSERT, '')   
    files = filedialog.askdirectory()
    text1.insert(tkinter.INSERT, files)     
    # files = filedialog.askopenfilenames(filetypes=(('image files', ['*.jpg','*.png','*.bmp','*.jpeg']),('All files', '*.*')),title='Select Input File')
    # fileList=list(files) 
    # 读取文件夹中的所有文件
    imgs = os.listdir(files)
    print(imgs)
    # 图片名列表
    names = []

    # 过滤：只保留png结尾的图片
    for img in imgs:
        if img.endswith(".png") or img.endswith(".jpg") or img.endswith(".jpeg") or img.endswith(".bmp"):
            names.append(files+'/'+img)
    global global_files
    global_files=files
    global global_names
    global_names=names
    print(names)
    return
#-------------------------------------------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------------------------------------------#
def to_label(self):
    hobby_polygons = hobby1.get()
    hobby_rectangle = hobby2.get()
    if hobby_polygons==True and hobby_rectangle==False:
        shape_type = 'polygon'
    else:
        shape_type = 'rectangle'
    text = text_for_DINO.get()
    for src in global_names:
        autolabeling(src,text,shape_type)
    return
#-------------------------------------------------------------------------------------------------------------------------------------------------------#
def open_labelme(self):
    cmd = "labelme"+" "+global_files
    os.system(cmd)
#创建操作窗口
root = tkinter.Tk()
root.title("自动标注")
root.minsize(500,250)   #窗口默认打开时的大小

#创建检测按钮
button1 = tkinter.Button(root, text="选择文件夹")
button1.place(relx = 0.65, rely = 0.15, width = 80)
 
#创建命令执行按钮
button2 = tkinter.Button(root, text="生成标注")
button2.place(relx = 0.65, rely = 0.40, width = 80)

#创建命令执行按钮
button3 = tkinter.Button(root, text="预览")
button3.place(relx = 0.65, rely = 0.65, width = 80)
#显示文本路径
text1 = tkinter.Text(root)
text1.place(relx = 0.1, rely = 0.15, relwidth=0.5, height = 30)#relheight=0.5, relwidth=0.8)
#输入框的位置大小

text_for_DINO=StringVar()
entry1 = tkinter.Entry(root,textvariable=text_for_DINO)#textvariable=w,w=StringVar(), w.get(),w.set('7')
entry1.place(relx = 0.1, rely = 0.40, relwidth=0.5, height = 30)
# text_for_DINO.set("I love Python!")

hobby1 = tkinter.BooleanVar()
hobby1.set(True)
check1 = tkinter.Checkbutton(root, text='polygons', variable=hobby1)#command=updata
#check1.pack(anchor='w')
check1.place(relx = 0.1, rely = 0.65)
hobby2 = tkinter.BooleanVar()
hobby2.set(False)
check2 = tkinter.Checkbutton(root, text='rectangle', variable=hobby2)#command=updata
#check2.pack()
check2.place(relx = 0.4, rely = 0.65)





button1.bind("<ButtonRelease-1>", select_folder )#点击按钮时触发的行为，这里调用了poc函数。即点击按钮时只会调用这个函数，函数外的语句不会执行
button2.bind("<ButtonRelease-1>", to_label ) #点击按钮时触发行为
button3.bind("<ButtonRelease-1>", open_labelme )
root.mainloop()