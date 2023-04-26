#----------------------------------------------#
#导入所需的库
#----------------------------------------------#
import pygame
import math
from sys import exit
import cv2
#----------------------------------------------#
#设置屏幕大小
#----------------------------------------------#
screen = pygame.display.set_mode((1280,720))
while True:
    #----------------------------------------------#
    #读入图片并监听鼠标
    #----------------------------------------------#
    background = pygame.image.load('masks2.jpg').convert()
    x,y = pygame.mouse.get_pos() 
    for event in pygame.event.get():
        #监听键盘
        if event.type == pygame.KEYDOWN:
            #trans.write(chr(event.key).encode())
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
                     
                     #是否是鼠标中键
                     elif index == 1:
                         print('The mouse wheel Pressed!')
                    
                     #是否是鼠标右键
                     elif index == 2:
                         print('Pressed RIGHT Button!')
                         print(str(x)+' '+str(y))
    
    #刷新屏幕
    screen.blit(background,(0,0))
    pygame.display.update()
