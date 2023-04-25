import pygame
import math
from sys import exit
import cv2
# image = cv2.imread("masks2.jpg")
screen = pygame.display.set_mode((1280,720))
while True:
    background = pygame.image.load('masks2.jpg').convert()
    x,y = pygame.mouse.get_pos()
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            #trans.write(chr(event.key).encode())
            print(chr(event.key))
        if event.type == pygame.MOUSEBUTTONDOWN:
            pressed_array = pygame.mouse.get_pressed()
            for index in range(len(pressed_array)):
                 if pressed_array[index]:
                     if index == 0:
                        print('Pressed LEFT Button!')
                        print(str(x)+' '+str(y))
                     elif index == 1:
                         print('The mouse wheel Pressed!')
                     elif index == 2:
                         print('Pressed RIGHT Button!')
                         print(str(x)+' '+str(y))
    screen.blit(background,(0,0))
    pygame.display.update()
