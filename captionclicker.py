import cv2
import numpy as np
from matplotlib import pyplot as plt
import pyautogui
from pynput.mouse import Button, Controller
import time
import pyperclip as pc
from ai import *

def main():
	image = takeScreenshot()

	#identify all instances of edit --> how many pictures there are
	edit = cv2.imread('edit.jpg')
	edit = cv2.cvtColor(np.array(edit), cv2.COLOR_BGR2GRAY) #<-- no need to convert?

	#other necessary images for template matching
	rightArr = cv2.imread('right_arrow_2.jpg')
	rightArr = cv2.cvtColor(np.array(rightArr), cv2.COLOR_BGR2GRAY)

	addDesc = cv2.imread('add_desc.jpg')
	addDesc = cv2.cvtColor(np.array(addDesc), cv2.COLOR_BGR2GRAY)

	matches = findImg(image, edit)

	clickCoords = getClick(matches)
	print(clickCoords)

	click(clickCoords)

	alt = cv2.imread('alt.jpg')
	alt = cv2.cvtColor(np.array(alt), cv2.COLOR_BGR2GRAY)

	matches =[]

	while(len(matches) == 0):
		image = takeScreenshot()
		matches = findImg(image, alt)
		
	coords = getClick(matches)
	click(coords)

	i1 = 'example.jpeg'
	i2 = 'example2.jpg'
	tp = 'tokenizer.pkl'
	m = 'model_18.h5'

	#temporarily hardcoded, but is able to collect number of pics dynamically based on template matches
	pics = 2
	picCount = 0

	de = cv2.imread('description.jpg')
	de = cv2.cvtColor(np.array(de), cv2.COLOR_BGR2GRAY)

	save = cv2.imread('save.jpg')
	save = cv2.cvtColor(np.array(save), cv2.COLOR_BGR2GRAY)
	
	#loop through each picture and insert caption using ai
	while(picCount < pics):
		image = takeScreenshot()
		match = findImg(image, de)
		# click(desc_coords)
		if(picCount == 0):
			caption1 = ai_run(i1, tp, m)
			pyautogui.typewrite(caption1)
		elif(picCount == 1):
			caption2 = ai_run(i2, tp, m)
			pyautogui.typewrite(caption2)
		arrMatch = findImg(image, rightArr)
		arrowCoords = getClick(arrMatch)
		click(arrowCoords)
		picCount += 1

	#save captions to tweet
	image = takeScreenshot()
	submitMatch = findImg(image, save)
	saveCoords = getClick(submitMatch)
	click(saveCoords)

#perform a click on given coordinates
def click(coords):
	mouse = Controller()
	mouse.position = coords
	mouse.click(Button.left, 1)

#take a screenshot of the current screen for reference
def takeScreenshot():
	image = pyautogui.screenshot()
	image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY) #cv2.COLOR_RGB2BGR for color if ever needed
	return image

#display an image (for testing purposes)
def displayImage(image):
	cv2.imshow("Image", image)
	cv2.waitKey(0)

#draw a circle on an image (for testing purposes)
def drawCircle(image, midpoint):
	cv2.circle(image, (midpoint[0], midpoint[1]), 10, (0,0,255), -1)
	displayImage(image)

#get the coordinates needed for click
def getClick(matches):
	smallestX = matches[0][0]
	y = matches[0][1]

	#click the first photo if multiple, otherwise "first" option (leftmost/highest on the screen)
	for match in matches:
		if(match[0] < smallestX):
			smallestX = match[0]
			y = match[1]
		elif(match[0] == smallestX):
			if(match[1] < y):
				y = match[1]

	return (smallestX, y)

#find a specific icon using template matching and return the coordinates of every match
def findImg(i, t):
	if(isinstance(i, str)):
		image = cv2.imread(i) #read in image
	else:
		image = i
	
	#reminder that image i is grayscale
	gray = image
	template = t

	#find all matches
	result= cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)

	#accuracy threshold
	threshold = .70

	print(1 - result)

	w, h = template.shape[::-1]

	#find every instance of a template match
	loc = np.where(result >= threshold)
	matches = []
	
	#keep track of all matches
	for pt in zip(*loc[::-1]):
		matches.append(pt)

	print('matches:')
	print(matches)
	print('--------')

	#draw rectangles around matches - for testing purposes
	for pt in matches:
		cv2.rectangle(image, pt, (pt[0] + w, pt[1] + h), (255,0,0), 1)

	#cv2.imshow("finalimage", image)
	cv2.waitKey(0)

	return matches

if __name__ == "__main__":
    main()

