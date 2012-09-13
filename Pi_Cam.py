import cv2.cv as cv
import time
import sys, decimal, math
import numpy as np, cv2

storage = cv.CreateMemStorage(0)
image_scale = 1.3 				# 1.38
harr_scale = 1.2
min_neighbors = 1
harr_flags = 0

	  
def detect_pi_logo(camImg):
	
		
	imgView = camImg
	template = cv.LoadImage('pi_logo.jpg', cv2.CV_LOAD_IMAGE_GRAYSCALE)
				
	#template = cv.LoadImageM('RobotSeagull-96x96.png', 
	#cv2.CV_LOAD_IMAGE_GRAYSCALE)
	
	#cv.SetImageROI(imgView, (100,100,50,50))
	#template = cv.CloneImage(logo2)
	
	cv.ShowImage('Template', template)
		
	W,H = cv.GetSize(imgView)
	w,h = cv.GetSize(template)
	
	width = W-w+1
	height = H-h+1
	
	result = cv.CreateImage((width,height),32,1) #cv.IPL_DEPTH_8U,1) 32
	
	cv.MatchTemplate(imgView, template, result, cv.CV_TM_SQDIFF)
	(min_val, max_val, min_loc, max_loc) = cv.MinMaxLoc(result)
	
	return  max_loc, min_val, w, h
	
	

#def detect_object(image):
#	size = cv.GetSize(image)
	#prepare memory
	#car = cv.CreateImage(size, 8, 1)
	#red = cv.CreateImage(size, 8, 1)
	#hsv = cv.CreateImage(size, 8, 3)
	#sat = cv.CreateImage(size, 8, 1)
	#split image into hsv, grab the sat
#	cv.CvtColor(image, hsv, cv.CV_BGR2HSV)
#	cv.Split(hsv, None, sat, None, None)
    #split image into rgb
#	cv.Split(image, None, None, red, None)
    #find the car by looking for red, high saturation
#	cv.Threshold(red, red, 128, 255, cv.CV_THRESH_BINARY)
#	cv.Threshold(sat, sat, 128, 255, cv.CV_THRESH_BINARY)
    # AND the two thresholds, finding the car
#	cv.Mul(red, sat, car)
    #remove noise, highlighting the car
#	cv.Erode(car, car, iterations = 5)
#	cv.Dilate(car, car, iterations = 5)
#	storage = cv.CreateMemStorage(0)
#	obj = cv.FindContours(car, storage, cv.CV_RETR_CCOMP, 
#	cv.CV_CHAIN_APPROX_SIMPLE)
	#cv.ShowImage("car", car)
	#cv.ShowImage("red", red)
	#cv.ShowImage("hsv", hsv)
	#cv.ShowImage("sat",sat)
	#cv.CvtColor(car, image, cv.CV_HSV2BGR)	
#	cv.ShowImage("Object Detection", car)
#	if not obj:
#		return(0,0,0,0)
#	else: 
#		return cv.BoundingRect(obj)
	#points = []
	#capture = cv.CaptureFromCam(0)
	#if not capture:
	#	print "Error opening capture device"
	#	sys.exit(1)
	#cv.SetCaptureProperty(capture, cv.CV_CAP_PROP_FRAME_WIDTH, 640)
	#cv.SetCaptureProperty(capture, cv.CV_CAP_PROP_FRAME_HEIGHT, 480)

def detect_face_draw(img):
	
	#size = cv.GetSize(img)
	#img = cv.CreateImage(size, 8, 1)
	
	grey = img
	small_img = cv.CreateImage((cv.Round(img.width/image_scale),
	cv.Round(img.height/image_scale)), 8, 1)
		
	# scale input image for faster processing
	cv.Resize(grey, small_img, cv.CV_INTER_NN)
	cv.EqualizeHist(small_img, small_img)
	# Start detection 
	if (cascade):
	  faces = cv.HaarDetectObjects(small_img, cascade, storage,
	  harr_scale, min_neighbors, harr_flags)
			
	if faces:    
		for (x,y,w,h), n in faces:		
	     # the input to cvHarrDetectObjects was resized, so scale the
	     # bounding box of each face and convert it to two CvPoints
	 	 pt1=(int(x*image_scale), int(y*image_scale))
		 pt2=(int((x+w)*image_scale), int((y+h)*image_scale))
		 	
		# Draw the rectangle on the image
		cv.Rectangle(img, pt1, pt2, cv.CV_RGB(0,255,0),3,8,0)
		cv.ShowImage('Face Detection - Harr Cascade', img) 
		

#cv.NamedWindow('Gaussian Blur to remove noise', 1)	

#cv.NamedWindow('Logo Found - Match Template', 1)
cv.NamedWindow('Greyscale', 1)
cv.NamedWindow('Edge Detection - Canny', 1)
cv.NamedWindow('Face Detection - Harr Cascade', 1)

# Load the Haar cascade
cascade_name="./haarcascade_frontalface_alt_tree.xml"
cascade = cv.Load(cascade_name)

cap = cv.LoadImage('/media/40BF-3C21/camview.jpg', cv.CV_LOAD_IMAGE_COLOR)

frames = 0
start_time = time.time()

while True:
      
  try:  
	cap = cv.LoadImage('/media/40BF-3C21/camview.jpg', cv.CV_LOAD_IMAGE_COLOR)
  except IOError:
	print 'In Loop - File busy'
  
  logocap =cv.CloneImage(cap)  
  procap = cv.CloneImage(cap)
  
  #cv.ShowImage('Logo Found - Match Template', logocap)
  cv.ShowImage('Processing Org', procap)
  
  yuv = cv.CreateImage(cv.GetSize(procap), 8, 3)
  grey = cv.CreateImage(cv.GetSize(procap), 8, 1)
  cv.CvtColor(procap, yuv, cv.CV_BGR2YCrCb)
  cv.Split(yuv, grey, None, None, None)
  
  cv.ShowImage('Greyscale', grey)
  
  # Gaussian blur to remove noise
  #blur = cv.CreateImage(cv.GetSize(grey), cv.IPL_DEPTH_8U, grey.channels)
  #cv.Smooth(grey, blur, cv.CV_GAUSSIAN, 5, 5)
  #cv.ShowImage('Gaussian Blur to remove noise', blur)
  
  # And do Canny edge detection
  canny = cv.CreateImage(cv.GetSize(grey), grey.depth, grey.channels)
  cv.Canny(grey, canny, 10, 100, 3)
  cv.ShowImage('Edge Detection - Canny', canny)    
    
  # Detect Faces
  detect_face_draw(grey)  
      
  # Detect Logo
  #location = detect_pi_logo(grey)
  
  #(#(x, y), v, w, h) =  location
  
  #if (v > 5.0):
	#cv.Rectangle(cap, (int(x), int(y)),(int (x)+50,int(y)+50),(255,255,255),1,0)
  
  print "Frame", frames
  frames += 1

  if frames % 10 == 0:
   currtime = time.time()
   numsecs = currtime - start_time
   
   fps = frames / numsecs
   a = round(fps,2)
   print "Average FPS:", a   
  
  c = cv.WaitKey(7) % 0x100
  
  if c == 27:
	  del(cap.cam)
	  cv.distroyAllWindows()
	  break     
    
   
