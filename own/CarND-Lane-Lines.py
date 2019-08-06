def draw_lines(img, lines, color=[255, 0, 0], thickness=12):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    #    for line in lines:
#        for x1,y1,x2,y2 in line:
#            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

    if lines is None:
        return
    if len(lines) == 0:
        return
    
    draw_left = True
    left_lines = []
    left_lines_x = []
    left_lines_y = []
    
    draw_right = True
    right_lines = []
    right_lines_x = []
    right_lines_y = []
    slopes = []
    new_lines = []
    
    #Finding the slope
    slope_threshold = 0.5
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        #Caluclateing the slope
        if x2 - x1 == 0:
            slope = 999.
        else:
            slope = (y2 - y1) / (x2 - x1)
            
        if abs(slope) > slope_threshold:
            slopes.append(slope)
            new_lines.append(line)
            
    lines = new_lines
    
    #Seperate lines into left and right lanes
    for i, line in enumerate(lines):
        x1, y1, x2, y2 = line[0]
        center = img.shape[1] / 2
        if slopes[i] < 0 and x1 < center and x2 < center:
            left_lines.append(line)
        else:
            right_lines.append(line)
            
    for line in right_lines:
        x1, y1, x2, y2 = line[0]
        right_lines_x.append(x1)
        right_lines_x.append(x2)
        right_lines_y.append(y1)
        right_lines_y.append(y2)
        
    for line in left_lines:
        x1, y1, x2, y2 = line[0]
        left_lines_x.append(x1)
        left_lines_x.append(x2)
        left_lines_y.append(y1)
        left_lines_y.append(y2)

    if len(left_lines_x) > 0:
        left_m, left_b = np.polyfit(left_lines_x, left_lines_y, 1)
    else:
        left_m, left_b = 1, 1
        draw_left = False
        
    if len(right_lines_x) > 0:
        right_m, right_b = np.polyfit(right_lines_x, right_lines_y, 1)
    else:
        right_m, right_b = 1, 1
        draw_right = False
        
    #Finding min and max points for left and right lines
    y1 = img.shape[0]
    y2 = img.shape[0] / 1.7
    
    left_x1 = (y1 - left_b) / left_m
    left_x2 = (y2 - left_b) / left_m
    
    right_x1 = (y1 - right_b) / right_m
    right_x2 = (y2 - right_b) / right_m
               
    y1 - int(y1)
    y2 = int(y2)
    left_x1 = int(left_x1)
    left_x2 = int(left_x2)
    right_x1 = int(right_x1)
    right_x2 = int(right_x2)
    
    if draw_left:
        cv2.line(img, (left_x1, y1), (left_x2, y2), color, thickness)
    
    if draw_right:
        cv2.line(img, (right_x1, y1), (right_x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)

##########################################################################################################################################

##########################################################################################################################################

#Read in and grayscale the image
imname = 'test_images/whiteCarLaneSwitch.jpg'
image = mpimg.imread(imname)

gray = grayscale(image)

#Define a kernel size and apply Gaussian smoothing
kernel_size = 3
blur_gray = gaussian_blur(gray, kernel_size)

#Get the x and y sizes of the image        
y_size = image.shape[0]
x_size = image.shape[1]

#Define parameters for Canny().
low_threshold = 80
high_threshold = 250
edges = canny(blur_gray, low_threshold, high_threshold)

#Create masked image to be used with cv2.fillPolly()
imshape = image.shape
vertices = np.array([[(0,imshape[0]),(x_size/2, y_size/1.7), (x_size/2, y_size/1.7), (imshape[1], imshape[0])]], dtype=np.int32)
masked_edges = region_of_interest(edges, vertices)

#Define the Hough transform paramaters
#Make a blank the same size as the image to draw on
rho = 1
theta = np.pi/180
threshold = 20
min_line_length = 5
max_line_gap = 5
line_image = np.copy(image)*0

#Run Hough on edge detected image
lines = hough_lines(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)

#Create a "color" binary image to combine with line image.
color_edges = np.dstack((edges, edges, edges))

#Draw the lines on the edge image.
combo = weighted_img(lines, image)
plt.imshow(combo)


##########################################################################################################################################

##########################################################################################################################################

def process_image(image_in):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)
    # -*- coding: utf-8 -*-

    #Read in and grayscale the image
    gray = grayscale(image_in)

    #Define a kernel size and apply Gaussian smoothing
    kernel_size = 3
    blur_gray = gaussian_blur(gray, kernel_size)

    #Get the x and y sizes of the image        
    y_size = image_in.shape[0]
    x_size = image_in.shape[1]

    #Define parameters for Canny().
    low_threshold = 80
    high_threshold = 250
    edges = canny(blur_gray, low_threshold, high_threshold)

    #Create masked image to be used with cv2.fillPolly()
    imshape = image.shape
    vertices = np.array([[(0,imshape[0]),(x_size/2, y_size/1.7), (x_size/2, y_size/1.7), (imshape[1], imshape[0])]], dtype=np.int32)
    masked_edges = region_of_interest(edges, vertices)

    #Define the Hough transform paramaters
    #Make a blank the same size as the image to draw on
    rho = 1
    theta = np.pi/180
    threshold = 20
    min_line_length = 5
    max_line_gap = 5
    line_image = np.copy(image_in)*0
    
    #Run Hough on edge detected image
    lines = hough_lines(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)
    
    #Create a "color" binary image to combine with line image.
    color_edges = np.dstack((edges, edges, edges))

    #Draw the lines on the edge image.
    combo = weighted_img(lines, image_in)
    plt.imshow(combo)
    return combo
