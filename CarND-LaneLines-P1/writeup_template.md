**Finding Lane Lines on the Road** 

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report

### Reflection

My pipeline consists of 8 steps:
	1) Get the image/video file.
	2) Convert the image file to grayscale using cv2's cvtColor() function.
	3) Apply Gaussian smoothing to the image using cv2's GaussianBlur() function.
	4) Apply a canny transform using cv2's Canny() function.
	5) Determine the region of interest on the image.
	6) Running cv2's HoughLinesP() function on the image.
	7) Drawing lines on the image.
	8) Combine the line drawn image and the original image and display the image.

Some detail on the above mentioned steps:
	Step 4: I chose to use a low threshold of 80 and a high threshold of 250 as this gave me the desired results.

	Step 5: I get the x and y sizes of the image and divide the x size by 2 and the y sizes by 1.7 in order to get the desired region of interest.

	Step 6: I chose to go with a rho of 2, a theta of pi/180, a threshold of 20, a minimum line length of 5 and a maximum line gap of 5 in order to detect the lines on the image.


In order to draw a single line on the left and right lanes, I modified the draw_lines() function by:

	1) Checking if the lines variable is empty.
	2) setting a slope threshold to 0.5, aswell as calculating the slope.
	3) Seperating the lines into left and right lines.
	4) Finding the best fit line for the left and right lines using the polyfit function.
	5) Drawing the left and right lines onto the image.



###2. Identify potential shortcomings with your current pipeline

One potential shortcoming would be that when the camera changes position the line detection would become slightly inaccurate. 

Another shortcoming could be obstacles in the region of interest.

One other shortcoming would be if there is no lines on the road for example: lines have not been painted yet

###3. Suggest possible improvements to your pipeline

A possible improvement would be to accommodate for camera movement/position


