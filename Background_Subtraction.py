import numpy as np
import cv2

#------------- Calculations of GMM on each frame of the video -----------------#

def pixel_calculations(frame, sig, pi, mean, back, fore, gaussians, alpha, ro):
    # get the dimensions of the frame
    rows, cols, channels = frame.shape
    rat = [0 for i in range(gaussians)]

    # For each pixel in the frame compute new mean and variance
    for z in range(0, channels):
        for k in range(0, rows):
            for i in range(0, cols):
                flag = 0
                temp = 0
                for j in range(0, gaussians):
                    if abs(frame[k][i][z] - mean[i + cols * k][j][z]) < (2.5 * (sig[i + cols * k][j]) ** (1 / 2.0)):
                        mean[i + cols * k][j][z] = (1 - ro) * mean[i + cols * k][j][z] + ro * frame[k][i][z]
                        sig[i + cols * k][j] = (1 - ro) * sig[i + cols * k][j] + ro * (frame[k][i][z] - mean[i + cols * k][j][z]) ** 2
                        pi[i + cols * k][j][z] = (1 - alpha) * pi[i + cols * k][j][z] + alpha
                        flag = 1
                    else:
                        pi[i + cols * k][j][z] = (1 - alpha) * pi[i + cols * k][j][z]
                    temp = temp + pi[i + cols * k][j][z]

                # Normalize the coputed weights and find the corresponding pi/sig values
                for j in range(0, gaussians):
                    pi[i + cols * k][j][z] = pi[i + cols * k][j][z] / temp
                    rat[j] = pi[i + cols * k][j][z] / sig[i + cols * k][j]

                # Arrange the mean, variance and weights in decreasing order as per the ratio pi/sig
                for j in range(0, gaussians):
                    swapped = False
                    for x in range(0, gaussians - j - 1):
                        if rat[x] < rat[x + 1]:
                            rat[x], rat[x + 1] = rat[x + 1], rat[x]
                            pi[i + cols * k][x][z], pi[i + cols * k][x + 1][z]= pi[i + cols * k][x + 1][z], pi[i + cols * k][x][z]
                            mean[i + cols * k][x][z], mean[i + cols * k][x + 1][z] = mean[i + cols * k][x + 1][z], mean[i + cols * k][x][z]
                            sig[i + cols * k][x], sig[i + cols * k][x + 1]= sig[i + cols * k][x + 1], sig[i + cols * k][x]
                            swapped = True
                    if swapped == False:
                        break

                # If the current pixel does not belong to any gaussian, update the one with least weightage
                if flag == 0:
                    mean[i + cols * k][gaussians - 1][z] = frame[k][i][z]
                    sig[i + cols * k][gaussians - 1] = 10000

                # Check if the current pixel belongs to background or foreground
                b = 0
                B = 0
                for j in range(0, gaussians):
                    b = b + pi[i + cols * k][j][z]
                    if b > 0.9:
                        B = j
                        break

                # Update the value of foreground and background pixel
                for j in range(0, B + 1):
                    if flag == 0 or abs(frame[k][i][z] - mean[i + cols * k][j][z]) > (2.5 * (sig[i + cols * k][j]) ** (1 / 2.0)):
                        fore[k][i][z] = frame[k][i][z]
                        back[k][i][z] = mean[i + cols * k][j][z]
                        break
                    else:
                        fore[k][i][z]= 255
                        back[k][i][z] = frame[k][i][z]
    # make the background pixels of the foreground frame white
    for z in range(0, channels):
        for k in range(0, rows):
            for i in range(0, cols):
                if fore[k][i][z] == 255:
                    fore[k][i][0] = fore[k][i][1] = fore[k][i][2] = 255
    fore = cv2.medianBlur(fore, 3)
    return back, fore

#------- Perform K means on the initial frame of the video for initialization of the gaussians ------#

def K_means(frame, gaussians):

    rows, cols, channels = frame.shape
    points = rows * cols
    r = np.zeros((points, gaussians, channels))

    # Considering the initial vaules of means
    u = [[50, 50, 50], [130, 130, 130], [210, 210, 210]]
    b = [[50, 50, 50], [130, 130, 130], [210, 210, 210]]
    itr = 0

    # compute till the convergence of means
    while (1):
        # clustering each pixel of the image to nearest mean
        for z in range(0, channels):
            for k in range(0, rows):
                for i in range(0, cols):
                    a = frame[k][i][z]
                    min = (a - u[0][z]) ** 2
                    r[i + cols * k][0][z] = 0
                    id = 0
                    for j in range(1, gaussians):
                        c = (a - u[j][z]) ** 2
                        if c < min:
                            min = c
                            id = j

                        r[i + cols * k][j][z] = 0

                    r[i + cols * k][id][z] = 1

        p = np.zeros((gaussians, channels))

        # Calculating the mean of the new clusters
        for z in range(0, channels):
            for j in range(0, gaussians):
                p[j][z] = 1
                for k in range(0, rows):
                    for i in range(0, cols):
                        u[j][z] = u[j][z] + frame[k][i][z] * r[i + cols * k][j][z]
                        p[j][z] = p[j][z] + r[i + cols * k][j][z]
                u[j][z] = u[j][z] / p[j][z]

        # Check if the new cluster mean is converged below a threshold
        sum = 0
        for z in range(0, channels):
            for j in range(0, gaussians):
                sum = sum + (b[j][z] - u[j][z]) ** 2
                b[j][z] = u[j][z]
        if sum < 100:
            break
        itr += 1

    # Calculate the Variances of the new clusters around the means
    si = np.zeros((gaussians))
    for j in range(0, gaussians):
        p[j][0] = 0
        for k in range(0, rows):
            for i in range(0, cols):
                si[j] = si[j] + (frame[k][i][0] - u[j][0]) ** 2 * r[i + cols * k][j][0]
                p[j][0] = p[j][0] + r[i + cols * k][j][0]

        si[j] = si[j] / p[j][0]

    return u, r, si

#----------- Make the Foreground and Background videos from the generated frames -----------------#

def Make_video(frame,total):

    # Get the dimensions of the frame
    rows, cols, channels = frame.shape

    # Initialize variable to write the video at 30FPS
    out = cv2.VideoWriter(r"D:\python\assignment1\video5\vbackground.avi", -1, 30.0, (cols, rows))
    count = 0

    # Write the frames into the video
    while (count < total):
        # Read the background frame
        frame = cv2.imread(r"D:\python\assignment1\video5\back%d.jpg" % count, 1)

        # Write the frame in to the video
        out.write(frame)

        # Display the frames
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        count += 1
    out.release()

    # Initialize variable to write the video at 30FPS
    out2 = cv2.VideoWriter(r"D:\python\assignment1\video5\vforeground.avi", -1, 30.0, (cols, rows))
    count = 0

    # Write the frames into the video
    while (count < total):

        # Read the foreground frame
        frame = cv2.imread(r"D:\python\assignment1\video5\fore%d.jpg" % count, 1)

        # Write the frame in to the video
        out2.write(frame)

        # Display the frames
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        count += 1
    out2.release()



#####---------------- Main Program ----------------------------------######

if __name__ == "__main__":
	# Read the input video
	cap = cv2.VideoCapture('video.mpg')

	# taking 1st frame of the video
	ret, frame = cap.read()
	rows, cols, channels = frame.shape
	points = rows * cols

	# Setting the parameters
	gaussians = 3
	alpha = 0.025
	ro = 0.11

	# kmeans on first frame
	u, r, si = K_means(frame, gaussians)


	# calculate the parameters for gaussians

	sig = np.zeros((points, gaussians))
	pi = np.zeros((points, gaussians, channels))
	mean = np.zeros((points, gaussians, channels))
	back = np.zeros((rows, cols, channels), dtype = np.uint8)
	fore = np.zeros((rows, cols, channels), dtype = np.uint8)


	for z in range(0, channels):
	    for j in range(0, gaussians):
		for k in range(0, rows):
		    for i in range(0, cols):
		        mean[i + cols * k][j][z] = u[j][z]
		        sig[i + cols * k][j] = si[j]
		        pi[i + cols * k][j][z] = (1 / gaussians) * (1 - alpha) + alpha * r[i + cols * k][j][z]

	count = 0


	# Updation of gaussian mixture model for each pixel in the new frame

	while (ret):

	    # take each frame from the video
	    ret, frame = cap.read()

	    # perform background and foreground detection on the frame using GMM and save it in back and fore
	    back, fore = pixel_calculations(frame, sig, pi, mean, back, fore, gaussians, alpha, ro)

	    # save the foreground and background frames
	    cv2.imwrite(r"D:\python\assignment1\video5\fore%d.jpg" % count, fore)
	    cv2.imwrite(r"D:\python\assignment1\video5\back%d.jpg" % count, back)

	    count += 1

	# Make the video from generated frames
	Make_video(frame,count)

	# Release the input video variable
	cap.release()
	cv2.destroyAllWindows()
