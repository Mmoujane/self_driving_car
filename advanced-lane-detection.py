import cv2
import numpy as np


def warp(image):
    w = image.shape[1]
    h = image.shape[0]

    src = np.float32([[w//3 ,h//2], [w//6 ,h], [3*(w//5) ,h//2], [5*(w//6) ,h]])
    dst = np.float32([[0, 0], [0, 2*h//3], [w//2, 0], [w//2, 2*h//3]])

    M = cv2.getPerspectiveTransform(src, dst)

    warped = cv2.warpPerspective(image, M, (w//2, 2*h//3), flags=cv2.INTER_LINEAR)

    return warped, M

def canny(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    kernel = 5
    blur = cv2.GaussianBlur(gray,(kernel, kernel),0)
    canny = cv2.Canny(gray, 50, 150)
    return canny

def threshold(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    ret, image = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
    if(ret == False):
        print('Error in thresholding')
    else:
        return image

def lane_pixels(threshold_image):
    h = threshold_image.shape[0]
    w = threshold_image.shape[1]
    histogram = np.sum(threshold_image[h//2:,:], axis=0)
    midpoint = int(histogram.shape[0]//2)
    left_lane_index = np.argmax(histogram[:midpoint])
    right_lane_index = np.argmax(histogram[midpoint:]) + midpoint
    window_number = 10
    window_width = w//5
    min_pixels = 40
    window_height = h//window_number
    current_lwindow = left_lane_index
    current_rwindow = right_lane_index
    nonzero = threshold_image.nonzero()
    nonzero_x = np.array(nonzero[1])
    nonzero_y = np.array(nonzero[0])
    lane_pixels_left = []
    lane_pixels_right = []
    for window in range(window_number):
        window_ylow = h - (window + 1)*window_height
        window_yhight = h - window*window_height
        window_xleft_low = current_lwindow - window_width//2
        window_xleft_hight = current_lwindow + window_height//2
        window_xright_low = current_rwindow - window_width//2
        window_xright_hight = current_rwindow + window_height//2
        cv2.rectangle(threshold_image, (window_xleft_low, window_ylow), (window_xleft_hight, window_yhight), (255, 0, 0), 4)
        cv2.rectangle(threshold_image, (window_xright_low, window_ylow), (window_xright_hight, window_yhight), (255, 0, 0), 4)
        selected_pixels_left = ((nonzero_y >= window_ylow) & (nonzero_y < window_yhight) & (nonzero_x >= window_xleft_low) & (nonzero_x < window_xleft_hight)).nonzero()[0]
        selected_pixels_right = ((nonzero_y >= window_ylow) & (nonzero_y < window_yhight) & (nonzero_x >= window_xright_low) & (nonzero_x < window_xright_hight)).nonzero()[0]
        lane_pixels_left.append(selected_pixels_left)
        lane_pixels_right.append(selected_pixels_right)
        if len(selected_pixels_right) >= min_pixels:
            current_rwindow = int(np.mean(nonzero_x[selected_pixels_right]))
        if len(selected_pixels_left) >= min_pixels:
            current_lwindow = int(np.mean(nonzero_x[selected_pixels_left]))
    
    try:
        lane_pixels_left = np.concatenate(lane_pixels_left)
        lane_pixels_right = np.concatenate(lane_pixels_right)

    except ValueError:
        pass

    left_x = nonzero_x[lane_pixels_left]
    left_y = nonzero_y[lane_pixels_left]
    right_x = nonzero_x[lane_pixels_right]
    right_y = nonzero_y[lane_pixels_right]

    return left_x, left_y, right_x, right_y

def poly_fit(left_x, left_y, right_x, right_y, img):
    left_fit = np.polyfit(left_y, left_x, 2)
    right_fit = np.polyfit(right_y, right_x, 2)
    y_values = np.linspace(0, img.shape[0], img.shape[0])
    left_fit_x = left_fit[0]*y_values**2 + left_fit[1]*y_values + left_fit[2]
    right_fit_x = right_fit[0]*y_values**2 + right_fit[1]*y_values + right_fit[2]
    return left_fit_x, right_fit_x, y_values

def find_curvature(left_fit_x, right_fit_x, y_values, image):
    ym_per_pix = 30 / 720  
    xm_per_pix = 3.7 / 650
    left_fit_cv = np.polyfit(y_values*ym_per_pix, left_fit_x*xm_per_pix, 2)
    right_fit_cv = np.polyfit(y_values*ym_per_pix, right_fit_x*xm_per_pix, 2)
    left_curvature = int(np.max(np.absolute(2*left_fit_cv[0])/(1+(2*ym_per_pix*left_fit_cv[0]*y_values + left_fit_cv[1])**2)**1.5))
    right_curvature = int(np.max(np.absolute(2*right_fit_cv[0])/(1+(2*ym_per_pix*right_fit_cv[0]*y_values + right_fit_cv[1])**2)**1.5))
    left_slope = 2*ym_per_pix*left_fit_cv[0]*y_values + left_fit_cv[1]
    right_slope = 2*ym_per_pix*right_fit_cv[0]*y_values + right_fit_cv[1]
    left_angle = np.degrees(np.arctan(left_slope))
    right_angle = np.degrees(np.arctan(right_slope))
    left_mean_angle = np.mean(left_angle)
    right_mean_angle = np.mean(right_angle)
    avr_angle = (left_mean_angle + right_mean_angle)/2
    out_img = np.dstack((image, image, image)) * 255
    left = np.array([np.transpose(np.vstack([left_fit_x, y_values]))])
    right = np.array([np.flipud(np.transpose(np.vstack([right_fit_x, y_values])))])
    points = np.hstack((left, right))
    cv2.fillPoly(out_img, np.int_(points), (0, 0, 255))
    return out_img, left_curvature, right_curvature, avr_angle

     



cap = cv2.VideoCapture(0)
while(cap.isOpened()):
    _, frame = cap.read()
    warped_image,M = warp(frame)
    threshold_image = threshold(warped_image)
    left_x, left_y, right_x, right_y = lane_pixels(threshold_image)
    left_fit_x, right_fit_x, y_values = poly_fit(left_x, left_y, right_x, right_y, threshold_image)
    out_img, left_curvature, right_curvature, avr_angle = find_curvature(left_fit_x, right_fit_x, y_values, threshold_image)
    invM = np.linalg.inv(M)
    inversed_out_img = cv2.warpPerspective(out_img, invM, (frame.shape[1], frame.shape[0]), flags=cv2.INTER_LINEAR)
    combo_image = cv2.addWeighted(frame, 0.8, inversed_out_img, 1, 1)
    curvature = 'curvature: ' + str((left_curvature+right_curvature)/2)
    angle = 'angle: ' + str(round(avr_angle)) + 'deg'
    cv2.putText(combo_image, curvature, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(combo_image, angle, (50, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    #cv2.imshow("result", threshold_image)
    cv2.imshow("result", threshold_image)
    #cv2.imshow("result", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

 