#importing all the necessary libraries
import cv2
import numpy as np
import math
#importing the required videos and image
cap = cv2.VideoCapture('multipleTags.mp4')
image = cv2.imread('testudo.png')
#converting the image into gray scale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
videoWriter = cv2.VideoWriter('filter3.avi', fourcc, 30.0, (1000,600))
#videoWriter1 = cv2.VideoWriter('result31.avi', fourcc, 30.0, (1000,600))
#videoWriter2 = cv2.VideoWriter('result32.avi', fourcc, 30.0, (1000,600))
class Movingaveragefilter:
    def __init__(self,window,amount):
        self.window = window
        self.average =0
        self.amount =amount
        self.shapes = []
    def includeshapes(self,pts):
        if len(self.shapes) < self.window:
            self.shapes.append(pts)
        else:
            self.shapes.pop(0)
            self.shapes.append(pts)
    def avg(self):
        shape = np.array(self.shapes)
        am= np.ones(1, self.window)
        am[0,self.window-1] = self.amount
        s = 0
        for i in range(self.window):
            s =s+am[0,i]*amount[i]
        self.average = s / np.sum(am)
        return self.average
    def length(self):
        ll = len(self.shapes)
        return ll
#computing the homography using world and camera co-ordinates
def homographyMatrix(alignment, x_coordinate, y_coordinate):
    #computing the camera co-ordinate values
    xc1=0
    yc1=0
    xc2=x_coordinate
    yc2 = 0
    xc3=x_coordinate
    yc3 =y_coordinate
    xc4=0
    yc4=y_coordinate

#computing the world co-ordinate values
    if alignment == 'bottom_right':
        x1=aprx[0][0][0]
        y1=aprx[0][0][1]
        x2=aprx[1][0][0]
        y2=aprx[1][0][1]
        x3=aprx[2][0][0]
        y3=aprx[2][0][1]
        x4=aprx[3][0][0]
        y4=aprx[3][0][1]

    elif alignment == 'bottom_left':

        x1=aprx[1][0][0]
        y1=aprx[1][0][1]
        x2=aprx[2][0][0]
        y2=aprx[2][0][1]
        x3=aprx[3][0][0]
        y3=aprx[3][0][1]
        x4=aprx[0][0][0]
        y4=aprx[0][0][1]

    elif alignment == 'top_left':

        x1=aprx[2][0][0]
        y1=aprx[2][0][1]
        x2=aprx[3][0][0]
        y2=aprx[3][0][1]
        x3=aprx[0][0][0]
        y3=aprx[0][0][1]
        x4=aprx[1][0][0]
        y4=aprx[1][0][1]

    elif alignment == 'top_right':
        x1=aprx[3][0][0]
        y1=aprx[3][0][1]
        x2=aprx[0][0][0]
        y2=aprx[0][0][1]
        x3=aprx[1][0][0]
        y3=aprx[1][0][1]
        x4=aprx[2][0][0]
        y4=aprx[2][0][1]
    #computing the A matrix from AH=0
    A = [[x1, y1, 1, 0, 0, 0, -xc1*x1, -xc1*y1, -xc1],
         [0, 0, 0, x1, y1, 1, -yc1*x1, -yc1*y1, -yc1],
         [x2, y2, 1, 0, 0, 0, -xc2*x2, -xc2*y2, -xc2],
         [0, 0, 0, x2, y2, 1, -yc2*x2, -yc2*y2, -yc2],
         [x3, y3, 1, 0, 0, 0, -xc3*x3, -xc3*y3, -xc3],
         [0, 0, 0, x3, y3, 1, -yc3*x3, -yc3*y3, -yc3],
         [x4, y4, 1, 0, 0, 0, -xc4*x4, -xc4*y4, -xc4],
         [0, 0, 0, x4, y4, 1, -yc4*x4, -yc4*y4, -yc4]]
    #since, A matrix is not a square matrix,finding the eigen vectors using singular value decomposition and forming the homography matrix
    h, e, vt = np.linalg.svd(A)
    #extracting the last row as the eigen values
    HH = np.array(vt[8, :]/vt[8, 8])
    H=HH.reshape((-1, 3))
    #taking the inverse of it
    inverse_H = np.linalg.inv(H)
    return (H, inverse_H)


#getting orientation wrt the videos

def get_i_j(img):
    a = {}
    s = 0
    list = []
#TOP LEFT
    for i in range(100, 151):
        for j in range(100, 151):
            s += img[i, j]
    top_left = s/2500
    a[top_left] = 'top_left'
    list.append(top_left)
#BOTTOM_RIGHT
    for i in range(250, 301):
        for j in range(250, 301):
            s += img[i, j]
    bottom_right = s/2500
    a[bottom_right] = 'bottom_right'
    list.append(bottom_right)
#BOTTOM_LEFT
    for i in range(250, 301):
        for j in range(100, 151):
            s += img[i, j]
    bottom_left = s/2500
    a[bottom_left] = 'bottom_left'
    list.append(bottom_left)
#TOP RIGHT
    for i in range(100, 151):
        for j in range(250, 301):
            s += img[i, j]
    top_right = s/2500
    a[top_right] = 'top_right'
    list.append(top_right)

    return a[max(list)]

#identifying the tag ID
def tagID(img, OID):
    s = 0
    ID = ''
    #when the white block is in top left corner
    if OID == 'top_left':
        order = ['bottom_left', 'bottom_right', 'top_right', 'top_left']
    #when the white block is in top right corner
    elif OID == 'top_right':
        order = ['bottom_right', 'top_right', 'top_left', 'bottom_left']
    #when the white corner is in bottom left corner
    elif OID == 'bottom_left':
        order = ['top_left', 'bottom_left', 'bottom_right', 'top_right']
    #when the white corner is in bottom right
    elif OID == 'bottom_right':
        order = ['top_right', 'top_left', 'bottom_left', 'bottom_right']

    alignment = {'bottom_left': [150, 200, 200, 250],
                'bottom_right': [200, 250, 200, 250],
                 'top_right': [200, 250, 150, 200],
                 'top_left': [150, 200, 150, 200]}
    #looping through each alignmentto detect the tag id
    for n in range(0, 4):
        for i in range(alignment[order[n]][0], alignment[order[n]][1]):
            for j in range(alignment[order[n]][2], alignment[order[n]][3]):
                s += img[i, j]
        #finding the highest average pixel values.
        bottom_left = s/2500
        if bottom_left > 180:
            ID = ID + '1'
        else:
            ID = ID+ '0'
    return ID

#getting KRT matrix wrt to the videos
#reference with the pdf provided.
def KRT_Mat(H, inverse_H):
    #given intrinsic camera parameters
    kk = np.array([[1406.08415449821, 0, 0], [2.20679787308599, 1417.99930662800, 0], [1014.13643417416, 566.347754321696, 1]])
    #transposing  it and taking the inverse of it
    K=kk.T
    #taking the inverse of K
    I_K = np.linalg.inv(K)
    #computing the next values of the variable H
    h=I_K.dot(inverse_H)
    #first column extraction
    h11 = h[:, 0]
    h1=h11.reshape(3, 1)
    #second column extraction
    h22 = h[:, 1]
    h2=h22.reshape(3, 1)
    #third column extraction
    h33 = h[:, 2]
    h3=h33.reshape(3, 1)
    #finding the lambda(scalar) value by normalising it using numpy linear algorithm function
    lmda=(np.linalg.norm(np.dot(I_K,h1)))+np.linalg.norm(np.dot(I_K,h2)) / 2
    lmda= 1/lmda
    #finding the R values individually and stacking it together
    R1 = lmda*h1
    R2 = lmda*h2
    R3 = np.cross(h11, h22)
    R33=(R3 * lmda * lmda)
    R3 = R33.reshape(3, 1)
    R = np.concatenate((R1, R2, R3), axis=1)

    T = lmda*h3
    return R, T, K

#getting the 3-D cude
def threeD(mframe, threeDPts):
    threeDP = np.int32(threeDPts)
    threeDPts=threeDP.reshape(-1, 2)
    J=[threeDPts[:4]]
    J1=[threeDPts[4:]]
    #bottom contour
    mframe = cv2.drawContours(mframe, J, -1, (150, 255, 70), 3)
    for i, j in zip(range(4), range(4, 8)):
        mframe = cv2.line(frame, tuple(threeDPts[i]), tuple(threeDPts[j]), (0, 150, 255), 3)
        #top contour
    mframe = cv2.drawContours(mframe, J1, -1, (255, 0, 150), 3)
    return mframe

threeD_axis = np.float32([[0, 0, 0],
                            [0, 500, 0],
                            [500, 500, 0],
                            [500, 0, 0],
                            [0, 0, -300],
                            [0, 500, -300],
                            [500, 500, -300],
                            [500, 0, -300]])
font = cv2.FONT_HERSHEY_SIMPLEX
#using the source points as reference points
srcx=500
srcy=500
#computing the shape of an image
(x, y, ch) = image.shape
win_bottom = 8
win_top = 10
while True:
    #reading the video frame by frame
    ret,frame = cap.read()
    #storing the frames for further process
    frame1 = frame.copy()
    ro, cl, chs = frame.shape
    bottom = Movingaveragefilter(win_bottom,15)
    top = Movingaveragefilter(win_top,7)
    #print(bottom)
    #print(top)
    #converting the frames to gray scale
    f1_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, f1_thresh = cv2.threshold(f1_gray, 200, 255, 0)
    #finding the contours
    contours, heirarchy = cv2.findContours(f1_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    max_area_c = np.zeros((1,1,2),dtype=int)
    if bottom.length() < win_bottom:
        bottom.includeshapes(contours)
        print(contours)

    else:
        bottom.includeshapes(contours)
        contours = bottom.avg().astype(int)

    for c in contours:
        arcLength = cv2.arcLength(c, True)
        aprx = cv2.approxPolyDP(c, 0.02 * arcLength, True)
        area = cv2.contourArea(c)
        if area > 2000 and area < 22600:
            max_area_c = c
            if len(aprx) == 4:
                #computing the homography matrix
                Homograph_tag, inverse_Homograph_tag = homographyMatrix('bottom_right', srcx, srcy)
                #using the source points and warp perspective, computing the frame
                tagwarp = cv2.warpPerspective(frame, Homograph_tag, (srcx, srcy))
                #converting it into grayscale frames
                tagwarp_gray = cv2.cvtColor(tagwarp, cv2.COLOR_BGR2GRAY)
                #threesholding the gra frames again
                ret, tagwarp_thresh = cv2.threshold(tagwarp_gray, 200, 255, cv2.THRESH_BINARY)
                #using thresholded warpped images, computing the orientation of the tag
                OID = get_i_j(tagwarp_thresh)
                #computing the tag id
                final_tag_id = tagID(tagwarp_thresh, OID)
                xx = aprx[0][0][0]
                yy = aprx[0][0][1]
                #deawing the contours in the stored frame-frame1
                cv2.drawContours(frame1,[c],0,(255, 0, 150),9)
                #adding texts to the frame1 as a part of tag detection
                cv2.putText(frame1, final_tag_id, (xx-50, yy-50), font, 1, (0, 150, 225), 2, cv2.LINE_AA)
                #calculating the homography matrix and inverse homography matrix using the above datas
                Homograph_img, inverse_Homograph_img = homographyMatrix(OID, x, y)
                #projecting the testudo image into the frames
                image_warp = cv2.warpPerspective(image, inverse_Homograph_img, (cl, ro))
                img_warp_gray = cv2.cvtColor(image_warp, cv2.COLOR_BGR2GRAY)
                ret, warp_img_thresh = cv2.threshold(img_warp_gray, 0, 250, cv2.THRESH_BINARY_INV)
                #creating a mask function
                mas = cv2.bitwise_and(frame, frame,mask=warp_img_thresh)
                #merging it with the frame with warped image
                merge = cv2.add(mas, image_warp)
                #computing the ktr matrix
                K,R,T = KRT_Mat(Homograph_img, inverse_Homograph_img)
                p=np.zeros((1,4))
                #projecting the 3-D cube
                threeDPts, nm= cv2.projectPoints(threeD_axis, K,R,T, p)
                threeDframe = threeD(frame, threeDPts)

#resizing the frames
    merge=cv2.resize(merge,(1000,600))
    frame1=cv2.resize(frame1,(1000,600))
    threeDframe=cv2.resize(threeDframe,(1000,600))


    #videoWriter.write(frame1)
    #videoWriter1.write(merge)
    videoWriter.write(threeDframe)

    #cv2.imshow("f1_thresh",f1_thresh)
    #cv2.imshow("Tag_detection", frame1)
    cv2.imshow("imposing testudo", merge)
    cv2.imshow("3D Cube", threeDframe)

    key = cv2.waitKey(1)
    if key == 27:
        break


cap.release()
cv2.destroyAllWindows()
