from sys import argv
from lenalib import *

def main():
    
    video = cv.VideoCapture(f'data/{argv[1]}.mp4') # Specify the video's file name
    template = cv.imread('data/Lena.png')

    while True:
        read, frame = video.read()
        if not read:
            break

        corners, contours, binaryFrame = findCorners(frame) 
        #cv.drawContours(frame,contours,-1,(0,255,0), 3)

        # Draw corners
        # for i in range(4):
        #     for corner in corners:
        #         cv.circle(frame, (corner[i][0],corner[i][1]), 2, (0,0,255), 4)

        for i, tag in enumerate(corners):
            H = estimateInverseHomography(tag, (200,200))
       
            #binaryTag = warp(H, binaryFrame, (200,200))
            detectedTag = cv.warpPerspective(frame, H, (200, 200))
            grayScale = cv.cvtColor(detectedTag, cv.COLOR_BGR2GRAY)
            _, binaryTag = cv.threshold(grayScale, 200, 255, cv.THRESH_BINARY)

            orientation, ID = decode(binaryTag)
            #template = cv.resize(template, binaryTag.shape)
            matchedTemplate = rotate(template, orientation)
            H1 = estimateForwardHomography(tag, matchedTemplate.shape[0:2])

            #frame1 = warp(H1, matchedTemplate, frame.shape)
            frame1 = cv.warpPerspective(matchedTemplate, H1, (frame.shape[1], frame.shape[0]))

            frame2 = renderMask(frame, contours[i])
            frame = cv.bitwise_or(frame1, frame2)

            print(f'ID[{i+1}] = {ID}')
            
        # Display live frame
        imshow('Live', frame)

        if cv.waitKey(1) >= 0:
            break

    cv.destroyAllWindows()

if __name__ == '__main__':
    main()