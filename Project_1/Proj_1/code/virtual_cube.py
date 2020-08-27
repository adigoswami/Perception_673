from cubelib import *

def main():
    
    video = cv.VideoCapture('data/multipleTags.mp4') # Specify the video's file name
    #out = cv.VideoWriter('code/demo/cubemultipleTags.avi',cv.VideoWriter_fourcc(*'XVID'), 30, (1920,1080))
    while True:
        read, frame = video.read()
        if not read:
            break

        corners, contours, binary = findCorners(frame) 
        
        #cv.drawContours(frame,contours,-1,(0,255,0), 3)
        for i in range(4):
            for corner in corners:
                cv.circle(frame, (corner[i][0],corner[i][1]), 2, (0,0,255), 4)

        for tag in corners:
            H = estimateForwardHomography(tag,(200, 200))
            #H = estimateHomography(tag)
            warped = warp(H,binary)
            #H_new = estimateInverseHomography(tag,(200, 200))
            P = Projection(H,k)
            virtualCube(frame,P)
            #arped = cv.warpPerspective(frame, H, (200, 200))
            #cv.imshow("warped", warped)
            
       # Display live frame
        #cv.namedWindow('Live', cv.WINDOW_NORMAL)
        cv.imshow('Live',frame)
        #out.write(frame)
        if cv.waitKey(1) >= 0:
            break
    
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()