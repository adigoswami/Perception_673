from mylib import *
from sys import argv

def main():
    polyDegree = 1

    videoIndex = int(argv[1])
    fileName = ['challenge_video.mp4', 'project_video.mp4']
    video = cv.VideoCapture(f'data/{fileName[videoIndex]}')
    outputVideo = cv.VideoWriter(f'output/{fileName[videoIndex]}.mp4', cv.VideoWriter_fourcc(*'XVID'), 30, (int(video.get(3)),int(video.get(4))))

    while True:
        read, frame = video.read()
        if not read:
            break
        
        binaryWarped = binarize(videoIndex, frame.copy())
        leftFit, rightFit, _ = laneFit(binaryWarped, polyDegree)

        # Generate the datapoints        
        points = np.linspace(0, binaryWarped.shape[0]-1, binaryWarped.shape[0])
        Points = constructRegressionMatrix(points, polyDegree)
        LeftFit_x = Points.dot(leftFit)
        rightFit_x = Points.dot(rightFit)
        leftPoints = np.array([np.vstack([LeftFit_x, points]).T])
        rightPoints = np.array([np.flipud(np.vstack([rightFit_x, points]).T)])
        points = np.hstack((leftPoints, rightPoints))

        # Draw the area enclosed by the lanes
        channel = np.zeros_like(binaryWarped).astype(np.uint8)
        colorWarped = np.dstack((channel, channel, channel))
        cv.fillPoly(colorWarped, np.int_([points]), (0,255, 0))
        cv.polylines(colorWarped, np.int32([leftPoints]), isClosed=False, color=(255,0,0), thickness=20)
        cv.polylines(colorWarped, np.int32([rightPoints]), isClosed=False, color=(0,0,255), thickness=20)
        unwarped = warp(videoIndex, colorWarped, undo=True)

        frame = cv.addWeighted(frame, 1, unwarped, 0.5, 0)
        
        m = round(rightFit[0],2)
        if m < -0.04:
            turn = 'Turning Right'
        else:
            turn = 'Going Straigth'
        cv.putText(frame, turn, (50,50), cv.FONT_HERSHEY_DUPLEX, 1.5, (0,255, 0), 2, cv.LINE_AA)
        cv.putText(frame, f'm = {m}', (50,100), cv.FONT_HERSHEY_DUPLEX, 1.5, (0,255, 0), 2, cv.LINE_AA)
        imshow('Live', frame)
        outputVideo.write(frame)

        if cv.waitKey(1) >= 0:
            break

if __name__ == '__main__':
    main()