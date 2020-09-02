//written by Jacob Wyner
#include "FlyCapture2.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/video/tracking.hpp"
#include <opencv2/video/video.hpp>

#include <chrono>
#include <string>
#include <iostream>
#include <cmath>
#include <math.h>
#include <queue>
#include <thread>
#include <mutex>
#include "Computation.h"
#include "CameraInterface.h"
# define M_PI           3.14159265358979323846  /* pi */
/*
useful links:
https://www.ptgrey.com/support/downloads/10504
https://www.ptgrey.com/tan/10861
http://docs.opencv.org/master/dd/d6a/classcv_1_1KalmanFilter.html#a0657173e411acbf40d2d3c6b46e03b19&gsc.tab=0
http://opencvexamples.blogspot.com/2014/01/kalman-filter-implementation-tracking.html
http://www.morethantechnical.com/2011/06/17/simple-kalman-filter-for-tracking-using-opencv-2-2-w-code/
*/

////////////////////////////////////////////////////////////////////////////////
//constants:
Mat I = Mat::eye(4, 4, DataType<float>::type);
Mat measure(4, 1, DataType<float>::type);

int minX = 222;
int minY = 10;
int maxX = 222 + 250;
int maxY = 10 + 445;

unsigned int type = CV_32F;

bool isRobotGoal = false;
bool isHumanGoal = false; 

RNG rng(12345);


void printScreens(Mat src, Mat green,Mat blue,Mat yellow) {
	//display images of all three objects
	imshow("green", green);//show the original image with centers drawn
	moveWindow("green", 10, 20);
	imshow("blue", blue);
	moveWindow("blue", 1000, 20);
	imshow("regular", src);
	moveWindow("regular", 10, 500);
	imshow("yellow", yellow);
	moveWindow("yellow", 1000, 500);
}
void drawVector(float vX, float vY, float x, float y, float e, Mat source) {
	//display vector of puck using arrow image with varying size and direction depending on velocity
	float c = 10;
	float d = 1000;
	Point one(x, y);
	Point two(c*vX + x, c*vY + y);
	Point distantTwo(d*vX + x, d*vY + y);
	arrowedLine(source, one, two, (0, 0, 255), 3, 4);
	if (e > 1) {
		line(source, one, distantTwo, (0, 255, 0), 1, 4);
	}
}

////////////////////////////////////////////////////////////////////////////////

int main()
{	//initiate threads to perform tracking and velocity calculations
	thread t1(&CameraInterface::getImage,CameraInterface());
    	thread t2(&Computation::run, Computation());
	t1.join();
	t2.join();
}

