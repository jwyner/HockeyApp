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

using namespace FlyCapture2;
using namespace cv;

# define M_PI           3.14159265358979323846  /* pi */
////////////////////////////////////////////////////////////////
//values for seeing only yellow
int lowHY = 13;
int highHY = 37;
int lowSY = 70;
int highSY = 129;
int lowVY = 157;
int highVY = 255;
//values for seeing only blue
int lowHB = 110;
int highHB = 154;
int lowSB = 101;
int highSB = 213;
int lowVB = 94;
int highVB = 154;
//values for seeing only dark green
int lowHG = 48;
int highHG = 103;
int lowSG = 10;
int highSG = 155;
int lowVG = 31;
int highVG = 81;
////////////////////////////////////////////////////////////////////////////////
const int arraySize = 30;
Point pArray[arraySize];
bool isFound = false;
float T = 1;
float frictionAccel = -1;
int counter;
queue <Mat> puck;//[vX, vY, pX, pY,id]
queue <Mat> blue;
queue <Mat> yellowOne;
queue <Mat> hsvImages;
queue <Mat> regImages;
//methods:
void Computation::computation() {
	//kalman filter set up code: initializes three seperate kalman filters, one for each object to be tracked
	KalmanFilter kfG(4, 2, 0);
	kfG.transitionMatrix = *(Mat_<float>(4, 4) << 1, 0, T, 0, 0, 1, 0, T, 0, 0, 1, 0, 0, 0, 0, 1);
	Mat_<float> measurementG(2, 1);
	measurementG.setTo(Scalar(0));
	bool isFirstG = true;
	vector<Point> gV, gKalmanV;
	KalmanFilter kfB(4, 2, 0);
	kfB.transitionMatrix = *(Mat_<float>(4, 4) << 1, 0, T, 0, 0, 1, 0, T, 0, 0, 1, 0, 0, 0, 0, 1);
	Mat_<float> measurementB(2, 1);
	measurementB.setTo(Scalar(0));
	bool isFirstB = true;
	vector<Point> bV, bKalmanV;
	KalmanFilter kfY(4, 2, 0);
	kfY.transitionMatrix = *(Mat_<float>(4, 4) << 1, 0, T, 0, 0, 1, 0, T, 0, 0, 1, 0, 0, 0, 0, 1);
	Mat_<float> measurementY(2, 1);
	measurementY.setTo(Scalar(0));
	bool isFirstY = true;
	vector<Point> yV, yKalmanV;

	char key = 0;
	counter = 0;
	clock_t tPrev = 0;
	while (true)
		{
		///////////////////////////////////////////////////////////////////////////////
		// Get the image
		if (!hsvImages.empty()) {
			Mat src_hsv = hsvImages.front();
			hsvImages.pop();
		////////////////////////////////////////////////////////////////////////////////////////
			//filter image to produce four separate images-only yellow, only blue, only green, and regular unfiltered
			Rect cropped = crop();
			src_hsv = src_hsv(cropped);//crop the image
			Mat src_filtered_green = onlyGreen(src_hsv);//produce hsv Mat image with only green showing
			Mat src_filtered_blue = onlyBlue(src_hsv);//produce hsv Mat image with only blue showing
			Mat src_filtered_yellow = onlyYellow(src_hsv);//produce hsv Mat image with only yellow showing
			removeSmallObjects(src_filtered_green);//remove small objects from image  
			removeSmallObjects(src_filtered_blue);//remove small objects from image
			removeSmallObjects(src_filtered_yellow);//^^^
			//////////////////////////////////////////////////////////////////////////////////
			//apply kalman filter
			Mat temp;
			Point g(track(src_filtered_green));//track green circle return location as point-puck
			temp = runKalman(true, isFirstG, kfG, gV, gKalmanV, g, src_hsv, measurementG, counter);
			temp.at<double>(4, 0) = counter;                      //^^^^used to be src, changed it to src_hsv, maybe delete from args altogether as it seems that algo does not need it
			puck.push(temp);//push data into puck queue
			Point b(track(src_filtered_blue));//track blue circle return location as point
			temp = runKalman(false, isFirstB, kfB, bV, bKalmanV, b, src_hsv, measurementB, counter);
			blue.push(temp);//push data into blue queue
			temp.at<double>(4, 0) = counter;
			Point yellow(track(src_filtered_yellow));
			temp = runKalman(false, isFirstY, kfY, yV, yKalmanV, yellow, src_hsv, measurementY, counter);
			temp.at<double>(4, 0) = counter;
			yellowOne.push(temp);//push data into yellow queue
			//get data from this format: [vX, vY, pX, pY,id]
			float vPuck = 0;
			float vBlue = 0;
			float vYellow = 0;
			if (!puck.empty()) {//make sure we don't try to pop an empty queue
				Mat pA = puck.front();//set Mat pA to the most recent puck value
				vPuck = sqrt(pow(pA.at<double>(1, 0), 2) + pow(pA.at<double>(1, 0), 2));//solve for resultant velocity
				puck.pop();//dequeues puck queue
				}
			if (!yellowOne.empty()) {
				Mat yA = yellowOne.front();//set Mat yA to front value
				 vYellow = sqrt(pow(yA.at<double>(1, 0), 2) + pow(yA.at<double>(1, 0), 2));//solve for resultant velocity
				yellowOne.pop();//dequeue yellow queue
				}
			if (!blue.empty()) {
				Mat bA = blue.front();//set MAt bA to most recent blue value
				 vBlue = sqrt(pow(bA.at<double>(1, 0), 2) + pow(bA.at<double>(1, 0), 2));//solve for resultant velocity
				blue.pop();//dequeue blue queue
				}

			counter++;//increment counter
			clock_t t = clock();//used to solve for elapsed time
			float elapsedTime = (((float)t / CLOCKS_PER_SEC) - ((float)tPrev / CLOCKS_PER_SEC));
			while(elapsedTime < 0.06){//keep the time consistent
					t = clock();
					elapsedTime = (((float)t / CLOCKS_PER_SEC) - ((float)tPrev / CLOCKS_PER_SEC));
				}
			tPrev = t;
			if (counter % 10  == 0) {
				cout << " t: " << elapsedTime<<" vP:"<<vPuck<< " vY:" << vYellow << " vB:" << vBlue <<" counter "<< counter<<"\n";
			}
		    }
		}

	}
	
	void Computation::pushDisk(Mat img) {puck.push(img); }
	void Computation::pushBlue(Mat img) {blue.push(img); }
	void Computation::pushYellow(Mat img) {yellowOne.push(img); }
	void Computation::pushReg(Mat img) {regImages.push(img); }
	void Computation::pushHsv(Mat img) { hsvImages.push(img); }
	void Computation::popDisk() {puck.pop(); }
	void Computation::popBlue() {blue.pop(); }
	void Computation::popYellow() {yellowOne.pop(); }
	bool Computation::isDiskEmpty() {return puck.empty(); }
	bool Computation::isBlueEmpty() {return blue.empty(); }
	bool Computation::isYellowEmpty() {return yellowOne.empty();}
	Mat Computation::diskFront() { return puck.front(); }
	Mat Computation::blueFront() { return blue.front(); }
	Mat Computation::yellowFront() { return yellowOne.front(); }



	Mat Computation::convertToGray(Mat src) {//converts image to grayscale format
		Mat src_gray;
		cvtColor(src, src_gray, CV_BGR2GRAY);
		return src_gray;
	}
	Mat Computation::convertToHSV(Mat src) {//converts image to hsv format
		Mat src_hsv;
		cvtColor(src, src_hsv, COLOR_BGR2HSV);
		return src_hsv;
	}
	Mat Computation::onlyGreen(Mat src) {//filters out all colors except green in hsv image
		Mat result;
		inRange(src, Scalar(lowHG, lowSG, lowVG), Scalar(highHG, highSG, highVG), result); //only see green colors
		return result;
	}
	Mat Computation::onlyBlue(Mat src) {//filters out all colors except blue in hsv image
		Mat result;
		inRange(src, Scalar(lowHB, lowSB, lowVB), Scalar(highHB, highSB, highVB), result); //only see blue colors
		return result;
	}
	Mat Computation::onlyYellow(Mat src) {
		Mat result;
		inRange(src, Scalar(lowHY, lowSY, lowVY), Scalar(highHY, highSY, highVY), result); //only see blue colors
		return result;
	}
	
	Point Computation::track(Mat src) {//tracks active pixels, calculates their centroid
		int counter = 1;
		int totalX = 0;
		int totalY = 0;
		for (int i = 0; i < src.rows; i++) {
			for (int j = 0; j < src.cols; j++) {
				int value = (int)src.at<uchar>(i, j);
				if (value > 200) {
					//calculate the centroid
					totalX = totalX + j;
					totalY = totalY + i;
					counter++;
					isFound = true;
				}
			}
		}
		int x = totalX / counter;
		int y = totalY / counter;
		return Point(x, y);//returns the centroid
	}
	Rect Computation::crop() {
		int x = 222;
		int y = 10;
		int height = 430;
		int width = 250;
		Rect cropped(x, y, width, height);
		return cropped;
	}
	void Computation::removeSmallObjects(Mat src) {
		erode(src, src, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
		dilate(src, src, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
	}
	void Computation::predictPuckLocation(Mat source, Point currentLocation, float vX, float vY, float a) {
		float theta = (atan(vY / vX));//angle between vY and vX in radians
		float dX = 0; float dY = 0;
		float aX = a*sin(theta);//acceleration in the x direction from friction
		float aY = a*cos(theta);//acceleration in the y direction from friction
		float c = sqrt(pow(vX, 2) + pow(vY, 2));//find resultant velocity
		float t = 0;
		float dt = 0.06375;//dt set by experimentation
		if (c > 0.2) {
			for (int i = 0; i < arraySize; i++) {
				t = t + dt;
				dX = vX*t + (1 / 2)*t*t*aX;//predicted difference in displacement in x direction
				if (dX > -4e-2 && dX < 4e-2) {//if it is very small ignore it 
					dX = 0;
				}
				dX = dX;//multiply it by a constant to adjust value
				dY = vY*t + (1 / 2)*t*t*aY;//predicted difference in displacement in y direction
				dY = dY;
				if ((currentLocation.x + dX) < 0 || (currentLocation.x + dX) > source.cols) {//check to see if the predicted value is outside the bounds of the x range
					vX = vX*-1;//flip the sign of the velocity in the x direction
					theta = (atan(vY / vX));//calculate new angle
					aX = a*sin(theta);//calcualte new acceleration in the x direction
					dX = (vX*t + (1 / 2)*t*t*aX);//calculate the new x displacement
					if (dX > -4e-2 && dX < 4e-2) {//again correct of error by removing small values
						dX = 0;
					}
					dX = dX;//multiply it by a constant to adjust the value
				}
				if ((currentLocation.y + dY) < 0 || (currentLocation.y + dY) > source.rows) {//check to see if the predicted value is outside the bounds of the y range
					vY = vY*-1;//flip the sign of the velocity in the y direction
					theta = (atan(vY / vX));  //calculate new angle
					aY = a*cos(theta);//calculate new acceleration in y direction
					dY = (vY*t + (1 / 2)*t*t*aY);//caluclate new change in displacement
					dY = dY;//multiply by constant to adjust value
				}

				Point f(currentLocation.x + dX, currentLocation.y + dY);//save the predicted next location
				currentLocation = f;//set the current location to that predicted value
				pArray[i] = f;//set the value in the array of predictions to predicted value
			}
		}
		else {
			for (int i = 0; i < arraySize; i++) {//if puck isnt moving set prediction array to current location
				Point f(currentLocation.x, currentLocation.y);//save the predicted next location
				currentLocation = f;//set the current location to that predicted value
				pArray[i] = f;//set the value in the array of predictions to predicted value
			}
		}
	}
	Mat Computation::runKalman(bool isPuck, bool isFirst, KalmanFilter kf, vector<Point> gV, vector<Point> kalmanV, Point p, Mat source, Mat_<float> measurement, int counter) {
		float max = 30;
		float min = 0;
		Mat data4d = Mat(5, 1, CV_64F);
		if (isFirst) {//initialization code
			kf.statePre.at<float>(0) = p.x;//posx
			kf.statePre.at<float>(1) = p.y;//posy
			kf.statePre.at<float>(2) = 0;//vx
			kf.statePre.at<float>(3) = 0;//vy
			setIdentity(kf.measurementMatrix);//set measurement matrix to identity matrix
			setIdentity(kf.processNoiseCov, Scalar::all(1e-4));//set covariance matrix to small value matrix
			setIdentity(kf.measurementNoiseCov, Scalar::all(1e-1));
			setIdentity(kf.errorCovPost, Scalar::all(0.2));
			gV.clear();
			kalmanV.clear();
			isFirst = false;
		}
		measurement(0) = p.x;//set x measurement array to the x value from point p
		measurement(1) = p.y;//set y measurement array to the y value from point p
		//kalman prediction
		Mat prediction = kf.predict();//set the prediction matrix
		Point predictLocationG(prediction.at<float>(0), prediction.at<float>(1));//get predicted locations
		Mat estimated = kf.correct(measurement);//update phase
		Point statePoint(estimated.at<float>(0), estimated.at<float>(1));//set state matrix
		Point measurePoint(p.x, p.y);//potentially get move this to inside if statement below
		gV.push_back(measurePoint);
		kalmanV.push_back(statePoint);
		Point test(statePoint.y, statePoint.x);
		Point error = (statePoint - measurePoint);
		float err = sqrt(pow((error.x + error.y), 2));//calculate error
		float output = min + ((max - min) / (max - min))*(err - min);//map error to output to draw scaled circle
		float vX = kf.statePost.at<float>(2);//get velocity in x direction
		float vY = kf.statePost.at<float>(3);//get velocity in y direction
		if (vY > -0.4 && vY < 0.4) { vY = 0; }//eliminates some error
		float c = sqrt(pow(vX, 2) + pow(vY, 2));//find resultant velocity
		//sets data4d matrix with values
		data4d.at<double>(0) = vX;//set x velocity
		data4d.at<double>(1) = vY;//set y velocity
		data4d.at<double>(2) = p.x;// set x position
		data4d.at<double>(3) = p.y;//set y position
		data4d.at<double>(4) = 0;//set counter as 0 for now, it is set it with correct value later
		float theta = (atan(vY / vX))*(180 / M_PI);//angle in degrees
		if (isPuck) {//check if the object being assessed is the puck
			if (c < 0.3) { c = 0; }//if c is really small set c = 0 to eliminate error
			if (c < 4) {
				frictionAccel = -1;//set coefficient of friction - value determined experimentally
				predictPuckLocation(source, measurePoint, vX, vY, frictionAccel);//return predicted puck location
			}
			else {
				frictionAccel = -2;//set coefficient of friction to higher value as speed increases
				predictPuckLocation(source, measurePoint, vX, vY, frictionAccel);//return predicted puck location
			}

		}
		return data4d;
	}
	void  Computation::run() {
		computation();
}

