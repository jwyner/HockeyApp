#include "FlyCapture2.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/video/tracking.hpp"
#include <opencv2/video/video.hpp>

#include <chrono>
#include <sstream>
#include <string>
#include <iostream>
#include <cmath>
#include <math.h>
#include <queue>
#include <thread>
#include <mutex>
using namespace FlyCapture2;
using namespace cv;
using namespace std;
//ostringstream ss;
class Computation
{
public:
	void pushDisk(Mat img);
	void pushBlue(Mat img);
	void pushYellow(Mat img);
	void pushReg(Mat img);
	void pushHsv(Mat img);
	void popDisk();
	void popBlue();
	void popYellow();
	bool isDiskEmpty();
	bool isBlueEmpty();
	bool isYellowEmpty();
	Mat diskFront();
	Mat blueFront();
	Mat yellowFront();
	void run();
	Rect crop();
	Mat convertToHSV(Mat src);
protected:
	void computation();
	Mat convertToGray(Mat src);
	Mat onlyGreen(Mat src);
	Mat onlyBlue(Mat src);
	Mat onlyYellow(Mat src);
	Point track(Mat src);
	void removeSmallObjects(Mat src);
	void predictPuckLocation(Mat source, Point currentLocation, float vX, float vY, float a);
	Mat runKalman(bool isPuck, bool isFirst, KalmanFilter kf, vector<Point> gV,
		vector<Point> kalmanV, Point p, Mat source, Mat_<float> measurement, int counter);
		
};