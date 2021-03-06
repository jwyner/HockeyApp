//written by Jacob Wyner
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
#include "Computation.h"
#include "CameraInterface.h"
using namespace FlyCapture2;
using namespace cv;
Computation comp;
	bool CameraInterface::getImage() {
		clock_t tPrev = 0;
		Error error;
		Camera camera;
		CameraInfo camInfo;
		// Connect the camera
		error = camera.Connect(0);
		if (error != PGRERROR_OK)
		{
			std::cout << "Failed to connect to camera" << std::endl;
			return false;
		}
		error = camera.StartCapture();
		if (error == PGRERROR_ISOCH_BANDWIDTH_EXCEEDED)
		{
			std::cout << "Bandwidth exceeded" << std::endl;
			return false;
		}
		else if (error != PGRERROR_OK)
		{
			std::cout << "Failed to start image capture" << std::endl;
			return false;
		}
		int counter = 0;
		while (true) {
			Image rawImage;
			Error error = camera.RetrieveBuffer(&rawImage);
			if (error != PGRERROR_OK)
			{
				std::cout << "capture error" << std::endl;
				continue;
			}
			//////////////////////////////////////////////////////////////////////////////////////////
			//filter image
			Image rgbImage;// convert to rgb image
			rawImage.Convert(FlyCapture2::PIXEL_FORMAT_BGR, &rgbImage);
			// convert to OpenCV Mat
			unsigned int rowBytes = (double)rgbImage.GetReceivedDataSize() / (double)rgbImage.GetRows();
			Mat src = Mat(rgbImage.GetRows(), rgbImage.GetCols(), CV_8UC3, rgbImage.GetData(), rowBytes);
			Mat src_hsv = comp.convertToHSV(src);//convert to hsv image from rgb source image
			comp.pushHsv(src_hsv);
			clock_t t = clock();
			float elapsedTime = (((float)t / CLOCKS_PER_SEC) - ((float)tPrev / CLOCKS_PER_SEC));//calculates elapsed time
			tPrev = t;
			counter++;
		}
		camera.Disconnect();
	}

