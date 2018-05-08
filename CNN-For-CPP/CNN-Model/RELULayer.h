#pragma once
#include <opencv2/opencv.hpp>

#include <stdio.h>
#include <tchar.h>
#include <iostream>
#include <string>
#include <vector>
#include <tuple>
#include <memory>
#include "CNNLayer.h"

class RELULayer : public CNNLayer {
public:
	RELULayer() :CNNLayer()
	{
	}

	vector<cv::Mat> execute(vector<cv::Mat> image) {
		vector<cv::Mat> rectifiedImg = image;
		for (int imgChannel = 0; imgChannel < rectifiedImg.size(); imgChannel++) {
			for (int y = 0; y < rectifiedImg.at(imgChannel).rows; y++) {
				for (int x = 0; x < rectifiedImg.at(imgChannel).cols; x++) {
					// Replaces all negative values in the img with 0
					rectifiedImg.at(imgChannel).row(y).col(x) = max(0, rectifiedImg.at(imgChannel).row(y).col(x));
				}
			}
		}

		/*cout << "Rectified Image:" << endl;
		for (int imgChannel = 0; imgChannel < rectifiedImg.size(); imgChannel++) {
			cout << "Channel: " << imgChannel << endl << rectifiedImg.at(imgChannel) << endl << endl;
		}*/

		return rectifiedImg;
	}

	void printLayer() {
		cout << "RELU Layer" << endl << endl;
	}
};