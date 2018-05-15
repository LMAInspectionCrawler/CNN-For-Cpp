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

	/**
		Constructor method for a RELU Layer. This layer takes an input matrix, and replaces all negative values with zero.
		This is done to prevent matrix values from becoming too large.
	*/
	RELULayer() :CNNLayer()
	{
	}

	/**
		This function implements how the RELU layer manipulates the input matrix

		@param image The matrix to be manipulated
		@return a matrix of the same dimensions with all of the negative values replaced with 0 and the positive values untouched
	*/
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

	/**
		This function prints out the layer's description and attributes.
	*/
	void printLayer() {
		cout << "RELU Layer" << endl << endl;
	}
};