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

class PoolingLayer : public CNNLayer {
private:
	int subsecWidth, subsecHeight, slideX, slideY;
public:
	PoolingLayer(int mySubsecWidth, int mySubsecHeight, int mySlideX, int mySlideY) :CNNLayer()
	{
		subsecWidth = mySubsecWidth;
		subsecHeight = mySubsecHeight;
		slideX = mySlideX;
		slideY = mySlideY;
	}

	vector<cv::Mat> execute(vector<cv::Mat> image) {
		vector<cv::Mat> downsampledImg;
		// Assumes that all 2D Mat's in image are the same size
		int oldWidth = image.at(0).cols;
		int oldHeight = image.at(0).rows;
		int newWidth = (oldWidth - subsecWidth) / slideX + 1;
		int newHeight = (oldHeight - subsecHeight) / slideY + 1;

		// Initialize downsampledImg
		for (int imgChannel = 0; imgChannel < image.size(); imgChannel++) {
			cv::Mat imgLayer = cv::Mat::zeros(newHeight, newWidth, CV_64FC1);
			downsampledImg.push_back(imgLayer);
		}

		int x = 0;
		int y = 0;
		while (y + subsecHeight <= oldHeight) {
			while (x + subsecWidth <= oldWidth) {
				for (int imgChannel = 0; imgChannel < downsampledImg.size(); imgChannel++) {
					cv::Mat subImage = cv::Mat(image.at(imgChannel), cv::Rect(x, y, subsecWidth, subsecHeight));
					double max = maxPool(subImage);
					downsampledImg.at(imgChannel).col(x).row(y) = max;
				}
				x += slideX;
			}
			y += slideY;
		}

		print3DMat("Downsampled Image", downsampledImg);

		return downsampledImg;
	}

	double maxPool(cv::Mat subImage) {
		double max = 0.0;
		for (int y = 0; y < subImage.rows; y++) {
			for (int x = 0; x < subImage.cols; x++) {
				double nextVal = subImage.at<double>(y, x);
				if (nextVal > max) {
					max = nextVal;
				}
			}
		}
		return max;
	}

	void print3DMat(string title, vector<cv::Mat> matrix) {
		cout << title << ":" << endl;
		for (int channel = 0; channel < matrix.size(); channel++) {
			cout << " - Channel: " << channel << endl << matrix.at(channel) << endl;
		}
		cout << endl;
	}

	void printLayer() {
		cout << "Pooling Layer" << endl;
		cout << "Subsection Width: " << subsecWidth << ", Subsection Height: " << subsecHeight <<
			", Slide X: " << slideX << ", Slide Y: " << slideY << endl << endl;
	}
};