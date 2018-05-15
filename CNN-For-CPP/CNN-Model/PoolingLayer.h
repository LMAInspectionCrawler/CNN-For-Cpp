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

	/**
		Constructor method for a Pooling layer. This layer downsamples a matrix into a smaller matrix, with only the most relevant features
		for classification. This layer takes a subsection of the input matrix, and replaces the entire subsection with the highest number from that
		subsection. Then, it slides over to the next subsection.

		@param mySubsecWidth The width of the image subsections that you will downsample
		@param mySubsecHeight The height of the image subsections that you will downsample
		@param mySlideX The amount to slide over in the x direction after each subsection has been downsampled
		@param mySlideY The amount to slide over in the y direction after each subsection has been downsampled
	*/
	PoolingLayer(int mySubsecWidth, int mySubsecHeight, int mySlideX, int mySlideY) :CNNLayer()
	{
		subsecWidth = mySubsecWidth;
		subsecHeight = mySubsecHeight;
		slideX = mySlideX;
		slideY = mySlideY;
	}

	/**
		This function implements how the pooling layer manipulates the input matrix.

		@param image The matrix to be manipulated
		@return a new matrix of the same depth dimension, but smaller x and y dimensions. The matrix only has the maxes from the input matrix's
			subsections
	*/
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

	/**
		Finds the maximum value in a subsection of a matrix
		
		@param subImage The subsection of a matrix
	*/
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

	/**
		This function convienently prints out a 3D matrix.
		TODO: Refactor this function into a Utilities class

		@param title The title of the 3D matrix
		@param matrix The 3D matrix to be printed
	*/
	void print3DMat(string title, vector<cv::Mat> matrix) {
		cout << title << ":" << endl;
		for (int channel = 0; channel < matrix.size(); channel++) {
			cout << " - Channel: " << channel << endl << matrix.at(channel) << endl;
		}
		cout << endl;
	}

	/**
		This function prints out the layer's description and attributes.
	*/
	void printLayer() {
		cout << "Pooling Layer" << endl;
		cout << "Subsection Width: " << subsecWidth << ", Subsection Height: " << subsecHeight <<
			", Slide X: " << slideX << ", Slide Y: " << slideY << endl << endl;
	}
};