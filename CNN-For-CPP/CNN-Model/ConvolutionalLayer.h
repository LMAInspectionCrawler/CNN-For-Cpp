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

class ConvolutionalLayer : public CNNLayer {
private:
	int filterNum, subsecWidth, subsecHeight, slideX, slideY, channels;
	vector<vector<cv::Mat>> filters;	// Each 3D filter is split into a vector of 2D Mat rectangles. Filters is a vector that contains multiple of these 3D filters.

public:

	/**
		Constructor method for a Convolutional Layer. This layer takes a subsection of the input matrix, looks for a specific feature in it by
		getting the dot product of a filter and that image subsection, and then slide over (or convolve) to another subsection. This is repeated
		for all subsections that fit into the image. Note: each subsection may be dot producted with multiple filters in a single layer to look for
		multiple features in the image.

		@param myFilterNum The amount of filters this layer will have. Each filter will start off randomized and through training will be able to recognize
			features of an image (ex. lines, blue dots, noses, faces, etc.)
		@param mySubsecWidth The width of the image subsections that you will look for the features in
		@param mySubsecHeight The height of the image subsections that you will look for the features in
		@param mySlideX The amount to slide over in the x direction after each subsection has been checked for features
		@param mySlideY The amount to slide over in the y direction after each subsection has been checked for features
	*/
	ConvolutionalLayer(int myFilterNum, int mySubsecWidth, int mySubsecHeight, int mySlideX, int mySlideY, int myChannels) :CNNLayer()
	{
		filterNum = myFilterNum;
		subsecWidth = mySubsecWidth;
		subsecHeight = mySubsecHeight;
		slideX = mySlideX;
		slideY = mySlideY;
		channels = myChannels;

		initializeFilters();
	}

	/**
		This function intializes the filter kernals with random values from 0.01 (inclusive) to 1.0 (exclusive).
		It will create filterNum amount of filters and push them into filters.
	*/
	void initializeFilters() {
		double low_inc = 0.01;
		double high_exc = 1.0;
		for (int i = 0; i < filterNum; i++) {
			// Each filter has a depth of the same amount as the input depth (RGB = 3)
			vector<cv::Mat> newFilter;
			for (int channelIndex = 0; channelIndex < channels; channelIndex++) {
				cv::Mat newFilterChannelRandom(subsecHeight, subsecWidth, CV_64FC1);
				randu(newFilterChannelRandom, low_inc, high_exc);
				newFilter.push_back(newFilterChannelRandom);
			}
			filters.push_back(newFilter);
		}
	}

	/**
		This function implements how the convolutional layer manipulates the input matrix

		@param image The matrix to be manipulated
		@return a vector of 2D matrices (essentially a 3D matrix) that contains all of the dot products with the filters and
			the input matrix. Each filter generates a 2D matrix of dot products, so the depth of the returned vector is equal to
			the amount of filters used in this layer.
	*/
	vector<cv::Mat> execute(vector<cv::Mat> image) {
		vector<cv::Mat> activationMap3D;
		int x = 0;
		int y = 0;
		int imageHeight = image.at(0).rows;
		int imageWidth = image.at(0).cols;

		for (int i = 0; i < filters.size(); i++) {
			cv::Mat activationMap2D = cv::Mat::zeros(imageHeight, imageWidth, CV_64FC1);
			activationMap3D.push_back(activationMap2D);
		}

		while (y + subsecHeight <= imageHeight) {
			while (x + subsecWidth <= imageWidth) {

				for (int filterIndex = 0; filterIndex < filters.size(); filterIndex++) {

					double dotProduct = 0.0;
					for (int imgChannel = 0; imgChannel < channels; imgChannel++) {
						cv::Mat subImage = cv::Mat(image.at(imgChannel), cv::Rect(x, y, subsecWidth, subsecHeight));
						dotProduct += subImage.dot(filters.at(filterIndex).at(imgChannel));
					}
					activationMap3D.at(filterIndex).col(x).row(y) = dotProduct;
				}
				x += slideX;
			}
			y += slideY;
		}

		cout << "Activation Map" << endl;
		for (int i = 0; i < activationMap3D.size(); i++) {
			cout << " - Layer: " << i << endl;
			cout << activationMap3D.at(i) << endl;
		}
		cout << endl;

		return activationMap3D;
	}

	/**
		This function prints out the layer's description and attributes.
	*/
	void printLayer() {
		cout << "Convolutional Layer" << endl;
		cout << "Filter number: " << filterNum << ", Subsection Width: " << subsecWidth << ", Subsection Height: " << subsecHeight <<
			", Slide X: " << slideX << ", Slide Y: " << slideY << ", Channel number: " << channels << endl;
		printFilters();
	}

	/**
		This function prints out all of the filters.
	*/
	void printFilters() {
		for (int filterIndex = 0; filterIndex < filters.size(); filterIndex++) {
			cout << " - Filter " << filterIndex << endl;
			for (int imgChannel = 0; imgChannel < channels; imgChannel++) {
				cout << " -- Channel " << imgChannel << endl << filters.at(filterIndex).at(imgChannel) << endl;
			}
		}
		cout << endl;
	}

};