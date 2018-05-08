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
	vector<vector<cv::Mat>> filters;

public:
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

	void printLayer() {
		cout << "Convolutional Layer" << endl;
		cout << "Filter number: " << filterNum << ", Subsection Width: " << subsecWidth << ", Subsection Height: " << subsecHeight <<
			", Slide X: " << slideX << ", Slide Y: " << slideY << ", Channel number: " << channels << endl;
		printFilters();
	}

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