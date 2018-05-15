#pragma once
#include <opencv2/opencv.hpp>

#include <stdio.h>
#include <tchar.h>
#include <iostream>
#include <string>
#include <vector>
#include <tuple>
#include <memory>

using namespace std;

class CNNLayer {
public:
	CNNLayer() {}

	/**
		This function executes a layer's functionality, meaning it passes through the input image and
		manipulates it according to the layer. This function is generic and should be implemented by subclasses.

		@param image The matrix to be manipulated
		@return The manipulated matrix
	*/
	virtual vector<cv::Mat> execute(vector<cv::Mat> image) {
		cout << "Execute function called on parent class with no implementation." << endl;
		return cv::Mat_<int>(0, 0);
	}

	/**
		This function prints out a layer's description and attributes. Each layer is in charge of implementing this function
		and how the layer should be printed out.
	*/
	virtual void printLayer() {
		cout << "Print Layer function called on parent class with no implementation." << endl;
	}
};