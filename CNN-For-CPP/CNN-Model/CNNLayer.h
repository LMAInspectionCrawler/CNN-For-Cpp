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

	virtual vector<cv::Mat> execute(vector<cv::Mat> image) {
		cout << "Execute function called on parent class with no implementation." << endl;
		return cv::Mat_<int>(0, 0);
	}

	virtual void printLayer() {
		cout << "Print Layer function called on parent class with no implementation." << endl;
	}
};