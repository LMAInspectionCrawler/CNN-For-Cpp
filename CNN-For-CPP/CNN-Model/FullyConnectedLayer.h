#pragma once

#include <stdio.h>
#include <tchar.h>
#include <iostream>
#include <string>
#include <vector>
#include <tuple>
#include <memory>

#include <opencv2/opencv.hpp>
#include "CNNLayer.h"

class Node {
private:
	double bias;
	vector<double> weights;
public:
	Node(int connectionNum, double low_inc = 0.01, double high_exc = 1.0) {
		bias = myRandom(low_inc, high_exc);

		for (int i = 0; i < connectionNum; i++) {
			double randomWeight = myRandom(low_inc, high_exc);
			weights.push_back(randomWeight);
		}
	}

	double myRandom(double min, double max) {
		double val = (double)rand() / (RAND_MAX + 1);
		return min + val * (max = min);
	}

	double evaluate(vector<cv::Mat> image) {
		if (weights.size() != image.size() * image.at(0).rows * image.at(0).cols) {
			cout << "Improper weight count for image dimensions" << endl;
			return 0.0;
		}

		double score = 0.0;
		int weightIndex = 0;
		for (int imgChannel = 0; imgChannel < image.size(); imgChannel++) {
			for (int row = 0; row < image.at(imgChannel).rows; row++) {
				for (int col = 0; col < image.at(imgChannel).cols; col++) {
					score += weights.at(weightIndex) * image.at(0).at<double>(row, col);
					weightIndex++;
				}
			}
		}
		score += bias;
		return score;
	}

	void printNode() {
		cout << "Bias: " << bias << endl;
		for (int i = 0; i < weights.size(); i++) {
			cout << "Weight " << i << ": " << weights.at(i) << endl;
		}
	}
};

class FullyConnectedLayer : public CNNLayer {
private:
	vector<Node> nodes;
	int nodeNum;
public:
	FullyConnectedLayer(int myNodeNum) :CNNLayer()
	{
		nodeNum = myNodeNum;
	}

	vector<cv::Mat> execute(vector<cv::Mat> image) {
		cout << "A fully connected layer cannot execute any modifications to an image."
			<< endl << "Please use vector<double> score(vector<cv::Mat> image)." << endl;
		return image;
	}

	vector<double> score(vector<cv::Mat> image) {
		vector<double> scores;

		for (int classIndex = 0; classIndex < nodes.size(); classIndex++) {
			double classScore = nodes.at(classIndex).evaluate(image);
			scores.push_back(classScore);
		}
		printScores(scores);
		return scores;
	}

	void initializeNodes(int connectionNum) {
		for (int i = 0; i < nodeNum; i++) {
			Node newNode(connectionNum);
			nodes.push_back(newNode);
		}
	}

	void printScores(vector<double> scores) {
		cout << "Scores:" << endl;
		for (int scoreIndex = 0; scoreIndex < scores.size(); scoreIndex++) {
			cout << "Class " << scoreIndex << ": " << scores.at(scoreIndex) << endl;
		}
	}

	void printLayer() {
		cout << "Fully Connected Layer" << endl;
		cout << "Node number: " << nodeNum << endl;
		for (int i = 0; i < nodes.size(); i++) {
			cout << "Node " << i << endl;
			nodes.at(i).printNode();
		}
		cout << endl;
	}
};
