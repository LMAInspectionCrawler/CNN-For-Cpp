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
	vector<double> weights;		// Each weight is a connection between the node and a 3D matrix's element. Each element has a connection.
public:
	/**
		Constructor method for a Node. This node is used in a fully connected layer to generate a classification score.

		@param connectionNum The number of connections this node will make. This should be equal to the amount of elements of the 3D
			matrix input into the Fully Connected Layer
		@param low_inc The lower bound to randomly generate weights and biases with (inclusive)
		@param hich_exc The upper bound to randomly generate weights and biases with (exclusive)
	*/
	Node(int connectionNum, double low_inc = 0.01, double high_exc = 1.0) {
		bias = myRandom(low_inc, high_exc);

		for (int i = 0; i < connectionNum; i++) {
			double randomWeight = myRandom(low_inc, high_exc);
			weights.push_back(randomWeight);
		}
	}

	/**
		Generates a random value between a min (inclusive) and max (exclusive)

		@param min The lower bound (inclusive)
		@param max The upper bound (exclusive)
	*/
	double myRandom(double min, double max) {
		double val = (double)rand() / (RAND_MAX + 1);
		return min + val * (max = min);
	}

	/**
		This function evalutes a score for the input matrix and a particular node.
		If used as the last layer, the returned score will be the classification score

		@param image The input matrix
		@return The score for the input matrix and this node
	*/
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

	/**
		This function prints out the node's bias and weights
	*/
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

	/**
		Constructor method for a Fully Connected Layer. This layer evaluates scores for an input matrix.
		Each score can be considered the classification probability for an image, and the amount of scores is equl to the amount
		of nodes.
		For example, a scoring of {Class 1 (Washer): 0.02, Class 2 (Tape): 0.93} means that the image input is probably a tape.

		@param myNodeNum The amount of classifications to generate. These are not labeled as "Washer" or "Tape." Instead, you have 
			two classifications and the index of the score corresponds to the class.
	*/
	FullyConnectedLayer(int myNodeNum) :CNNLayer()
	{
		nodeNum = myNodeNum;
	}

	/**
		The fully connected layer is unique because, it does not use the execute function to manipulate an image like the Covolutional layer,
		RELU layer, or Pooling layer do. Instead, you should call the score(image) function that will generator a vector of scores.

		Do not use this method! This only exists because FullyConnectedLayer inherits from CNNLayer. This can be refactored later.
		TODO: Refactor CNNLayer into a new subclass that has the execute function and have Convolutional, RELU, and Pooling inherit from. That
		way this function can be removed from the Fully Connected layer.

		@param image The input matrix
		@return An unmodified matrix
	*/
	vector<cv::Mat> execute(vector<cv::Mat> image) {
		cout << "A fully connected layer cannot execute any modifications to an image."
			<< endl << "Please use vector<double> score(vector<cv::Mat> image)." << endl;
		return image;
	}

	/**
		This function classifies an image into a vector of scores.
		Each score can be considered the classification probability for an image, and the amount of scores is equl to the amount
		of nodes.
		For example, a scoring of {Class 1 (Washer): 0.02, Class 2 (Tape): 0.93} means that the image input is probably a tape.

		@param image The input matrix to be classified
	*/
	vector<double> score(vector<cv::Mat> image) {
		vector<double> scores;

		for (int classIndex = 0; classIndex < nodes.size(); classIndex++) {
			double classScore = nodes.at(classIndex).evaluate(image);
			scores.push_back(classScore);
		}
		printScores(scores);
		return scores;
	}

	/**
		This function creates all of the nodes for this layer. It uses the connectionNum, because each node needs to know how many
		connections to make, since each node has a weighted connection to every element in an input matrix.
	*/
	void initializeNodes(int connectionNum) {
		for (int i = 0; i < nodeNum; i++) {
			Node newNode(connectionNum);
			nodes.push_back(newNode);
		}
	}

	/**
		This function prints out all of the scores.
	*/
	void printScores(vector<double> scores) {
		cout << "Scores:" << endl;
		for (int scoreIndex = 0; scoreIndex < scores.size(); scoreIndex++) {
			cout << "Class " << scoreIndex << ": " << scores.at(scoreIndex) << endl;
		}
	}

	/**
		This function prints out the layer's description and attributes.
	*/
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
