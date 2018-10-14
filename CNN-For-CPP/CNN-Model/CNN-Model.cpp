// CNN-Model.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <opencv2/opencv.hpp>

#include <iostream>
#include <string>
#include <vector>
#include <tuple>

#include "ConvolutionalNeuralNetwork.h"

using namespace std;

ConvolutionalNeuralNetwork trainCNN(ConvolutionalNeuralNetwork cnn, vector<tuple<cv::Mat, string>> labeledSet, double desiredAccuracy);
ConvolutionalNeuralNetwork gradientDescentStep(ConvolutionalNeuralNetwork cnn, vector<tuple<cv::Mat, string>> labeledTrainingSet);
vector<double> backpropagation(ConvolutionalNeuralNetwork cnn, cv::Mat image, string imageLabel);
double testAccuracy(ConvolutionalNeuralNetwork cnn, vector<tuple<cv::Mat, string>> labeledTestingSet);
vector<double> averageAdjustments(vector<vector<double>> adjustments);
vector<double> testCNN(ConvolutionalNeuralNetwork cnn, cv::Mat image);
string classify(vector<double> scores);

int main()
{

	cv::Mat tinyMatrix = (cv::Mat_<double>(2, 4) << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
	cout << "tinyMatrix:" << endl << " " << tinyMatrix << endl << endl;

	ConvolutionalNeuralNetwork cnn;
	cnn.addConvolutionalLayer(3, 2, 2, 1, 1, 1);
	cnn.addActivationLayer("RELU");
	cnn.addPoolingLayer(2, 2, 1, 1);
	cnn.addFullyConnectedLayer(3);


	// trainCNN(cnn, labeledSet, .9);
	vector<double> classification = cnn.forwardPass(tinyMatrix);
	cnn.printNetwork();

	system("pause");
    return 0;
}

/**
	Trains a CNN model's weights, biases, and kernal values to a desiredAccuracy.
	There is a limit to the amount of gradient descent steps possible.

	@param cnn The cnn model
	@param labeledSet A vector of images with their accompanying labels {(img1, label1), (img2, label2), ...}. Needs to be larger than six images.
	@param desiredAccuracy The training will not step until this desiredAccuracy is met our the maximum amount of steps is reached
	@return The cnn model with the updated weights, biases, and kernal values
*/
ConvolutionalNeuralNetwork trainCNN(ConvolutionalNeuralNetwork cnn, vector<tuple<cv::Mat, string>> labeledSet, double desiredAccuracy) {
	const int MAX_STEPS = 100; // For performance reasons, we may want to cap the amount of steps even if the desired accuracy is never reached

	double accuracy = 0.0;

	// Split labeled set into 5/6 for training and 1/6 for testing accuracy
	size_t const oneSixthSize = labeledSet.size() / 6;
	vector<tuple<cv::Mat, string>> labeledTrainingSet(labeledSet.begin(), labeledSet.begin() + (5 * oneSixthSize));
	vector<tuple<cv::Mat, string>> labeledTestingSet(labeledSet.begin() + (5 * oneSixthSize), labeledSet.end());

	int stepCnt = 0;
	while (accuracy < desiredAccuracy && stepCnt < MAX_STEPS) {
		cnn = gradientDescentStep(cnn, labeledTrainingSet);
		accuracy = testAccuracy(cnn, labeledTestingSet);
		stepCnt++;
	}

	return cnn;
}

/**
	Each gradient descent step adjusts a CNN model's weights, biases, and kernal values to lower the cost of the model.
	The cost of the model is defined as the summed squares of the differences between the actual answer and expected answer.
	C(W) = (a0 - e0)^2 + (a1 - e1)^2 + ...

	@param cnn The cnn model
	@param labeledTrainingSet A vector of images with their accompanying labels {(img1, label1), (img2, label2), ...} meant for training
	the model
	@return The cnn model with the updated weights, biases, and kernal values
*/
ConvolutionalNeuralNetwork gradientDescentStep(ConvolutionalNeuralNetwork cnn, vector<tuple<cv::Mat, string>> labeledTrainingSet) {
	vector<vector<double>> adjustments;

	for (int trainImgIndex = 0; trainImgIndex < labeledTrainingSet.size(); trainImgIndex++) {
		cv::Mat img = get<0>(labeledTrainingSet.at(trainImgIndex));
		string label = get<1>(labeledTrainingSet.at(trainImgIndex));
		vector<double> adjustmentsForEachImg = backpropagation(cnn, img, label);
		adjustments.push_back(adjustmentsForEachImg);
	}

	vector<double> finalAdjustments = averageAdjustments(adjustments);
	cnn.updateParams(finalAdjustments);
	return cnn;
}

/**
	This function determines the changes to the weights, biases, and kernal values to find the optimizations 
	of the cnn model for a specific training image.

	@param cnn The cnn model
	@param image The specific training image to find optimizations for
	@param imageLabel The specific training image's label
	@return A vector of adjustments to make to the weights, biases, and kernal values (0.03, -0.15, 0.32, ...)
*/
vector<double> backpropagation(ConvolutionalNeuralNetwork cnn, cv::Mat image, string imageLabel) {
	vector<double> changes;
	// TODO implement this function

	// Determine the ideal scores
	
	// For each classification, figure out how to change the kernal values, weights, and biases

	// TODO: watch the video about partial derivatives again. Can you figure that out once and use it for the entire training,
	// or do you need redo it for each image or something? Maybe use another function that returns a vector<vector<double>
	// where the first vector<double> can be for kernal values, the second for biases, and the third for weights. And the
	// double values is how much that value affects the cost function
	return changes;
}

/**
	This function tests the accuracy of the CNN model by using a set of testing images with labels. The function
	uses the CNN model to guess the classifications of each image and compares the classification to the real label.
	Finally, the function counts the number of correct guesses and divides by the total amount of images.

	@param cnn The cnn model
	@param labeledTestingSet A vector of images with their accompanying labels {(img1, label1), (img2, label2), ...}
	meant for testing the model
	@return an accuracy value between 0.0 - 100.0
*/
double testAccuracy(ConvolutionalNeuralNetwork cnn, vector<tuple<cv::Mat, string>> labeledTestingSet) {
	double accuracy = 0.0;
	int correct = 0;

	for (int testImgIndex = 0; testImgIndex < labeledTestingSet.size(); testImgIndex++) {
		cv::Mat image = get<0>(labeledTestingSet.at(testImgIndex));
		string label = get<1>(labeledTestingSet.at(testImgIndex));

		vector<double> scores = testCNN(cnn, image);
		string classification = classify(scores);
		if (classification.compare(label) == 0) {
			correct++;
		}
	}

	accuracy = (double)(correct / labeledTestingSet.size());
	return accuracy;
}

/**
	This function takes all of the adjusments to optimize the CNN model for each training image, and combines
	them to get one list of adjusments to make that will improve the model for all cases generally. Improvement
	is measured by reducing the Cost function.
	The cost of the model is defined as the summed squares of the differences between the actual answer and expected answer.
	C(W) = (a0 - e0)^2 + (a1 - e1)^2 + ...

	@param adjustments A list of adjustments that each training image wants to make to improve the scores for itself
		{image 1 adjustments, images 2 adjustments, ...} -> {(w0, w1, ....), (w0, w1, ...), ...}
		-> {(0.15, -0.03, ....), (0.02, 0.09, ...), ...}
	@return A list of adjustments to optimize the CNN model for all training images
*/
vector<double> averageAdjustments(vector<vector<double>> adjustments) {
	vector<double> avgAdj;	// TODO make the same size as the first vector of adjustments
	for (int weightBiasKernalIndex = 0; weightBiasKernalIndex < avgAdj.size(); weightBiasKernalIndex++) {
		for (int imgIndex = 0; imgIndex < adjustments.size(); imgIndex++) {
			avgAdj.at(weightBiasKernalIndex) += adjustments.at(imgIndex).at(weightBiasKernalIndex);
		}
	}
	return avgAdj;
}

/**
	Tests a CNN model by generating scores of classifications for an image.

	@param cnn The cnn model
	@param image An image to classify
	@return A list of scores for image classification (0.89, 0.02, ...)
*/
vector<double> testCNN(ConvolutionalNeuralNetwork cnn, cv::Mat image) {
	vector<double> scores;
	scores = cnn.forwardPass(image);
	return scores;
}

/**
	Determines the classification based on score values (ie. the classification is whatever class had
	the largest score). In the future, this may include a way for objects to be classified as multiple classes.

	@param scores A list of scores for one image classification (0.89, 0.02, ...)
	@return A classification for one image (ex. "1")
*/
string classify(vector<double> scores) {
	int classIndex = -1;
	double largestScore = 0.0;
	for (int i = 0; i < scores.size(); i++) {
		if (scores.at(i) > largestScore) {
			largestScore = scores.at(i);
			classIndex = i;
		}
	}
	string classification = to_string(classIndex);
	return classification;
}
