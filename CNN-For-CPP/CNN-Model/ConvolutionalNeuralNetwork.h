#include "targetver.h"

#include <stdio.h>
#include <tchar.h>
#include <iostream>
#include <string>
#include <vector>
#include <tuple>
#include <memory>

#include <opencv2/opencv.hpp>
#include "CNNLayer.h"
#include "ConvolutionalLayer.h"
#include "RELULayer.h"
#include "PoolingLayer.h"
#include "FullyConnectedLayer.h"

using namespace std;

class ConvolutionalNeuralNetwork {
private:
	// Note: I used shared_ptr to avoid memory leaks. I first tried unique_ptr, but you can't have a vector of unique_ptrs.
	// Please refer to https://stackoverflow.com/questions/16126578/vectors-and-polymorphism-in-c
	vector<shared_ptr<CNNLayer>> layers;

public:

	/**
	Changes the CNN's weights, biases, and kernal values.
	@param changes A list of values to add to the current weights, biases, and kernal values
	(weight0_changes, weight1_changes, ..., bias0_changes, ... kernal0_changes, ...) -> (0.13, 0.04, -0.05)
	*/
	void updateParams(vector<double> changes) {
		//TODO changes weights, biases, and kernal values
		cout << "Parameters updated" << endl;
	}

	/**
	Passes an image through the CNN to generate classification scores for an image.
	@param image The image to be classified
	@return A list of scores for image classification (0.89, 0.02, ...)
	*/
	vector<double> forwardPass(cv::Mat image) {
		vector<double> scores;
		// TODO plan out method
		vector<cv::Mat> modifiedImg = prepareImage(image);

		for (int layerIndex = 0; layerIndex < layers.size(); layerIndex++) {
			shared_ptr<CNNLayer> nextLayer = layers.at(layerIndex);

			if (typeid(*nextLayer).name() == typeid(FullyConnectedLayer).name()) {
				shared_ptr<FullyConnectedLayer> fcLayer = dynamic_pointer_cast<FullyConnectedLayer> (nextLayer);
				fcLayer->initializeNodes(modifiedImg.size() * modifiedImg.at(0).rows * modifiedImg.at(0).cols);
				scores = fcLayer->score(modifiedImg);
			}
			else {
				modifiedImg = nextLayer->execute(modifiedImg);
			}
		}

		return scores;
	}

	/**
	Since OpenCV's support for 3D Mats is very bad and limited to a depth of 4, we convert the image into a vector
	of 2D Mats
	@param image An RGB image to be classified
	*/
	vector<cv::Mat> prepareImage(cv::Mat image) {
		if (image.channels() == 3) {
			vector<cv::Mat> bgr_channels;
			split(image, bgr_channels);
			return bgr_channels;
		}
		vector<cv::Mat> imageLayers;
		imageLayers.push_back(image);
		return imageLayers;
	}

	void printNetwork() {
		printLine();
		cout << "CNN MODEL" << endl << endl;
		for (int layerIndex = 0; layerIndex < layers.size(); layerIndex++) {
			layers.at(layerIndex)->printLayer();
		}
		printLine();
	}

	void printLine() {
		for (int i = 0; i < 30; i++) {
			cout << "-";
		}
		cout << endl;
	}

	/**
	This layer applies a series of filters to each subsection of the image to recieve their dot-product for each subsection.
	Then the layer slides over a set x distance and repeats the process with the new (potentially overlapping subsection).
	Once the layer has reached the end of the row, it slides over a set y distance and repeats.
	At the end, for each filter you get a 2D rectangle with the dot-products of that filter's kernal with each subsection.
	Finally, those 2D rectangle for each filter are stacked into a 3D box with the depth of the amount of filters.
	@param filterNum The amount of features to look for in this layer. A feature doesn't mean anything at first, because the filter's
	kernal is randomized in the beginning. However, after training features start to automatically appear such as lines, curves,
	highly-contrasted subsections, etc. On futher layers, features may include noses, eyes, or even entire faces or letters.
	@param subsecWidth The width of the subsection that you will be looking at. This also affects the filter's kernal width because
	it is the same size as the subsection.
	@param subsecHeight The height of the subsection that you will be looking at. This also affects the filter's kernal height because
	it is the same size as the subsection.
	@param slideX The distance in the x direction to slide over (typically 1-4 is a good number)
	@param slideY The distance in the y direction to slide over (typically it's the same as slideX)
	*/
	void addConvolutionalLayer(int filterNum, int subsecWidth, int subsecHeight, int slideX, int slideY, int channels) {
		//CNNLayer *layer = new ConvolutionalLayer(filterNum, subsecWidth, subsecHeight, slideX, slideY);
		shared_ptr<CNNLayer> layer(new ConvolutionalLayer(filterNum, subsecWidth, subsecHeight, slideX, slideY, channels));
		layers.push_back(layer);
	}

	/**
	This layer applies an activation function to a each element in a 3D box after the convolutional layer.
	An activation function can include RELU or Sigmoid (Not implemented yet).
	RELU has been proven to better train deep networks.

	@param type The activation function to apply. Currently only 'RELU' is supported
	*/
	void addActivationLayer(string type = "RELU") {
		shared_ptr<CNNLayer> layer(new RELULayer());
		layers.push_back(layer);
	}

	/**
	This layer reduces the dimensionality of the 3D box for performance and generalization reasons. By generalization, I mean
	that features at early layers of the forward pass may look for lines and curves, while after pooling, the next layers may look
	for features of noses and eyes, and finally layers at the end of the forward pass may look for high level features like faces.
	This layer looks at a subsection of the box, finds the largest number in that subsection, and reduces the entire subsection
	to that largest number.
	|5|2|
	|8|3| --> |8|
	@param subsecWidth The width of the subsection to reduce the dimensionality of
	@param subsecHeight The height of the subsection to reduce the dimensionality of
	@param slideX The distance in the x direction to slide over (typically 1-4 is a good number)
	@param slideY The distance in the y direction to slide over (typically it's the same as slideX)
	*/
	void addPoolingLayer(int subsecWidth, int subsecHeight, int slideX, int slideY) {
		shared_ptr<CNNLayer> layer(new PoolingLayer(subsecWidth, subsecHeight, slideX, slideY));
		layers.push_back(layer);
	}

	/**
	This layer uses a series of nodes that are connected to every element in the 3D box. Each node's output can be calculated by
	the following formula, where Node_i is a node in the layer, elem_j is an element in the 3D box, weight_i_j is the connection between
	Node_i and elem_j, and bias_i is the node's bias.
	Node_i's Output = sum(elem_j * weight_i_j + bias_i) for each element in the box
	Note: The final fully connected layer should have a nodeNum equal to the amount of classifications wanted (ex. nodeNum = 10 for 0-9 classifications
	for handwritten digits)
	@param nodeNum The number of nodes in this layer
	*/
	void addFullyConnectedLayer(int nodeNum) {
		shared_ptr<CNNLayer> layer(new FullyConnectedLayer(nodeNum));
		layers.push_back(layer);
	}

};