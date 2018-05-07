#include "targetver.h"
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
		return cv::Mat_<int>(0,0);
	}
};

class ConvolutionalLayer : public CNNLayer {
protected:
	int filterNum, subsecWidth, subsecHeight, slideX, slideY, channels;
	vector<vector<cv::Mat>> filters;

public:
	ConvolutionalLayer(int myFilterNum, int mySubsecWidth, int mySubsecHeight, int mySlideX, int mySlideY, int myChannels):CNNLayer()
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
					cout << "filters.at(" << filterIndex << "): " << endl;

					double dotProduct = 0.0;
					for (int imgChannel = 0; imgChannel < channels; imgChannel++) {
						cout << "Channel: " << imgChannel << endl << filters.at(filterIndex).at(imgChannel) << endl;
						cv::Mat subImage = cv::Mat(image.at(imgChannel), cv::Rect(x, y, subsecWidth, subsecHeight));
						dotProduct += subImage.dot(filters.at(filterIndex).at(imgChannel));
					}
					activationMap3D.at(filterIndex).col(x).row(y) = dotProduct;
				}
				x += slideX;
			}
			y += slideY;
		}

		cout << "Activation Map:" << endl;
		for (int i = 0; i < activationMap3D.size(); i++) {
			cout << "Layer: " << i << endl;
			cout << activationMap3D.at(i) << endl << endl;
		}

		return activationMap3D;
	}
};

class ConvolutionalNeuralNetwork {
protected:
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
		vector<cv::Mat> imageChannels = prepareImage(image);
		
		for (int layerIndex = 0; layerIndex < layers.size(); layerIndex++) {
			shared_ptr<CNNLayer> nextLayer = layers.at(layerIndex);
			imageChannels = nextLayer->execute(imageChannels);
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

	@param type The activation function to apply. Currently only 'RELU' is supported
	*/
	void addActivationLayer(string type = "RELU") {
		// TODO plan out method
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
		// TODO plan out method
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
		// TODO plan out method
	}

};

ConvolutionalNeuralNetwork trainCNN(ConvolutionalNeuralNetwork cnn, vector<tuple<cv::Mat, string>> labeledSet, double desiredAccuracy);
ConvolutionalNeuralNetwork gradientDescentStep(ConvolutionalNeuralNetwork cnn, vector<tuple<cv::Mat, string>> labeledTrainingSet);
vector<double> backpropagation(ConvolutionalNeuralNetwork cnn, cv::Mat image, string imageLabel);
double testAccuracy(ConvolutionalNeuralNetwork cnn, vector<tuple<cv::Mat, string>> labeledTestingSet);
vector<double> averageAdjustments(vector<vector<double>> adjustments);
vector<double> testCNN(ConvolutionalNeuralNetwork cnn, cv::Mat image);
string classify(vector<double> scores);