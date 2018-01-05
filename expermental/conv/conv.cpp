#include <Athena/Athena.hpp>
#include <Athena/XtensorBackend.hpp>
#include <Athena/NNPACKBackend.hpp>

#include <iostream>
#include <chrono>

#include <string.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;

Mat loadImage(std::string path)
{
	Mat image;
	image = imread(path, 1);
	cv::cvtColor(image, image, CV_BGR2GRAY);
	Mat img = image.clone();
	image.convertTo(img, CV_32FC1);
	return img/255.f;
}

At::Tensor mat2Tensor(const Mat& image, At::Backend& backend)
{
	std::vector<float> data(image.size().height*image.size().width);
	memcpy(&data[0], image.data, data.size()*sizeof(float));
	return At::Tensor(data, {1,1,image.size().height,image.size().width}, backend);
}

Mat tensor2Mat(const At::Tensor& image)
{
	Mat res(image.shape()[2]*image.shape()[1]*image.shape()[0], image.shape()[3], CV_32F);
	auto data = image.host();
	memcpy(res.data, &data[0], data.size()*sizeof(float));
	return res;
}

int main()
{
	At::XtensorBackend backend;
	At::NNPackBackend nnpBackend;

	backend.useAlgorithm<At::Conv2DForward>("conv2DForward",nnpBackend); //Works!

	auto algo = backend.getAlgorithm<At::Conv2DForward>("conv2DForward");

	Mat image = loadImage("banner_640199_gc3lf.jpg");
	At::Tensor img = mat2Tensor(image, backend);
	At::Tensor bias({0,0,0,0}, {4}, backend);
	At::Tensor kernel({
			0,0,0,
			0,1,0,
			0,0,0,

			0.11111,0.11111,0.11111,
			0.11111,0.11111,0.11111,
			0.11111,0.11111,0.11111,

			0,-1,0,
			-1,4,-1,
			0,-1,0,

			-1,-1,-1,
			-1, 8,-1,
			-1,-1,-1}, {4,1,3,3}, backend);
	At::Tensor conv = algo(img, kernel, bias, {{1,1}});

	Mat res = tensor2Mat(conv.abs());
	imshow("display", res);
	waitKey(0);
}
