#include <Athena/Athena.hpp>
#include <Athena/Backend/XtensorBackend.hpp>
#include <Athena/Backend/NNPACKBackend.hpp>

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
	auto data = image.host<float>();
	memcpy(res.data, &data[0], data.size()*sizeof(float));
	return res;
}

void testConv(At::Backend& backend)
{
	using At::Tensor;
	At::AdaGradOptimizer optimizer;
	optimizer.alpha_ = 0.3f;

	At::NNPackBackend nnpBackend;
	// backend.useAlgorithm<At::Conv2DForward>("conv2DForward",nnpBackend); //Works!

	auto algo = backend.getAlgorithm<At::Conv2DForward>("conv2DForward");
	auto forward = backend.getAlgorithm<At::Conv2DForward>("conv2DForward");
	auto backword = backend.getAlgorithm<At::Conv2DBackward>("conv2DBackward");

	backend.useAlgorithm<At::Conv2DBackward>("conv2DBackward",nnpBackend);
	backend.useAlgorithm<At::Conv2DForward>("conv2DForward",nnpBackend);
	auto forward2 = backend.getAlgorithm<At::Conv2DForward>("conv2DForward");
	auto backword2 = backend.getAlgorithm<At::Conv2DBackward>("conv2DBackward");

	Tensor x((std::vector<float>){1,2,3,4,5,6,7,8,9},{1,1,3,3});
	Tensor k = At::ones({1,1,3,3});
	Tensor b = At::zeros({1});
	Tensor dO((std::vector<float>){1}, {1});

	{
		Tensor res = forward(x, k, b, {{1,1}});
		Tensor dE = res - dO;
		Tensor dW, db;
		Tensor foo = backword(x, k, dW, db, dE, {{1,1}});
		std::cout << res << '\n';
		std::cout << foo << '\n';
		std::cout << dW << '\n';
		std::cout << db << '\n' << '\n';
	}

	{
		std::cout << "NNPACK: " << '\n';
		Tensor res = forward2(x, k, b, {{1,1}});
		Tensor dE = res - dO;
		Tensor dW, db;
		Tensor foo = backword2(x, k, dW, db, dE, {{1,1}});
		std::cout << res << '\n';
		std::cout << foo << '\n';
		std::cout << dW << '\n';
		std::cout << db << '\n';
	}
}

int main()
{
	At::XtensorBackend backend;
	At::Tensor::setDefaultBackend(&backend);

	//backend.useAlgorithm<At::Conv2DForward>("conv2DForward",nnpBackend); //Works!
	// backend.useAlgorithm<At::Conv2DBackward>("conv2DBackward",nnpBackend);

	testConv(backend);

	/*Mat image = loadImage("banner_640199_gc3lf.jpg");
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
	At::Tensor conv = algo(img, kernel, bias, {{1,1}})/4.f;

	//Backprop
	// At::Tensor dW(kernel.shape(), backend);
	// At::Tensor db(bias.shape(), backend);
	// At::Tensor tconv = back(img, kernel, dW, db, conv, {{1,1}});



	Mat res = tensor2Mat(conv.abs());
	imshow("display", res);
	waitKey(0);*/
}
