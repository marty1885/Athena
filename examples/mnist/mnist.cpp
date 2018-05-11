#include <Athena/Athena.hpp>
#include <Athena/Backend/XtensorBackend.hpp>
#include <Athena/Backend/NNPACKBackend.hpp>

#include "mnist_reader.hpp"

#include <iostream>
#include <chrono>

using namespace std::chrono;

int maxElementIndex(const std::vector<float>& vec)
{
	return std::distance(vec.begin(), std::max_element(vec.begin(), vec.end()));
}

At::Tensor imagesToTensor(const std::vector<std::vector<uint8_t>>& arr, At::Backend& backend)
{
	std::vector<float> res;
	res.reserve(arr.size()*arr[0].size());
	for(const auto& img : arr)
	{
		for(const auto v : img)
			res.push_back(v/255.f);
	}

	return At::Tensor(res,{(intmax_t)arr.size(),(intmax_t)arr[0].size()}, backend);
}

std::vector<float> onehot(int ind, int total)
{
	std::vector<float> vec(total);
	for(int i=0;i<total;i++)
		vec[i] = (i==ind)?1.f:0.f;
	return vec;
}

At::Tensor labelsToOnehot(const std::vector<uint8_t>& labels, At::Backend& backend)
{
	std::vector<float> buffer;
	buffer.reserve(labels.size()*10);

	for(const auto& label : labels)
	{
		std::vector<float> onehotVec = onehot(label, 10);
		for(const auto v : onehotVec)
			buffer.push_back((float)v);
	}

	return At::Tensor(buffer, {(intmax_t)labels.size(), 10}, backend);
}

int main()
{
	At::XtensorBackend backend;

	//Use the NNPACK backend to accelerate things. Remove if NNPACK is not avliable
	At::NNPackBackend nnpBackend;
	backend.useAlgorithm<At::FCForwardFunction>("fullyconnectedForward", nnpBackend);
	backend.useAlgorithm<At::FCBackwardFunction>("fullyconnectedBackward", nnpBackend);

	At::SequentialNetwork net(&backend);

	auto dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>("../mnist");
	At::Tensor traningImage = imagesToTensor(dataset.training_images, backend);
	At::Tensor traningLabels = labelsToOnehot(dataset.training_labels, backend);
	At::Tensor testingImage = imagesToTensor(dataset.test_images, backend);
	At::Tensor testingLabels = labelsToOnehot(dataset.test_labels, backend);

	net.add(At::FullyConnectedLayer(784,50));
	net.add(At::SigmoidLayer());
	net.add(At::FullyConnectedLayer(50,10));
	net.add(At::SigmoidLayer());
	net.compile();

	net.summary({At::Shape::None, 784});

	At::NestrovOptimizer opt;
	opt.alpha_ = 0.1f;
	At::MSELoss loss;

	size_t epoch = 10;
	size_t batchSize = 16;

	int count = 0;
	auto onBatch = [&](float l)
	{
		int sampleNum = traningImage.shape()[0];
		std::cout << count << "/" << sampleNum << "\r" << std::flush;
		count += batchSize;
	};

	auto onEpoch = [&](float l)
	{
		std::cout << "Epoch Loss: " << l << std::endl;
		count = 0;
	};

	high_resolution_clock::time_point t1 = high_resolution_clock::now();

	net.fit(opt,loss,traningImage,traningLabels,batchSize,epoch,onBatch,onEpoch);

	high_resolution_clock::time_point t2 = high_resolution_clock::now();
	duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
	std::cout << "It took me " << time_span.count() << " seconds." << std::endl;

	int correct = 0;
	for(intmax_t i=0;i<testingImage.shape()[0];i++)
	{
		At::Tensor x = testingImage.slice({i},{1});
		At::Tensor res = net.predict(x);
		int predictLabel = maxElementIndex(res.host());
		if(predictLabel == dataset.test_labels[i])
			correct++;
	}
	std::cout << "Accuracy: " << correct/(float)testingImage.shape()[0] << std::endl;
}
