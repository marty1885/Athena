#include <Athena/Layers/Layers.hpp>

using namespace At;

FullyConnectedLayer::FullyConnectedLayer(Backend* backend)
	:Layer(backend, true)
{
	setType("fullyConnected");
}

FullyConnectedLayer::FullyConnectedLayer(intmax_t input, intmax_t output, Backend* backend)
	:FullyConnectedLayer(backend)
{
	inputSize_ = input;
	outputSize_ = output;
}

FullyConnectedLayer::FullyConnectedLayer(intmax_t output, Backend* backend)
	:FullyConnectedLayer(backend)
{
	outputSize_ = output;
}

Shape FullyConnectedLayer::outputShape(const Shape& s)
{
	return Shape({Shape::None, outputSize_});
}

void FullyConnectedLayer::build()
{
	weights_.push_back(weightInitalizer_->create({inputSize_, outputSize_}, inputSize_, outputSize_, backend()));
	weights_.push_back(At::zeros({outputSize_}, *backend()));

	forwardAlgorithm_ = backend()->getAlgorithm<FCForwardFunction>("fullyconnectedForward");
	backwardAlgorithm_ = backend()->getAlgorithm<FCBackwardFunction>("fullyconnectedBackward");
}

Tensor FullyConnectedLayer::forward(const Tensor& x)
{
	if(x.dimension() != 2)
		throw AtError("Expecting a 2D tensor for a Fully Connected layer. But get " + std::to_string(x.dimension())
		+ "D. shape = " + to_string(x.shape()));
	return forwardAlgorithm_(x, weights_[0], weights_[1]);
}

void FullyConnectedLayer::backword(const Tensor& x, const Tensor& y,
	Tensor& dx, const Tensor& dy)
{
	dx = backwardAlgorithm_(dy, weights_[0]);

	dE = dy;
	dW = dot(x.transpose(), dy);
}

void FullyConnectedLayer::update(Optimizer* optimizer)
{
	optimizer->update(weights_[0], dW);
	optimizer->update(weights_[1], dE.sum(0));
}

SigmoidLayer::SigmoidLayer(Backend* backend) : ActivationLayer(backend)
{
	setType("sigmoid");
}

void SigmoidLayer::build()
{
	forwardAlgorithm_ = backend()->getAlgorithm<SigmoidForward>("sigmoidForward");
	backwardAlgorithm_ = backend()->getAlgorithm<SigmoidBackward>("sigmoidBackward");
}

Tensor SigmoidLayer::forward(const Tensor& x)
{
	return forwardAlgorithm_(x);
}

void SigmoidLayer::backword(const Tensor& x, const Tensor& y,
	Tensor& dx, const Tensor& dy)
{
	dx = backwardAlgorithm_(dy, y);
}

TanhLayer::TanhLayer(Backend* backend) : ActivationLayer(backend)
{
	setType("tanh");
}

void TanhLayer::build()
{
	forwardAlgorithm_ = backend()->getAlgorithm<TanhForward>("tanhForward");
	backwardAlgorithm_ = backend()->getAlgorithm<TanhBackward>("tanhBackward");
}

Tensor TanhLayer::forward(const Tensor& x)
{
	return forwardAlgorithm_(x);
}

void TanhLayer::backword(const Tensor& x, const Tensor& y,
	Tensor& dx, const Tensor& dy)
{
	dx = backwardAlgorithm_(dy, y);
}

ReluLayer::ReluLayer(Backend* backend) : ActivationLayer(backend)
{
	setType("relu");
}

void ReluLayer::build()
{
	forwardAlgorithm_ = backend()->getAlgorithm<TanhForward>("reluForward");
	backwardAlgorithm_ = backend()->getAlgorithm<TanhBackward>("reluBackward");
}

Tensor ReluLayer::forward(const Tensor& x)
{
	return forwardAlgorithm_(x);
}

void ReluLayer::backword(const Tensor& x, const Tensor& y,
	Tensor& dx, const Tensor& dy)
{
	dx = backwardAlgorithm_(dy, y);
}

Conv2DLayer::Conv2DLayer(intmax_t inputChannels, intmax_t outputChannels, Shape windowSize, std::array<intmax_t, 2> strides, Backend* backend)
	: Layer(backend, true)
{
	outputChannels_ = outputChannels;
	inputChannels_ = inputChannels;
	windowSize_ = windowSize;
	strides_ = strides;

	setType("conv2D");
}

Tensor Conv2DLayer::forward(const Tensor& x)
{
	if(x.dimension() != 4)
		throw AtError("Conv2D expecting a 4D tensor but got " + std::to_string(x.dimension()) + "D. Shape = " + to_string(x.shape()));
	if(x.shape()[1] != inputChannels_)
		throw AtError("Expecting " + std::to_string(inputChannels_) + " input channes, but get " + std::to_string(x.shape()[1]));
	Tensor t = forwardAlgorithm_(x, weights_[0], weights_[1], strides_);
	return t;
}

void Conv2DLayer::backword(const Tensor& x, const Tensor& y,
	Tensor& dx, const Tensor& dy)
{
	dx = backwardAlgorithm_(x, weights_[0], dW_, db_, dy, strides_);
}

Shape Conv2DLayer::outputShape(const Shape& s)
{
	return {s[0], outputChannels_, (s[2]-windowSize_[0])/strides_[0]+1, (s[3]-windowSize_[1])/strides_[1]+1};
}

void Conv2DLayer::build()
{
	BoxedValues config;
	intmax_t inputChannels = inputChannels_;
	//TODO: Check this is a good idea
	Tensor kernel = weightInitalizer_->create({outputChannels_,inputChannels,windowSize_[0], windowSize_[1]}
		, windowSize_[0]*windowSize_[1], 1, backend());
	weights_.push_back(kernel);
	weights_.push_back(At::zeros({outputChannels_}, *backend()));
	config.set<Shape>("kernelShape", weights_[0].shape());
	config.set<Shape>("stride", Shape({strides_[0], strides_[1]}));

	forwardAlgorithm_ = backend()->getAlgorithm<Conv2DForward>("conv2DForward", config);
	backwardAlgorithm_ = backend()->getAlgorithm<Conv2DBackward>("conv2DBackward", config);
}

void Conv2DLayer::update(Optimizer* optimizer)
{
	optimizer->update(weights_[0], dW_);
	optimizer->update(weights_[1], db_);
}