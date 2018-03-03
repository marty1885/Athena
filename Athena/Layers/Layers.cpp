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
	if((bool)w_ != true)
		w_ = weightInitalizer_->create({inputSize_, outputSize_}, inputSize_, outputSize_, backend());
	if((bool)b_ != true)
		b_ = At::zeros({outputSize_}, *backend());

	forwardAlgorithm_ = backend()->getAlgorithm<FCForwardFunction>("fullyconnectedForward");
	backwardAlgorithm_ = backend()->getAlgorithm<FCBackwardFunction>("fullyconnectedBackward");
}

Tensor FullyConnectedLayer::forward(const Tensor& x)
{
	if(x.dimension() != 2)
		throw AtError("Expecting a 2D tensor for a Fully Connected layer. But get " + std::to_string(x.dimension())
		+ "D. shape = " + to_string(x.shape()));
	if(!forwardAlgorithm_)
		return x.dot(w_)+b_;
	return forwardAlgorithm_(x, w_, b_);
}

void FullyConnectedLayer::backword(const Tensor& x, const Tensor& y,
	Tensor& dx, const Tensor& dy)
{
	if(backwardAlgorithm_)
		dx = backwardAlgorithm_(dy, w_);
	else
		dx = dy.dot(transpose(w_));

	dE = dy;
	dW = dot(x.transpose(), dy);
}

void FullyConnectedLayer::update(Optimizer* optimizer)
{
	optimizer->update(w_, dW);
	optimizer->update(b_, dE.sum(0));
}

BoxedValues FullyConnectedLayer::states() const
{
	BoxedValues params;
	params.set<std::string>("__type", "FullyConnectedLayer");
	params.set<BoxedValues>("weight", w_.states());
	params.set<BoxedValues>("bias", b_.states());
	return params;
}

void FullyConnectedLayer::loadStates(const BoxedValues& states)
{
	w_.loadStates(states.get<BoxedValues>("weight"));
	b_.loadStates(states.get<BoxedValues>("bias"));
}

std::vector<Tensor> FullyConnectedLayer::weights() const
{
	return {w_, b_};
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
	if(forwardAlgorithm_)
		return forwardAlgorithm_(x);
	return 1/(1+exp(-x));
}

void SigmoidLayer::backword(const Tensor& x, const Tensor& y,
	Tensor& dx, const Tensor& dy)
{
	if(backwardAlgorithm_)
		dx = backwardAlgorithm_(dy, y);
	else
		dx = dy*(y*(1-y));
}

BoxedValues SigmoidLayer::states() const
{
	BoxedValues params;
	params.set<std::string>("__type", "SigmoidLayer");
	return params;
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
	forwardAlgorithm_ = backend()->getAlgorithm<ReluForward>("reluForward");
	backwardAlgorithm_ = backend()->getAlgorithm<ReluBackward>("reluBackward");
}

Tensor ReluLayer::forward(const Tensor& x)
{
	if(forwardAlgorithm_)
		return forwardAlgorithm_(x);
	return x*(x>0.f);
}

void ReluLayer::backword(const Tensor& x, const Tensor& y,
	Tensor& dx, const Tensor& dy)
{
	if(backwardAlgorithm_)
		dx = backwardAlgorithm_(dy, y);
	dx = dy*(y>0);
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
	Tensor t = forwardAlgorithm_(x, kernel_, bias_, strides_);
	return t;
}

void Conv2DLayer::backword(const Tensor& x, const Tensor& y,
	Tensor& dx, const Tensor& dy)
{
	dx = backwardAlgorithm_(x, kernel_, dW_, db_, dy, strides_);
}

Shape Conv2DLayer::outputShape(const Shape& s)
{
	return {s[0], outputChannels_, (s[2]-windowSize_[0])/strides_[0]+1, (s[3]-windowSize_[1])/strides_[1]+1};
}

void Conv2DLayer::build()
{
	BoxedValues config;
	intmax_t inputChannels = inputChannels_;
	//TODO: Check intializing with these params is a good idea
	if((bool)kernel_)
		kernel_ = weightInitalizer_->create({outputChannels_,inputChannels,windowSize_[0], windowSize_[1]}
			, windowSize_[0]*windowSize_[1], 1, backend());
	if((bool)bias_)
		bias_ = At::zeros({outputChannels_}, *backend());
	config.set<Shape>("kernelShape", kernel_.shape());
	config.set<Shape>("stride", Shape({strides_[0], strides_[1]}));

	forwardAlgorithm_ = backend()->getAlgorithm<Conv2DForward>("conv2DForward", config);
	backwardAlgorithm_ = backend()->getAlgorithm<Conv2DBackward>("conv2DBackward", config);
}

void Conv2DLayer::update(Optimizer* optimizer)
{
	optimizer->update(kernel_, dW_);
	optimizer->update(bias_, db_);
}

std::vector<Tensor> Conv2DLayer::weights() const
{
	return {kernel_, bias_};
}

LeakyReluLayer::LeakyReluLayer(float alpha, Backend* backend)
	: ActivationLayer(backend), alpha_(alpha)
{
	setType("leakyRelu");
}

void LeakyReluLayer::build()
{
	forwardAlgorithm_ = backend()->getAlgorithm<LeakyReluForward>("leakyReluForward");
	backwardAlgorithm_ = backend()->getAlgorithm<LeakyReluBackward>("leakyReluBackward");
}

Tensor LeakyReluLayer::forward(const Tensor& x)
{
	if(forwardAlgorithm_)
		return forwardAlgorithm_(x, alpha_);
	return (x>0)*x + (x<0)*x*alpha_;
}

void LeakyReluLayer::backword(const Tensor& x, const Tensor& y,
	Tensor& dx, const Tensor& dy)
{
	if(backwardAlgorithm_)
		dx = backwardAlgorithm_(dy, y, alpha_);
	else
		dx = dy*(y>0) + alpha_*dy*(y<0);
}
