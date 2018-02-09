#ifndef LAYERS_HPP
#define LAYERS_HPP

#include <Athena/Backend.hpp>
#include <Athena/Tensor.hpp>
#include <Athena/Utils/Shape.hpp>

namespace At
{

class Layer
{
public:
	virtual ~Layer()
	{
	}

	Layer()
	{
	}

	Layer(Backend* backend=nullptr, bool trainable=false):
		backend_(backend), trainable_(trainable)
	{
	}

	virtual Tensor forward(const Tensor& input)
	{
		return Tensor();
	}

	virtual void backword(const Tensor& x, const Tensor& y,
		Tensor& dx, const Tensor& dy)
	{
	}
	
	//Caclulate the output shape given a input shape
	virtual Shape outputShape(const Shape& s) = 0;

	bool trainable() const
	{
		return trainable_;
	}

	void setTrainable(bool val)
	{
		trainable_ = val;
	}

	const std::vector<Tensor>& weights() const
	{
		return weights_;
	}

	bool isInitialized() const
	{
		return backend_ != nullptr;
	}

	std::vector<Tensor>& weights()
	{
		return const_cast<std::vector<Tensor>&>
			(static_cast<const Layer*>(this)->weights());
	}

	std::string type() const
	{
		return type_;
	}

	Backend* backend() const
	{
		return backend_;
	}

	void setBackend(Backend* backend)
	{
		backend_ = backend;
	}

	void setName(const std::string& name)
	{
		name_ = name;
	}

	std::string name()
	{
		return name_;
	}

	virtual void update(Optimizer* optimizer) {} //Implement this if the layer can be trained

	virtual void build() {}

protected:

	void setType(const std::string& str)
	{
		type_ = str;
	}

	std::vector<Tensor> weights_;
	std::string type_;
	std::string name_;
	Backend* backend_;
	bool trainable_ = false;
};

class FullyConnectedLayer : public Layer
{
public:
	FullyConnectedLayer(Backend* backend = nullptr)
		:Layer(backend, true)
	{
		setType("fullyConnected");
	}

	FullyConnectedLayer(intmax_t input, intmax_t output, Backend* backend = nullptr)
		:FullyConnectedLayer(backend)
	{
		inputSize_ = input;
		outputSize_ = output;
	}

	FullyConnectedLayer(intmax_t output, Backend* backend = nullptr)
		:FullyConnectedLayer(backend)
	{
		outputSize_ = output;
	}

	virtual Shape outputShape(const Shape& s)
	{
		return Shape({Shape::None, outputSize_});
	}

	virtual void build() override
	{
		weights_.push_back(At::rand(-1,1, {inputSize_, outputSize_}, *backend()));
		weights_.push_back(At::rand(-1,1, {outputSize_}, *backend()));

		forwardAlgorithm_ = backend()->getAlgorithm<FCForwardFunction>("fullyconnectedForward");
		backwardAlgorithm_ = backend()->getAlgorithm<FCBackwardFunction>("fullyconnectedBackward");
	}

	virtual Tensor forward(const Tensor& x) override
	{
		if(x.dimension() != 2)
			throw AtError("Expecting a 2D tensor for a Fully Connected layer. But get " + std::to_string(x.dimension())
			+ "D. shape = " + to_string(x.shape()));
		return forwardAlgorithm_(x, weights_[0], weights_[1]);
	}

	virtual void backword(const Tensor& x, const Tensor& y,
		Tensor& dx, const Tensor& dy) override
	{
		dx = backwardAlgorithm_(dy, weights_[0]);

		dE = dy;
		dW = dot(x.transpose(), dy);
	}

	virtual void update(Optimizer* optimizer) override
	{
		optimizer->update(weights_[0], dW);
		optimizer->update(weights_[1], dE.sum(0));
	}

protected:
	delegate<FCForwardFunction> forwardAlgorithm_;
	delegate<FCBackwardFunction> backwardAlgorithm_;

	intmax_t inputSize_ = 0;
	intmax_t outputSize_ = 0;

	Tensor dE;
	Tensor dW;
};

class ActivationLayer : public Layer
{
public:
	ActivationLayer(Backend* backend = nullptr) : Layer(backend)
	{
		setType("activation");
	}

	virtual Shape outputShape(const Shape& s) override
	{
		return s;
	}
};

class SigmoidLayer : public ActivationLayer
{
public:
	SigmoidLayer(Backend* backend = nullptr) : ActivationLayer(backend)
	{
		setType("sigmoid");
	}

	virtual void build() override
	{
		forwardAlgorithm_ = backend()->getAlgorithm<SigmoidForward>("sigmoidForward");
		backwardAlgorithm_ = backend()->getAlgorithm<SigmoidBackward>("sigmoidBackward");
	}

	virtual Tensor forward(const Tensor& x) override
	{
		return forwardAlgorithm_(x);
	}

	virtual void backword(const Tensor& x, const Tensor& y,
		Tensor& dx, const Tensor& dy) override
	{
		dx = backwardAlgorithm_(dy, y);
	}
protected:
	delegate<SigmoidForward> forwardAlgorithm_;
	delegate<SigmoidBackward> backwardAlgorithm_;
};


class TanhLayer : public ActivationLayer
{
public:
	TanhLayer(Backend* backend) : ActivationLayer(backend)
	{
		setType("tanh");
	}

	virtual void build() override
	{
		forwardAlgorithm_ = backend()->getAlgorithm<TanhForward>("tanhForward");
		backwardAlgorithm_ = backend()->getAlgorithm<TanhBackward>("tanhBackward");
	}

	virtual Tensor forward(const Tensor& x) override
	{
		return forwardAlgorithm_(x);
	}

	virtual void backword(const Tensor& x, const Tensor& y,
		Tensor& dx, const Tensor& dy) override
	{
		dx = backwardAlgorithm_(dy, y);
	}

protected:
	delegate<TanhForward> forwardAlgorithm_;
	delegate<TanhBackward> backwardAlgorithm_;
};

class ReluLayer : public ActivationLayer
{
public:
	ReluLayer(Backend* backend = nullptr) : ActivationLayer(backend)
	{
		setType("relu");
	}

	virtual void build() override
	{
		forwardAlgorithm_ = backend()->getAlgorithm<TanhForward>("reluForward");
		backwardAlgorithm_ = backend()->getAlgorithm<TanhBackward>("reluBackward");
	}

	virtual Tensor forward(const Tensor& x) override
	{
		return forwardAlgorithm_(x);
	}

	virtual void backword(const Tensor& x, const Tensor& y,
		Tensor& dx, const Tensor& dy) override
	{
		dx = backwardAlgorithm_(dy, y);
	}

protected:
	delegate<ReluForward> forwardAlgorithm_;
	delegate<ReluBackward> backwardAlgorithm_;
};

class ReshapeLayer : public Layer
{
public:
	ReshapeLayer(Shape targetShape, Backend* backend = nullptr) : Layer(backend)
	{
		setType("reshape");
		outputShape_ = targetShape;
	}

	virtual Shape outputShape(const Shape& s) override
	{
		return outputShape_;
	}

	virtual Tensor forward(const Tensor& x) override
	{
		incomeShape_ = x.shape();
		return x.reshape(outputShape_);
	}

	virtual void backword(const Tensor& x, const Tensor& y,
		Tensor& dx, const Tensor& dy) override
	{
		dx = dy.reshape(incomeShape_);
	}

protected:
	Shape incomeShape_;
	Shape outputShape_;
};

class FalattenLayer : public Layer
{
public:
	FalattenLayer(Backend* backend = nullptr) : Layer(backend)
	{
		setType("flatten");
	}

	virtual Shape outputShape(const Shape& s) override
	{
		return {s[0], s.volume()/s[0]};
	}

	virtual Tensor forward(const Tensor& x) override
	{
		incomeShape_ = x.shape();
		return x.reshape(outputShape(x.shape()));
	}

	virtual void backword(const Tensor& x, const Tensor& y,
		Tensor& dx, const Tensor& dy) override
	{
		dx = dy.reshape(incomeShape_);
	}

protected:
	Shape incomeShape_;
};

class Conv2DLayer : public Layer
{
public:
	Conv2DLayer(intmax_t inputChannels, intmax_t outputChannels, Shape windowSize, std::array<intmax_t, 2> strides={{1,1}}, Backend* backend=nullptr)
		: Layer(backend, true)
	{
		outputChannels_ = outputChannels;
		inputChannels_ = inputChannels;
		windowSize_ = windowSize;
		strides_ = strides;

		setType("conv2D");
	}

	virtual Tensor forward(const Tensor& x) override
	{
		if(x.dimension() != 4)
			throw AtError("Conv2D expecting a 4D tensor but got " + std::to_string(x.dimension()) + "D. Shape = " + to_string(x.shape()));
		return forwardAlgorithm_(x, weights_[0], weights_[1], strides_);
	}

	virtual void backword(const Tensor& x, const Tensor& y,
		Tensor& dx, const Tensor& dy) override
	{
		dx = backwardAlgorithm_(x, weights_[0], dW_, db_, dy, strides_);
	}

	virtual Shape outputShape(const Shape& s) override
	{
		return {s[0], outputChannels_, (s[2]-windowSize_[0])/strides_[0]+1, (s[3]-windowSize_[1])/strides_[1]+1};
	}

	virtual void build() override
	{
		intmax_t inputChannels = inputChannels_;
		weights_.push_back(At::rand(-1, 1, {outputChannels_,inputChannels,windowSize_[0], windowSize_[1]}, *backend()));
		weights_.push_back(At::rand(-1, 1, {outputChannels_}, *backend()));
		forwardAlgorithm_ = backend()->getAlgorithm<Conv2DForward>("conv2DForward");
		backwardAlgorithm_ = backend()->getAlgorithm<Conv2DBackward>("conv2DBackward");
	}

	virtual void update(Optimizer* optimizer) override
	{
		optimizer->update(weights_[0], dW_);
		optimizer->update(weights_[1], db_);
	}

protected:

	intmax_t outputChannels_;
	intmax_t inputChannels_;
	Shape windowSize_;
	std::array<intmax_t, 2> strides_;

	delegate<Conv2DForward> forwardAlgorithm_;
	delegate<Conv2DBackward> backwardAlgorithm_;

	Tensor dW_;
	Tensor db_;
};

}



#endif
