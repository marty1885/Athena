#ifndef LAYERS_HPP
#define LAYERS_HPP

#include <Athena/Backend.hpp>
#include <Athena/Tensor.hpp>
#include <Athena/Shape.hpp>

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

	virtual void setInputShape(const Shape& s)
	{
		inputShape_ = s;
	}

	virtual void setOutputShape(const Shape& s)
	{
		outputShape_ = s;
	}

	virtual Shape inputShape()
	{
		return inputShape_;
	}

	virtual Shape outputShape()
	{
		return outputShape_;
	}

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
	Shape inputShape_;
	Shape outputShape_;
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
		setInputShape(Shape({Shape::None, input}));
		setOutputShape(Shape({Shape::None, output}));
	}

	FullyConnectedLayer(intmax_t output, Backend* backend = nullptr)
		:FullyConnectedLayer(backend)
	{
		setOutputShape(Shape({Shape::None, output}));
	}

	virtual void build() override
	{
		weights_.push_back(At::rand(-1,1, {inputShape()[1], outputShape()[1]}, *backend()));
		weights_.push_back(At::rand(-1,1, {outputShape()[1]}, *backend()));

		forwardAlgorithm_ = backend()->getAlgorithm<FCForwardFunction>("fullyconnectedForward");
		backwardAlgorithm_ = backend()->getAlgorithm<FCBackwardFunction>("fullyconnectedBackward");
	}

	virtual Tensor forward(const Tensor& x) override
	{
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

	Tensor dE;
	Tensor dW;
};

//HACK: A way to get ActivationLayer's input shape == output shape
class ActivationLayer : public Layer
{
public:
	ActivationLayer(Backend* backend = nullptr) : Layer(backend)
	{
		setType("activation");
	}

	virtual void setInputShape(const Shape& s) override
	{
		inputShape_ = s;
		outputShape_ = s;
	}

	virtual void setOutputShape(const Shape& s) override
	{
		inputShape_ = s;
		outputShape_ = s;
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
		setOutputShape(targetShape);
	}

	virtual Tensor forward(const Tensor& x) override
	{
		incomeShape_ = x.shape();
		Shape target = outputShape();
		target[0] = incomeShape_[0];
		return x.reshape(target);
	}

	virtual void backword(const Tensor& x, const Tensor& y,
		Tensor& dx, const Tensor& dy) override
	{
		dx = dy.reshape(incomeShape_);
	}

protected:
	Shape incomeShape_;
};

class FalattenLayer : public Layer
{
public:
	FalattenLayer(Backend* backend = nullptr) : Layer(backend)
	{
		setType("flatten");
	}

	virtual void setInputShape(const Shape& s) override
	{
		inputShape_ = s;
		outputShape_ = {s[0], s.volume()/s[0]};
	}

	virtual Tensor forward(const Tensor& x) override
	{
		incomeShape_ = x.shape();
		Shape target = outputShape();
		target[0] = incomeShape_[0];
		return x.reshape(target);
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
	Conv2DLayer(intmax_t outputChannels, Shape windowSize, std::array<intmax_t, 2> strides={{1,1}}, Backend* backend=nullptr)
		: Layer(backend)
	{
		outputChannels_ = outputChannels;
		windowSize_ = windowSize;
		strides_ = strides;

		setType("conv2D");
	}

	virtual Tensor forward(const Tensor& x) override
	{
		return forwardAlgorithm_(x, weights_[0], weights_[1], strides_);
	}

	virtual void backword(const Tensor& x, const Tensor& y,
		Tensor& dx, const Tensor& dy) override
	{
		dx = backwardAlgorithm_(x, weights_[0], dW_, db_, dy, strides_);
	}

	virtual void setInputShape(const Shape& s) override
	{
		inputShape_ = s;
		outputShape_ = {s[0], outputChannels_, (s[2]-windowSize_[0])/strides_[0]+1, (s[3]-windowSize_[1])/strides_[1]+1};
	}

	virtual void build() override
	{
		intmax_t inputChannels = inputShape()[1];
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
	Shape windowSize_;
	std::array<intmax_t, 2> strides_;

	delegate<Conv2DForward> forwardAlgorithm_;
	delegate<Conv2DBackward> backwardAlgorithm_;

	Tensor dW_;
	Tensor db_;
};

class InputLayer : public Layer
{
public:
	InputLayer(Shape s, Backend* backend=nullptr) : Layer(backend)
	{
		setInputShape(s);
		setOutputShape(s);

		setType("input");
	}

	virtual Tensor forward(const Tensor& x) override
	{
		return x;
	}
};

class RecurrentLayer : public Layer
{
public:

protected:

	Tensor hiddenState_;
};

}



#endif
