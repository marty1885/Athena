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

	void setInputShape(const Shape& s)
	{
		inputShape_ = s;
	}

	void setOutputShape(const Shape& s)
	{
		outputShape_ = s;
	}

	Shape inputShape()
	{
		return inputShape_;
	}

	Shape outputShape()
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
		:FullyConnectedLayer()
	{
		setInputShape(Shape({input}));
		setOutputShape(Shape({output}));
	}

	virtual void build() override
	{
		weights_.push_back(At::rand(-1,1, {inputShape()[0], outputShape()[0]}, *backend()));
		weights_.push_back(At::rand(-1,1, outputShape(), *backend()));

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

class SigmoidLayer : public Layer
{
public:
	SigmoidLayer(Backend* backend = nullptr) : Layer(backend)
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


class TanhLayer : public Layer
{
public:
	TanhLayer(Backend* backend) : Layer(backend)
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

class ReluLayer : public Layer
{
public:
	ReluLayer(Backend* backend = nullptr) : Layer(backend)
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

class RecurrentLayer : public Layer
{
public:

protected:

	Tensor hiddenState_;
};

}



#endif
