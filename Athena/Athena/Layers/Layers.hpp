#ifndef LAYERS_HPP
#define LAYERS_HPP

#include <Athena/Optimizer.hpp>
#include <Athena/Backend.hpp>
#include <Athena/Tensor.hpp>
#include <Athena/Utils/Shape.hpp>

#include <memory>

#include <cmath>

namespace At
{

class WeightInitalizer
{
public:
	virtual Tensor create(const Shape& shape , intmax_t fanIn, intmax_t fanOut, Backend* backend) const = 0;
};

//TODO: Need to check if these are right. But they seems to work fine
class Xavier : public WeightInitalizer
{
public:
	virtual Tensor create(const Shape& shape , intmax_t fanIn, intmax_t fanOut, Backend* backend) const override
	{
		return normal(0, 1, shape, *backend) / std::sqrt((float)fanIn+(float)fanOut);
	}
};

class He : public WeightInitalizer
{
public:
	virtual Tensor create(const Shape& shape , intmax_t fanIn, intmax_t fanOut, Backend* backend) const override
	{
		return normal(0, 1, shape, *backend) / std::sqrt(2.f/(float)fanIn);
	}
};

class Layer
{
public:
	virtual ~Layer()
	{
	}

	Layer():
		weightInitalizer_(std::make_shared<Xavier>())
	{
	}

	Layer(Backend* backend=nullptr, bool trainable=false):
		weightInitalizer_(std::make_shared<Xavier>()), backend_(backend), trainable_(trainable)
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

	virtual std::vector<Tensor> weights() const
	{
		return std::vector<Tensor>();
	}

	bool isInitialized() const
	{
		return backend_ != nullptr;
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

	std::string name() const
	{
		return name_;
	}

	virtual void update(Optimizer* optimizer) {} //Implement this if the layer can be trained

	virtual void build() {}

	virtual BoxedValues states() const
	{
		throw AtError("Error: Trying to save a layer that does not suppot saving");
	}

	virtual void loadStates(const BoxedValues& states) {}


protected:

	void setType(const std::string& str)
	{
		type_ = str;
	}

	std::string type_;
	std::string name_;
	std::shared_ptr<WeightInitalizer> weightInitalizer_;
	Backend* backend_ = nullptr;
	bool trainable_ = false;
};

class FullyConnectedLayer : public Layer
{
public:
	FullyConnectedLayer(Backend* backend = nullptr);
	FullyConnectedLayer(intmax_t input, intmax_t output, Backend* backend = nullptr);
	FullyConnectedLayer(intmax_t output, Backend* backend = nullptr);
	virtual Shape outputShape(const Shape& s);
	virtual void build() override;
	virtual Tensor forward(const Tensor& x) override;
	virtual void backword(const Tensor& x, const Tensor& y,
		Tensor& dx, const Tensor& dy) override;
	virtual void update(Optimizer* optimizer) override;
	std::vector<Tensor> weights() const override;
	virtual BoxedValues states() const override;
	virtual void loadStates(const BoxedValues& states) override;

protected:
	delegate<FCForwardFunction> forwardAlgorithm_;
	delegate<FCBackwardFunction> backwardAlgorithm_;

	intmax_t inputSize_ = 0;
	intmax_t outputSize_ = 0;

	Tensor w_;
	Tensor b_;

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
	SigmoidLayer(Backend* backend = nullptr);
	virtual void build() override;
	virtual Tensor forward(const Tensor& x) override;
	virtual void backword(const Tensor& x, const Tensor& y,
		Tensor& dx, const Tensor& dy) override;
	virtual BoxedValues states() const override;
protected:
	delegate<SigmoidForward> forwardAlgorithm_;
	delegate<SigmoidBackward> backwardAlgorithm_;
};


class TanhLayer : public ActivationLayer
{
public:
	TanhLayer(Backend* backend = nullptr);
	virtual void build() override;
	virtual Tensor forward(const Tensor& x) override;
	virtual void backword(const Tensor& x, const Tensor& y,
		Tensor& dx, const Tensor& dy) override;
protected:
	delegate<TanhForward> forwardAlgorithm_;
	delegate<TanhBackward> backwardAlgorithm_;
};

class ReluLayer : public ActivationLayer
{
public:
	ReluLayer(Backend* backend = nullptr);
	virtual void build() override;
	virtual Tensor forward(const Tensor& x) override;
	virtual void backword(const Tensor& x, const Tensor& y,
		Tensor& dx, const Tensor& dy) override;

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

class FlattenLayer : public Layer
{
public:
	FlattenLayer(Backend* backend = nullptr) : Layer(backend)
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
	Conv2DLayer(intmax_t inputChannels, intmax_t outputChannels, Shape windowSize, std::array<intmax_t, 2> strides={{1,1}}, Backend* backend=nullptr);
	virtual Tensor forward(const Tensor& x) override;
	virtual void backword(const Tensor& x, const Tensor& y,
		Tensor& dx, const Tensor& dy) override;
	virtual Shape outputShape(const Shape& s) override;
	virtual void build() override;
	virtual void update(Optimizer* optimizer) override;
	std::vector<Tensor> weights() const override;

protected:

	intmax_t outputChannels_;
	intmax_t inputChannels_;
	Shape windowSize_;
	std::array<intmax_t, 2> strides_;

	delegate<Conv2DForward> forwardAlgorithm_;
	delegate<Conv2DBackward> backwardAlgorithm_;

	Tensor kernel_;
	Tensor bias_;

	Tensor dW_;
	Tensor db_;
};

class LeakyReluLayer : public ActivationLayer
{
public:
	LeakyReluLayer(float alpha = 0.1f, Backend* backend = nullptr);
	virtual void build() override;
	virtual Tensor forward(const Tensor& x) override;
	virtual void backword(const Tensor& x, const Tensor& y,
		Tensor& dx, const Tensor& dy) override;

protected:
	delegate<LeakyReluForward> forwardAlgorithm_;
	delegate<LeakyReluBackward> backwardAlgorithm_;
	float alpha_;
};

}



#endif
