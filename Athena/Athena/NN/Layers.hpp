#ifndef LAYERS_HPP
#define LAYERS_HPP

#include <Athena/NN/Optimizer.hpp>
#include <Athena/Backend/Backend.hpp>
#include <Athena/Tensor.hpp>
#include <Athena/Utils/Shape.hpp>

#include <memory>

#include <cmath>

namespace At
{

class WeightInitalizer
{
public:
	virtual ~WeightInitalizer() = default;
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
		return normal(0, 1, shape, *backend) * std::sqrt(2.f/(float)fanIn);
	}
};

class Layer
{
public:
	virtual ~Layer() = default;

	Layer():
		weightInitalizer_(std::make_shared<Xavier>())
	{
	}

	Layer(Backend* backend=nullptr, bool trainable=false):
		weightInitalizer_(std::make_shared<Xavier>()), backend_(backend), trainable_(trainable)
	{
	}

	//Temp function, use to gap the code difference between new API
	virtual void forward(const SmallVector<const Tensor*> x, SmallVector<Tensor*> y)
	{
		if(x.size() != 1)
			throw AtError("Forwording with multiple variables are not implemnted.");
		*y[0] = forward(*x[0]);
	}

	virtual Tensor forward(const Tensor& input)
	{
		throw AtError("Forwarding with one variable not implemented");
	}

	virtual void backword(const Tensor& x, const Tensor& y,
		Tensor& dx, const Tensor& dy)
	{
		throw AtError("Backwording with one variable not implemented");
	}

	virtual void backword(const SmallVector<const Tensor*> x, const SmallVector<const Tensor*> y
		,SmallVector<Tensor*> dx ,const SmallVector<const Tensor*> dy)
	{
		AtAssert(x.size() == 1 && y.size() == 1 && dy.size() == 1,
			"Backwording with multiple variables are not implemnted.");

		backword(*x[0], *y[0], *dx[0], *dy[0]);
	}
	
	//Caclulate the output shape given a input shape
	virtual Shape outputShape(const Shape& s) = 0;

	inline bool trainable() const
	{
		return trainable_;
	}

	inline void setTrainable(bool val)
	{
		trainable_ = val;
	}

	virtual std::vector<Tensor> weights() const
	{
		return std::vector<Tensor>();
	}

	inline bool isInitialized() const
	{
		return backend_ != nullptr;
	}

	inline std::string type() const
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

	inline std::string name() const
	{
		return name_;
	}

	template <typename T>
	void setWeightInitalizer()
	{
		weightInitalizer_ = std::make_shared<T>();
	}

	virtual void update(Optimizer* optimizer) {} //Implement this if the layer can be trained

	virtual void build() {}

	virtual BoxedValues states() const
	{
		throw AtError("Error: Trying to save a layer that does not suppot saving");
	}

	virtual void loadStates(const BoxedValues& states) {}


protected:

	inline void setType(const std::string& str)
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
	virtual Shape outputShape(const Shape& s) override;
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

	virtual BoxedValues states() const override
	{
		BoxedValues params;
		params.set<std::string>("__type", type());
		return params;
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

	virtual BoxedValues states() const override
	{
		BoxedValues params;
		params.set<std::string>("__type", type());
		params.set<Shape>("incomeShape", incomeShape_);
		params.set<Shape>("outputShape", outputShape_);
		return params;
	}

	void loadStates(const BoxedValues& states) override
	{
		incomeShape_ = states.get<Shape>("incomeShape");
		outputShape_ = states.get<Shape>("outputShape");
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

	virtual BoxedValues states() const override
	{
		BoxedValues params;
		params.set<std::string>("__type", type());
		return params;
	}

protected:
	Shape incomeShape_;
};

class Conv2DLayer : public Layer
{
public:
	Conv2DLayer(intmax_t inputChannels, intmax_t outputChannels, Shape windowSize, Shape strides={{1,1}}, Backend* backend=nullptr);
	virtual Tensor forward(const Tensor& x) override;
	virtual void backword(const Tensor& x, const Tensor& y,
		Tensor& dx, const Tensor& dy) override;
	virtual Shape outputShape(const Shape& s) override;
	virtual void build() override;
	virtual void update(Optimizer* optimizer) override;
	std::vector<Tensor> weights() const override;
	inline Shape stride() const {return strides_;};

	void setStrides(const Shape& stride)
	{
		strides_ = stride;
	}

	virtual BoxedValues states() const override
	{
		BoxedValues params;
		params.set<std::string>("__type", type());
		params.set<BoxedValues>("kernel", kernel_.states());
		params.set<BoxedValues>("bias", bias_.states());
		params.set<Shape>("strides", strides_);
		return params;
	}

	virtual void loadStates(const BoxedValues& states) override
	{
		kernel_.loadStates(states.get<BoxedValues>("kernel"));
		bias_.loadStates(states.get<BoxedValues>("bias"));
		strides_ = states.get<Shape>("strides");
		outputChannels_ = kernel_.shape()[0];
		inputChannels_ = kernel_.shape()[1];
		windowSize_ = {kernel_.shape()[2], kernel_.shape()[3]};
	}

protected:

	intmax_t outputChannels_;
	intmax_t inputChannels_;
	Shape windowSize_;
	Shape strides_;

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
	virtual BoxedValues states() const override
	{
		BoxedValues params;
		params.set<std::string>("__type", type());
		params.set<float>("alpha", alpha_);
		return params;
	}

	virtual void loadStates(const BoxedValues& states) override
	{
		alpha_ = states.get<float>("alpha");
	}

protected:
	delegate<LeakyReluForward> forwardAlgorithm_;
	delegate<LeakyReluBackward> backwardAlgorithm_;
	float alpha_;
};

class DropoutLayer : public Layer
{
public:
	DropoutLayer(float rate, Backend* backend = nullptr);
	virtual void build() override;
	virtual Tensor forward(const Tensor& x) override;
	virtual void backword(const Tensor& x, const Tensor& y,
		Tensor& dx, const Tensor& dy) override;
	virtual Shape outputShape(const Shape& s) override
	{
		return s;
	}
	virtual BoxedValues states() const override
	{
		BoxedValues params;
		params.set<std::string>("__type", type());
		params.set<float>("rate", rate_);
		return params;
	}

	virtual void loadStates(const BoxedValues& states) override
	{
		rate_ = states.get<float>("rate");
	}

	void setBypass(bool bypass)
	{
		bypass_ = bypass;
	}

	bool bypass() const
	{
		return bypass_;
	}
protected:
	float rate_;
	Tensor filter_;
	bool bypass_ = false;
};

//Short names
using FullyConnected = FullyConnectedLayer;
using DenseLayer = FullyConnectedLayer;
using Dense = FullyConnectedLayer;
using Dense = FullyConnectedLayer;
using Conv2D = Conv2DLayer;
using Sigmoid = SigmoidLayer;
using Tanh = TanhLayer;
using Relu = ReluLayer;
using LeakyRelu = LeakyReluLayer;
using Reshape = ReshapeLayer;
using Flatten = FlattenLayer;
using Dropout = DropoutLayer;

}



#endif
