#include <Athena/NN/Model.hpp>
#include <Athena/Utils/Archive.hpp>

#include <iostream>
#include <string>
#include <sstream>
#include <map>

using namespace At;

SequentialNetwork::~SequentialNetwork()
{
	for(auto layer : layers_)
		delete layer;
}

void SequentialNetwork::summary(const Shape& inputShape) const
{
	auto repeat = [](const std::string& str, int n) {std::ostringstream os;for(int i=0;i<n;i++)os << str; return os.str();};
	auto trimString = [](const std::string& str, size_t n){
		std::string res;
		res.reserve(n);
		size_t end = std::min(str.size(), n);
		res += str.substr(0, end);
		if(end == n)
			return res;
		return res+std::string(n-end, ' ');
	};

	std::cout << repeat("─",80) << '\n';
	std::cout << trimString("Layer (type)",23) << " " << trimString("Output shape", 20) << " " << trimString("Params #", 16) << '\n';
	size_t trainableWeights = 0;
	Shape currentInputShape = inputShape;
	for(size_t i=0;i<depth();i++)
	{
		if(i == 0)
			std::cout << repeat("=",80) << '\n';
		else
			std::cout << repeat("─",80) << '\n';
		const auto& l = layers_[i];
		std::cout << trimString(l->name()+" ("+l->type()+")", 23) << " ";

		const auto& shape = l->outputShape(currentInputShape);
		currentInputShape = shape;
		std::cout << trimString(to_string(shape), 20) << " ";

		if(l->trainable() == false)
			std::cout << trimString("0", 16);
		else
		{
			size_t val = 0;
			const auto& weights = l->weights();
			for(const auto& w : weights)
				val += w.size();
			trainableWeights += val;
			std::cout << trimString(std::to_string(val), 15) << " ";
		}

		std::cout << '\n';
		if(i != depth()-1)
			std::cout << '\n';
	}

	std::cout << repeat("=",80) << '\n';

	std::cout << "Trainable weights: " << trainableWeights << '\n';
}

void SequentialNetwork::fit(Optimizer& optimizer, LossFunction& loss, const Tensor& input, const Tensor& desireOutput,
	size_t batchSize, size_t epoch)
{
	fit(optimizer, loss, input, desireOutput, batchSize, epoch, [](float){}, [](float){});
}

void SequentialNetwork::fit(Optimizer& optimizer, LossFunction& loss, const Tensor& input, const Tensor& desireOutput,
	size_t batchSize, size_t epoch, delegate<void(float)> onBatchEnumerate, delegate<void(float)> onEpochEnumerate)
{
	size_t datasetSize = input.shape()[0];

	auto inputShape = input.shape();
	auto outputShape = desireOutput.shape();
	std::vector<Tensor> layerOutputs(layers_.size()+1);

	for(size_t i=0;i<epoch;i++)
	{
		float epochLoss = 0;
		for(size_t j=0;j<datasetSize;j+=batchSize)
		{
			intmax_t sliceSize = std::min((intmax_t)batchSize, (intmax_t)(datasetSize-j));
			Tensor x = input.slice({(intmax_t)j}, {sliceSize});
			Tensor y = desireOutput.slice({(intmax_t)j} ,{sliceSize});

			inputShape[0] = sliceSize;
			outputShape[0] = sliceSize;

			x.resize(inputShape);
			y.resize(outputShape);

			//This variable should not be modified, no need to clone
			layerOutputs[0] = x;

			int index = 0;
			for(auto& layer : layers_)
			{
				const auto& currentInput = layerOutputs[index];
				Tensor out;
				layer->forward({&currentInput}, {&out});
				out.eval();
				layerOutputs[++index] = std::move(out);
			}

			if(layerOutputs.back().shape() != y.shape())
				throw AtError("Expecting model output with shape " + to_string(y.shape())
					+ " but get " + to_string(layerOutputs.back().shape()));

			Tensor dE = loss.df(layerOutputs.back(), y);
			dE.eval();

			for(int k=layers_.size()-1;k>=0;k--)
			{
				auto& layer = layers_[k];
				Tensor dx;
				layer->backword({&layerOutputs[k]},{&layerOutputs[k+1]}, {&dx}, {&dE});
				if(layer->trainable())
					layer->update(&optimizer);
				dx.eval();

				dE = std::move(dx);
			}
			float batchLoss = loss.f(layerOutputs.back(), y).host<float>()[0];
			onBatchEnumerate(batchLoss);
			epochLoss += batchLoss*((float)(sliceSize)/datasetSize);
		}
		onEpochEnumerate(epochLoss);

	}
}

Tensor SequentialNetwork::predict(const Tensor& input)
{
	Tensor t = input;
	for(auto& layer : layers_)
	{
		Tensor y;
		layer->forward({&t}, {&y});
		t = std::move(y);
	}
	t.eval();

	return t;
}

float SequentialNetwork::test(const Tensor& input, const Tensor& desireOutput, LossFunction& loss)
{
	Tensor t = predict(input);
	return loss.f(t, desireOutput).host<float>()[0];
}

void SequentialNetwork::compile()
{
	std::map<std::string, int> layerApperence;

	for(auto& layer : layers_)
	{
		if(layer->name() == "")
		{
			std::string typeStr = layer->type();
			//std::map initializes variable by default even when accessing it
			layer->setName(typeStr + "_" + std::to_string(++layerApperence[typeStr]));
		}

		if(layer->backend() == nullptr)
			layer->setBackend(backend_);

		layer->build();
	}
}

BoxedValues SequentialNetwork::states() const
{
	BoxedValues params;
	params.set<std::string>("__type", "SequentialNetwork");
	for(const auto& layer : layers_)
		params.set<BoxedValues>(layer->name(), layer->states());
	return params;
}

void SequentialNetwork::loadStates(const BoxedValues& states)
{
	for(const auto& [key, val] : states)
	{
		if(key == "__type")
		{
			if(boxed_cast<std::string>(val) != "SequentialNetwork")
				throw AtError("Can't load the states. The given states are not for SequentialNetwork");
			continue;
		}
		
		auto l = layer(key);
		if(l == nullptr)
			throw AtError("Can't find layer \"" + key +
				"\". Maybe forget to initalize the model before loading the states or the model changed?");
		l->loadStates(boxed_cast<BoxedValues>(val));
	}
}


void SequentialNetwork::save(const std::string path) const
{
	Archiver::save(states(), path);
}

void SequentialNetwork::load(const std::string path)
{
	loadStates(Archiver::load(path));
}
