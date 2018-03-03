#include <Athena/Model.hpp>

#include <iostream>
#include <string>
#include <sstream>
#include <map>

using namespace At;

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
	std::cout << trimString("Layer (type)",23) << " " << trimString("Output shape", 15) << " " << trimString("Params #", 16) << '\n';
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
		std::ostringstream stream;
		stream << shape;
		std::string str =  stream.str();
		std::cout << trimString(str, 15) << " ";

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
	{
		params.set<BoxedValues>(layer->name(), layer->states());
	}
	return params;
}

void SequentialNetwork::loadStates(const BoxedValues& states)
{
	for(const auto& [key, val] : states)
	{
		if(key == "__type")
		{
			if(boxed_cast<std::string>(val) != "SequentialNetwork")
				throw AtError("Can't load.");
			continue;
		}
		
		auto layer = getLayer(key);
		if(layer == nullptr)
			throw AtError("Can't find layer \"" + key + "\". Maybe forget to initalize the model before loading the states?");
		layer->loadStates(boxed_cast<BoxedValues>(val));
	}
}