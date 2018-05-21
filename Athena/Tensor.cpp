#include <Athena/Tensor.hpp>

using namespace At;

Backend* Tensor::defaultBackend_ = nullptr;

void Tensor::loadStates(const BoxedValues& states)
{
	const auto& s = states.get<Shape>("shape");
	auto boxedPtr = states.ptr("values");
	Backend* bk = backend()==nullptr ? defaultBackend() : backend();

	if(auto ptr = boxed_cast<std::vector<float>>(boxedPtr); boxedPtr != nullptr)
	{
		const auto& values = states.get<std::vector<float>>("values");
		*this = Tensor(values, s, *bk);
	}
	else if(auto ptr = boxed_cast<std::vector<double>>(boxedPtr); boxedPtr != nullptr)
	{
		const auto& values = states.get<std::vector<double>>("values");
		*this = Tensor(values, s, *bk);
	}
	else if(auto ptr = boxed_cast<std::vector<int32_t>>(boxedPtr); boxedPtr != nullptr)
	{
		const auto& values = states.get<std::vector<int32_t>>("values");
		*this = Tensor(values, s, *bk);
	}
	else if(auto ptr = boxed_cast<std::vector<int32_t>>(boxedPtr); boxedPtr != nullptr)
	{
		const auto& values = states.get<std::vector<int32_t>>("values");
		*this = Tensor(values, s, *bk);
	}
}

