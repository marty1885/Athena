#include <Athena/Backend/Backend.hpp>
#include <Athena/Utils/Error.hpp>

using namespace At;

static const std::string notImplementString = " not implemented in backend. Method cannot be called.";

TensorImpl* Backend::createTensor(const std::vector<float>& vec, const Shape& shape)
{
	throw AtError(__func__ + notImplementString);
}

TensorImpl* Backend::createTensor(const Shape& dims)
{
	throw AtError(__func__ + notImplementString);
}

void Backend::destoryTensor(TensorImpl* handle)
{
	throw AtError(__func__ + notImplementString);
}

TensorImpl* Backend::zeros(const Shape& shape)
{
	throw AtError(__func__ + notImplementString);
}

TensorImpl* Backend::ones(const Shape& shape)
{
	throw AtError(__func__ + notImplementString);
}

TensorImpl* Backend::rand(float lEdge, float rEdge, const Shape& shape)
{
	throw AtError(__func__ + notImplementString);
}

TensorImpl* Backend::normal(float mean, float stddev, const Shape& shape)
{
	throw AtError(__func__ + notImplementString);
}
