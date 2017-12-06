#include <Athena/Backend.hpp>
#include <Athena/Error.hpp>

using namespace At;

static const std::string notImplementString = " not implemented in backend. Method cannot be called.";

void* Backend::createTensor(const std::vector<float>& vec, const Shape& shape)
{
	throw AtError(__func__ + notImplementString);
}

void* Backend::createTensor(const Shape& dims)
{
	throw AtError(__func__ + notImplementString);
}

void* Backend::copyTensor(const void* src)
{
	throw AtError(__func__ + notImplementString);
}

void Backend::destoryTensor(void* handle)
{
	throw AtError(__func__ + notImplementString);
}


void Backend::device(void* handle, const float* ptr)
{
	throw AtError(__func__ + notImplementString);
}

void Backend::host(void* handle, float* ptr) const
{
	throw AtError(__func__ + notImplementString);
}

void* Backend::zeros(const Shape& shape)
{
	throw AtError(__func__ + notImplementString);
}

void* Backend::ones(const Shape& shape)
{
	throw AtError(__func__ + notImplementString);
}

void* Backend::rand(float lEdge, float rEdge, const Shape& shape)
{
	throw AtError(__func__ + notImplementString);
}

void* Backend::normal(float mean, float stddev, const Shape& shape)
{
	throw AtError(__func__ + notImplementString);
}

void* Backend::add(const void* handle1,const  void* handle2)
{
	throw AtError(__func__ + notImplementString);
}

void* Backend::multiply(const void* handle1,const  void* handle2)
{
	throw AtError(__func__ + notImplementString);
}

void* Backend::scalarMul(const  void* handle, float x)
{
	throw AtError(__func__ + notImplementString);
}

void* Backend::scalarAdd(const void* handle, float val)
{
	throw AtError(__func__ + notImplementString);
}

void Backend::selfScalarAdd(void* handle, float val)
{
	throw AtError(__func__ + notImplementString);
}

void* Backend::div(const void* handle1,const  void* handle2)
{
	throw AtError(__func__ + notImplementString);
}

void* Backend::subtract(const void* handle1,const  void* handle2)
{
	throw AtError(__func__ + notImplementString);
}

void* Backend::dot(const void* handle1, const void* handle2)
{
	throw AtError(__func__ + notImplementString);
}


void* Backend::sum(const void* handle, const Shape& axis)
{
	throw AtError(__func__ + notImplementString);
}

void* Backend::pow(const void* handle, float e)
{
	throw AtError(__func__ + notImplementString);
}

void* Backend::sqrt(const void* handle)
{
	throw AtError(__func__ + notImplementString);
}

void* Backend::abs(const void* handle)
{
	throw AtError(__func__ + notImplementString);
}


Shape Backend::shape(void* handle) const
{
	throw AtError(__func__ + notImplementString);
}

void Backend::reshape(void* handle, const Shape& targetShape)
{
	throw AtError(__func__ + notImplementString);
}

void* Backend::transpose(void* handle)
{
	throw AtError(__func__ + notImplementString);
}

void* Backend::slice(void* handle, const Shape& begin, const Shape& size)
{
	throw AtError(__func__ + notImplementString);
}

void* Backend::concatenate(const void* handle1, const void* handle2, int axis)
{
	throw AtError(__func__ + notImplementString);
}

void* Backend::stack(const void* handle1, const void* handle2, int axis)
{
	throw AtError(__func__ + notImplementString);
}

size_t Backend::size(const void* handle)
{
	throw AtError(__func__ + notImplementString);
}
