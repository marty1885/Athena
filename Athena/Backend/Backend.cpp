#include <Athena/Backend/Backend.hpp>
#include <Athena/Utils/Error.hpp>

using namespace At;

static const std::string notImplementString = " not implemented in backend. Method cannot be called.";

TensorImpl* Backend::createTensor(const std::vector<float>& vec, const Shape& shape)
{
	throw AtError(__func__ + notImplementString);
}

TensorImpl* Backend::createTensor(const std::vector<double>& vec, const Shape& shape)
{
	throw AtError(__func__ + notImplementString);
}

TensorImpl* Backend::createTensor(const std::vector<int32_t>& vec, const Shape& shape)
{
	throw AtError(__func__ + notImplementString);
}

TensorImpl* Backend::createTensor(const std::vector<int16_t>& vec, const Shape& shape)
{
	throw AtError(__func__ + notImplementString);
}

TensorImpl* Backend::createTensor(const std::vector<bool>& vec, const Shape& shape)
{
	throw AtError(__func__ + notImplementString);
}

TensorImpl* Backend::createTensor(const Shape& dims)
{
	throw AtError(__func__ + notImplementString);
}

TensorImpl* Backend::clone(const TensorImpl* handle)
{
	throw AtError(__func__ + notImplementString);
}

void Backend::destoryTensor(TensorImpl* handle)
{
	throw AtError(__func__ + notImplementString);
}

TensorImpl* Backend::zeros(const Shape& shape, DType detype)
{
	throw AtError(__func__ + notImplementString);
}

TensorImpl* Backend::ones(const Shape& shape, DType detype)
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

void Backend::eval(TensorImpl* impl)
{
	//eval is nop by default
}

Shape Backend::shape(const TensorImpl* impl) const
{
	throw AtError(__func__ + notImplementString);
}
intmax_t Backend::size(const TensorImpl* impl) const
{
	throw AtError(__func__ + notImplementString);
}

DType Backend::dtype(const TensorImpl* impl) const
{
	throw AtError(__func__ + notImplementString);
}

void Backend::selfReciprocate(TensorImpl* impl)
{
	throw AtError(__func__ + notImplementString);
}

void Backend::selfAdd(TensorImpl* impl, float val)
{
	throw AtError(__func__ + notImplementString);
}

void Backend::selfMul(TensorImpl* impl, float val)
{
	throw AtError(__func__ + notImplementString);
}

void Backend::selfAdd(TensorImpl* impl, const TensorImpl* other)
{
	throw AtError(__func__ + notImplementString);
}

void Backend::selfMul(TensorImpl* impl, const TensorImpl* other)
{
	throw AtError(__func__ + notImplementString);
}

void Backend::selfSub(TensorImpl* impl, const TensorImpl* other)
{
	throw AtError(__func__ + notImplementString);
}

void Backend::selfDiv(TensorImpl* impl, const TensorImpl* other)
{
	throw AtError(__func__ + notImplementString);
}

TensorImpl* Backend::sqrt(const TensorImpl* impl)
{
	throw AtError(__func__ + notImplementString);
}

TensorImpl* Backend::abs(const TensorImpl* impl)
{
	throw AtError(__func__ + notImplementString);
}

TensorImpl* Backend::exp(const TensorImpl* impl)
{
	throw AtError(__func__ + notImplementString);
}

TensorImpl* Backend::log(const TensorImpl* impl)
{
	throw AtError(__func__ + notImplementString);
}

TensorImpl* Backend::pow(const TensorImpl* impl, float val)
{
	throw AtError(__func__ + notImplementString);
}

TensorImpl* Backend::dot(const TensorImpl* impl, const TensorImpl* other)
{
	throw AtError(__func__ + notImplementString);
}

void Backend::modDims(TensorImpl* impl, const Shape& wantedShape)
{
	throw AtError(__func__ + notImplementString);
}

TensorImpl* Backend::reshape(const TensorImpl* impl, const Shape& wantedShape)
{
	throw AtError(__func__ + notImplementString);
}

TensorImpl* Backend::transpose(const TensorImpl* impl)
{
	throw AtError(__func__ + notImplementString);
}

TensorImpl* Backend::stack(const TensorImpl* impl, const TensorImpl* other, int axis)
{
	throw AtError(__func__ + notImplementString);
}

TensorImpl* Backend::concatenate(const std::vector<TensorImpl const*>& arrs, int axis)
{
	throw AtError(__func__ + notImplementString);
}

TensorImpl* Backend::chunk(const TensorImpl* impl, const Shape& begin, const Shape& size)
{
	throw AtError(__func__ + notImplementString);
}

TensorImpl* Backend::sum(const TensorImpl* impl, intmax_t axis)
{
	throw AtError(__func__ + notImplementString);
}

TensorImpl* Backend::sum(const TensorImpl* impl, const std::vector<intmax_t>& axis)
{
	throw AtError(__func__ + notImplementString);
}

void Backend::host(const TensorImpl* impl, float* ptr) const
{
	throw AtError(__func__ + notImplementString);
}

void Backend::host(const TensorImpl* impl, double* ptr) const
{
	throw AtError(__func__ + notImplementString);
}

void Backend::host(const TensorImpl* impl, int32_t* ptr) const
{
	throw AtError(__func__ + notImplementString);
}

void Backend::host(const TensorImpl* impl, int16_t* ptr) const
{
	throw AtError(__func__ + notImplementString);
}

void Backend::host(const TensorImpl* impl, bool* ptr) const
{
	throw AtError(__func__ + notImplementString);
}

void Backend::device(TensorImpl* impl, const float* ptr)
{
	throw AtError(__func__ + notImplementString);
}

void Backend::device(TensorImpl* impl, const double* ptr)
{
	throw AtError(__func__ + notImplementString);
}

void Backend::device(TensorImpl* impl, const int32_t* ptr)
{
	throw AtError(__func__ + notImplementString);
}

void Backend::device(TensorImpl* impl, const int16_t* ptr)
{
	throw AtError(__func__ + notImplementString);
}

void Backend::device(TensorImpl* impl, const bool* ptr)
{
	throw AtError(__func__ + notImplementString);
}

void* Backend::hostPtr(TensorImpl* impl)
{
	return nullptr;
}

const void* Backend::hostPtr(const TensorImpl* impl)
{
	return nullptr;
}

TensorImpl* Backend::greaterThan(const TensorImpl* impl,float val)
{
	throw AtError(__func__ + notImplementString);
}

TensorImpl* Backend::lesserThan(const TensorImpl* impl,float val)
{
	throw AtError(__func__ + notImplementString);
}

TensorImpl* Backend::greaterOrEqual(const TensorImpl* impl,float val)
{
	throw AtError(__func__ + notImplementString);
}

TensorImpl* Backend::lesserOrEqual(const TensorImpl* impl,float val)
{
	throw AtError(__func__ + notImplementString);
}

TensorImpl* Backend::equalTo(const TensorImpl* impl,float val)
{
	throw AtError(__func__ + notImplementString);
}
