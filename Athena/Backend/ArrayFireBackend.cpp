#include <Athena/Backend/ArrayFireBackend.hpp>
#include <Athena/Utils/Shape.hpp>
#include <Athena/Backend/TensorImpl.hpp>
#include <Athena/Tensor.hpp>

#include <array>
#include <random>

using namespace At;

inline af::dim4 shapeToDim4(const Shape& s)
{
	if(s.size() > 4)
		throw AtError("ArrayFire backend only supports upto 4D Tensor, got " + std::to_string(s.size()) + "D.");
	Shape inv = s;
	std::reverse(inv.begin(), inv.end());
	af::dim4 dims;
	for(size_t i=0;i<4;i++)
		dims[i] = (i<s.size() ? inv[i] : 1);
	return dims;
}

inline Shape shapeFromDim4(const af::dim4& dim, intmax_t num)
{
	assert(num >= 1 && num <= 4);
	Shape s;
	s.resize(num);
	for(int i=0;i<num;i++)
		s[i] = dim[num-i-1];
	return s;
}

inline std::vector<int> range(int start, int end)
{
	std::vector<int> vec(end-start+1);
	for(int i=0;i<(int)vec.size();i++)
		vec[i] = start+i;
	return vec;
}

inline DType afTypeToDType (af::dtype dtype)
{
	if(dtype == f32)
		return DType::float32;
	else if(dtype == f64)
		return DType::float64;
	else if(dtype == s32)
		return DType::int32;
	else if(dtype == s16)
		return DType::int16;
	else if(dtype == b8)
		return DType::bool8;
	else
		return DType::unknown;
}

inline af_dtype dtypeToAFType(DType dtype)
{
	if(dtype == DType::float32)
		return f32;
	else if(dtype == DType::float64)
		return f64;
	else if(dtype == DType::int32)
		return s32;
	else if(dtype == DType::int16)
		return s16;
	else if(dtype == DType::bool8)
		return b8;
	throw AtError(std::string("Cannot convert ") + to_string(dtype) + " to ArrayFire type");
}

inline size_t typeToSize(af::dtype dtype)
{
	if(dtype == f32)
		return sizeof(float);
	else if(dtype == f64)
		return sizeof(double);
	else if(dtype == s32)
		return sizeof(int32_t);
	else if(dtype == s64)
		return sizeof(int64_t);
	else if(dtype == s16)
		return sizeof(int16_t);
	else if(dtype == b8)
		return sizeof(bool);
	return 0;
}

template <typename T>
inline void copyToHost(const af::array& arr, T* ptr)
{
	if(afTypeToDType(arr.type()) != typeToDType<T>())
		throw AtError("Cannot copy data from device to host, type does not match");
	arr.host(ptr);
}

template <typename T>
inline void writeToDevice(af::array& arr, const T* ptr)
{
	if(afTypeToDType(arr.type()) != typeToDType<T>())
		throw AtError("Cannot copy write from host to device, type does not match");
	arr.write(ptr, arr.bytes(), afHost);
}

//ArrayFire trats bool8 as char. Special case
template <>
inline void writeToDevice(af::array& arr, const bool* ptr)
{
	if(afTypeToDType(arr.type()) != DType::bool8)
		throw AtError("Cannot copy write from host to device, type does not match");
	arr.write((char*)ptr, arr.bytes(), afHost);
}


//TODO: Really need a better backend. AF only supports upto 4D arrays
//But 5D is needed for recurrent layer + conv layers
//And, AF works in fortran order, not C order
class AFTensorImpl : public TensorImpl
{
public:
	AFTensorImpl(ArrayFireBackend* backend) : TensorImpl(backend)
	{
	}

	//AFTensorImpl owns the array, pass a copy if don't want the array be modified
	//ArrayFire's array are referenced by default
	AFTensorImpl(af::array arr, const Shape& s, ArrayFireBackend* backend) : TensorImpl(backend)
	{
		if(s.size() > 4)
			throw AtError("ArrayFire backend only supports upto 4D Tensor, got " + std::to_string(s.size()) + "D.");
		arrShape_ = s;
		arr_ = std::move(arr);
	}

	const af::array& get() const
	{
		return arr_;
	}

	af::array& get()
	{
		return arr_;
	}

	af::array arr_;
	//A seprate variable to track the shape of the array
	//due to AF alwas have a 4D array and is not in C order
	Shape arrShape_;
};

inline af::array& get(Tensor& t)
{
	return ((AFTensorImpl*)t.pimpl())->get();
}

inline const af::array& get(const Tensor& t)
{
	return ((const AFTensorImpl*)t.pimpl())->get();
}

ArrayFireBackend::ArrayFireBackend(AFBackend afBackend)
{
	setType("arrayfire");
	setAFBackend(afBackend);

	//Enable brodcasting
	af::gforSet(true);

	addAlgorithm<FCForwardFunction>("fullyconnectedForward",
	[this](const Tensor& in, const Tensor& weight, const Tensor& bias)->Tensor
		{
			const auto& i = get(in);
			const auto& w = get(weight);
			const auto& b = get(bias);
			af::array res = af::transpose(af::matmulTT(i, w)) + b;
			assert(res.numdims() == 1 || res.numdims() == 2);
			return createTensor(std::move(res), shapeFromDim4(res.dims(), 2));
		});

	addAlgorithm<FCBackwardFunction>("fullyconnectedBackward",
		[this](const Tensor& dx, const Tensor& weight)->Tensor
		{
			const auto& i = get(dx);
			const auto& w = get(weight);
			//Not the most effective way
			af::array res = af::transpose(af::matmulTN(i, w));
			return createTensor(
				std::move(res), shapeFromDim4(res.dims(), 2));
		});

	addAlgorithm<SigmoidForward>("sigmoidForward",
		[this](const Tensor& x)->Tensor
		{
			const auto& t = get(x);
			return createTensor(1/(1+af::exp(-t)), x.shape());
		});

	addAlgorithm<SigmoidBackward>("sigmoidBackward",
		[this](const Tensor& a, const Tensor& b)->Tensor
		{
			const auto& dy = get(a);
			const auto& y = get(b);
			return createTensor(dy*(y*(1-y)), b.shape());
		});

	addAlgorithm<TanhForward>("tanhForward",
		[this](const Tensor& x)->Tensor
		{
			const auto& t = get(x);
			auto res = af::tanh(t);
			return createTensor(std::move(res), x.shape());
		});

	addAlgorithm<TanhBackward>("tanhBackward",
		[this](const Tensor& a, const Tensor& b)->Tensor
		{
			const auto& dy = get(a);
			const auto& y = get(b);
			auto res = dy * (1 - af::pow(af::tanh(y), 2));
			return createTensor(std::move(res), b.shape());
		});
	
	addAlgorithm<ReluForward>("reluForward",
		[this](const Tensor& x)->Tensor
		{
			const auto& t = get(x);
			af::array res = (t>0)*t;
			return createTensor(std::move(res), x.shape());
		});
	
	addAlgorithm<ReluBackward>("reluBackward",
		[this](const Tensor& a, const Tensor& b)->Tensor
		{
			const auto& dy = get(a);
			const auto& y = get(b);
			af::array res = dy*(y>0);
			return createTensor(std::move(res), a.shape());
		});
	
	addAlgorithm<LeakyReluForward>("leakyReluForward",
		[this](const Tensor& x, float alpha)->Tensor
		{
			const auto& t = get(x);
			af::array res = t*(t>0) + alpha*t*(t<0);
			return createTensor(std::move(res), x.shape());
		});

	addAlgorithm<LeakyReluBackward>("leakyReluBackward",
		[this](const Tensor& a, const Tensor& b, float alpha)->Tensor
		{
			const auto& dy = get(a);
			const auto& y = get(b);
			af::array res = dy*(y>0) + alpha*dy*(y<0);
			return createTensor(std::move(res), a.shape());
		});
	//ArrayFire does not have deconvolution for now. The backword function can't be implement.
	//Maybe implment it via cuDNN/MIOpen?
	addAlgorithm<Conv2DForward>("conv2DForward",
		[this](const Tensor& x, const Tensor& kernel, const Tensor& bias, const Shape& strides)->Tensor
	{
		intmax_t batchSize = x.shape()[0];
		intmax_t inputHeight = x.shape()[2];
		intmax_t inputWidth = x.shape()[3];

		intmax_t outputChannels = kernel.shape()[0];
		intmax_t filterHeight = kernel.shape()[2];
		intmax_t filterWidth = kernel.shape()[3];
		intmax_t outputHeight = (inputHeight-filterHeight)/strides[0]+1;
		intmax_t outputWidth = (inputWidth-filterWidth)/strides[1]+1;

		assert(x.shape()[1] == kernel.shape()[1]);
		Shape outputShape({batchSize, outputChannels, outputHeight, outputWidth});
		af::array arr = af::convolve(get(x), get(kernel))+get(bias);
		intmax_t hw = filterWidth/2;
		intmax_t hh = filterHeight/2;
		arr = arr(af::seq(hw, af::end-hw), af::seq(hh, af::end-hh), af::span, af::span);
		return new AFTensorImpl(std::move(arr), outputShape, this);
	}
	,[](const BoxedValues& config)->bool
	{
		Shape kernelShape = config.get<Shape>("kernelShape");
		Shape stride = config.get<Shape>("stride");
		return (kernelShape[2] <= 16 && kernelShape[3] <= 16 &&
			stride[0] == 1 && stride[1] == 1);
	});
}

TensorImpl* ArrayFireBackend::createTensor(const Shape& dims)
{
	return this->zeros(dims);
}


TensorImpl* ArrayFireBackend::createTensor(const af::array& arr, const Shape& s)
{
	return new AFTensorImpl(arr, s, this);
}

void ArrayFireBackend::destoryTensor(TensorImpl* impl)
{
	delete impl;
}

template <typename T>
inline af::array arrayFromVec(const std::vector<T>& vec, const Shape& shape)
{
	if(vec.size() != (size_t)shape.volume())
		throw AtError("Cannot create a tenstor with shape " + to_string(shape) + " from vector of size " + std::to_string(vec.size()));
	auto dims = shapeToDim4(shape);
	return af::array(dims, &vec[0]);
}

//std::vector<bool> is a special case where std::vector<bool> stores a array of 1 bit booleans. Expand before use
template <>
inline af::array arrayFromVec(const std::vector<bool>& vec, const Shape& shape)
{
	if(vec.size() != (size_t)shape.volume())
		throw AtError("Cannot create a tenstor with shape " + to_string(shape) + " from vector of size " + std::to_string(vec.size()));
	auto dims = shapeToDim4(shape);

	//AF uses char as bool internally: https://github.com/arrayfire/arrayfire/issues/346
	char* arr = new char[vec.size()];
	for(size_t i=0;i<vec.size();i++)
		arr[i] = (char)vec[i];
	af::array res = af::array(dims, arr);
	delete [] arr;
	return res;
}

TensorImpl* ArrayFireBackend::createTensor(const std::vector<float>& vec, const Shape& shape)
{
	return new AFTensorImpl(arrayFromVec(vec, shape), shape, this);
}

TensorImpl* ArrayFireBackend::createTensor(const std::vector<double>& vec, const Shape& shape)
{
	return new AFTensorImpl(arrayFromVec(vec, shape), shape, this);
}

TensorImpl* ArrayFireBackend::createTensor(const std::vector<int32_t>& vec, const Shape& shape)
{
	return new AFTensorImpl(arrayFromVec(vec, shape), shape, this);
}

TensorImpl* ArrayFireBackend::createTensor(const std::vector<int16_t>& vec, const Shape& shape)
{
	return new AFTensorImpl(arrayFromVec(vec, shape), shape, this);
}

TensorImpl* ArrayFireBackend::createTensor(const std::vector<bool>& vec, const Shape& shape)
{
	return new AFTensorImpl(arrayFromVec(vec, shape), shape, this);
}

TensorImpl* ArrayFireBackend::clone(const TensorImpl* handle)
{
	auto ptr = (const AFTensorImpl*)handle;
	return new AFTensorImpl(ptr->arr_.copy(), ptr->arrShape_, (ArrayFireBackend*) this);
}

TensorImpl* ArrayFireBackend::zeros(const Shape& shape, DType dtype)
{
	auto dims = shapeToDim4(shape);
	return new AFTensorImpl(std::move(af::constant(0, dims, dtypeToAFType(dtype))), shape, this);
}

TensorImpl* ArrayFireBackend::ones(const Shape& shape, DType dtype)
{
	auto dims = shapeToDim4(shape);
	return new AFTensorImpl(std::move(af::constant(1, dims, dtypeToAFType(dtype))), shape, this);
}

TensorImpl* ArrayFireBackend::rand(float lEdge, float rEdge, const Shape& shape)
{
	auto dims = shapeToDim4(shape);
	float span = rEdge - lEdge;
	//af::randu genrates float between 0 and 1, map it to the requested range
	af::array arr = (af::randu(dims)*span)-lEdge;
	arr.eval();
	return createTensor(std::move(arr), shape);
}

TensorImpl* ArrayFireBackend::normal(float mean, float stddev, const Shape& shape)
{
	//Should use arrayfire's af::randn all the way
	if(mean == 0.f && stddev == 1)
	{
		auto dims = shapeToDim4(shape);
		af::array arr = af::randn(dims);
		arr.eval();
		return createTensor(std::move(arr), shape);
	}

	std::minstd_rand eng; //Should be good enoguh for our purpose
	std::normal_distribution<float> dist(mean, stddev);
	std::vector<float> vec;

	size_t size = std::accumulate(shape.begin(), shape.end(), 1L, std::multiplies<size_t>());
	vec.resize(size);
	for(auto& v : vec)
		v = dist(eng);
	return createTensor(std::move(vec), shape);
}

void ArrayFireBackend::eval(TensorImpl* impl)
{
	auto ptr = (const AFTensorImpl*)impl;
	ptr->arr_.eval();
}

ArrayFireBackend::AFBackend ArrayFireBackend::getAFBackend() const
{
	auto b = af::getActiveBackend();
	if(b == AF_BACKEND_DEFAULT)
		return AFBackend::Default;
	else if(b == AF_BACKEND_CPU)
		return AFBackend::CPU;
	else if(b == AF_BACKEND_CUDA)
		return AFBackend::CUDA;
	else if(b == AF_BACKEND_OPENCL)
		return AFBackend::OpenCL;
	throw AtError("Unknown ArrayFire backend.");
}

void ArrayFireBackend::setAFBackend(AFBackend type)
{
	try
	{
		if(type == AFBackend::Default)
			af::setBackend(AF_BACKEND_DEFAULT);
		else if(type == AFBackend::CPU)
			af::setBackend(AF_BACKEND_CPU);
		else if(type == AFBackend::CUDA)
			af::setBackend(AF_BACKEND_CUDA);
		else if(type == AFBackend::OpenCL)
			af::setBackend(AF_BACKEND_OPENCL);
	}
	catch(af::exception& e)
	{
		throw AtError(std::string("Failed to set AF backend.\n") + e.what());
	}
}

Shape ArrayFireBackend::shape(const TensorImpl* impl) const
{
	return ((const AFTensorImpl*)impl)->arrShape_;
}

intmax_t ArrayFireBackend::size(const TensorImpl* impl) const
{
	auto ptr = (const AFTensorImpl*)impl;
	return (intmax_t)(ptr->arr_.bytes()/typeToSize(ptr->arr_.type()));
}

DType ArrayFireBackend::dtype(const TensorImpl* impl) const
{
	auto ptr = (const AFTensorImpl*)impl;
	return afTypeToDType(ptr->arr_.type());
}

void ArrayFireBackend::selfReciprocate(TensorImpl* impl)
{
	auto ptr = (AFTensorImpl*)impl;
	ptr->arr_ = 1.f/ptr->arr_;
}

void ArrayFireBackend::selfAdd(TensorImpl* impl, float val)
{
	auto ptr = (AFTensorImpl*)impl;
	ptr->arr_ += val;
}

void ArrayFireBackend::selfMul(TensorImpl* impl, float val)
{
	auto ptr = (AFTensorImpl*)impl;
	ptr->arr_ *= val;
}

void ArrayFireBackend::selfAdd(TensorImpl* impl, const TensorImpl* other)
{
	auto ptr = (AFTensorImpl*)impl;
	auto o = (const AFTensorImpl*)other;
	ptr->arr_ += o->arr_;
}

void ArrayFireBackend::selfMul(TensorImpl* impl, const TensorImpl* other)
{
	auto ptr = (AFTensorImpl*)impl;
	auto o = (const AFTensorImpl*)other;
	ptr->arr_ *= o->arr_;
}

void ArrayFireBackend::selfSub(TensorImpl* impl, const TensorImpl* other)
{
	auto ptr = (AFTensorImpl*)impl;
	auto o = (const AFTensorImpl*)other;
	ptr->arr_ -= o->arr_;
}

void ArrayFireBackend::selfDiv(TensorImpl* impl, const TensorImpl* other)
{
	auto ptr = (AFTensorImpl*)impl;
	auto o = (const AFTensorImpl*)other;
	ptr->arr_ /= o->arr_;
}

TensorImpl* ArrayFireBackend::sqrt(const TensorImpl* impl)
{
	auto ptr = (const AFTensorImpl*)impl;
	return new AFTensorImpl(af::sqrt(ptr->arr_), ptr->arrShape_, this);
}

TensorImpl* ArrayFireBackend::abs(const TensorImpl* impl)
{
	auto ptr = (const AFTensorImpl*)impl;
	return new AFTensorImpl(af::abs(ptr->arr_), ptr->arrShape_, this);
}

TensorImpl* ArrayFireBackend::exp(const TensorImpl* impl)
{
	auto ptr = (const AFTensorImpl*)impl;
	return new AFTensorImpl(af::exp(ptr->arr_), ptr->arrShape_, this);
}

TensorImpl* ArrayFireBackend::log(const TensorImpl* impl)
{
	auto ptr = (const AFTensorImpl*)impl;
	return new AFTensorImpl(af::log(ptr->arr_), ptr->arrShape_, this);
}

TensorImpl* ArrayFireBackend::pow(const TensorImpl* impl, float val)
{
	auto ptr = (const AFTensorImpl*)impl;
	return new AFTensorImpl(af::pow(ptr->arr_, val), ptr->arrShape_, this);
}

TensorImpl* ArrayFireBackend::dot(const TensorImpl* impl, const TensorImpl* other)
{
	auto& a1 = ((const AFTensorImpl*)impl)->arr_;
	auto& a2 = ((const AFTensorImpl*)other)->arr_;;
	int size = std::max(a1.numdims(), a2.numdims());
	af::array res;
	if(size >= 2)
		res = af::transpose(af::matmulTT(a1, a2));
	else
		res = af::dot(a1, a2);
	return new AFTensorImpl(res, shapeFromDim4(res.dims(), size), this);
}

void ArrayFireBackend::modDims(TensorImpl* impl, const Shape& wantedShape)
{
	auto ptr = (AFTensorImpl*)impl;
	auto dims = shapeToDim4(wantedShape);
	ptr->arrShape_ = wantedShape;
	ptr->arr_ = af::moddims(ptr->arr_, dims);
}

TensorImpl* ArrayFireBackend::reshape(const TensorImpl* impl, const Shape& wantedShape)
{
	AFTensorImpl* res = (AFTensorImpl*)clone(impl);
	modDims(res, wantedShape);
	return res;
}

TensorImpl* ArrayFireBackend::transpose(const TensorImpl* impl)
{
	auto ptr = (AFTensorImpl*)impl;
	Shape s = ptr->arrShape_;
	std::reverse(s.begin(), s.end());
	return new AFTensorImpl(af::transpose(ptr->arr_), s, this);
}

TensorImpl* ArrayFireBackend::concatenate(const std::vector<TensorImpl const*>& arrs, int axis)
{
	size_t maxNumDims = 0;
	Shape s = shape(arrs[0]);
	s[axis] = 0;
	//TODO: Check all array has the same # of dims
	for(auto t : arrs)
	{
		auto impl = (AFTensorImpl const*)t;
		maxNumDims = std::max(maxNumDims, shape(impl).size());

		s[axis] += shape(impl)[axis];
	}
	std::vector<af_array> inputs(arrs.size());
	for(size_t i=0;i<arrs.size();i++)
		inputs[i] = ((AFTensorImpl*)arrs[i])->get().get();
	af_array arr;
	af_join_many(&arr, maxNumDims-axis-1, arrs.size(), &inputs[0]);

	return new AFTensorImpl(af::array(arr), s, this);
}

TensorImpl* ArrayFireBackend::chunk(const TensorImpl* impl, const Shape& begin, const Shape& size)
{
	auto ptr = (AFTensorImpl*)impl;
	AtAssert(begin.size() == size.size() && begin.size() <= 4);
	af::seq dim[4];
	size_t sliceDims = begin.size();
	Shape s = shape(ptr);
	size_t tensorDims = s.size();
	AtAssert(sliceDims != 0);
	//XXX: af::end and af::span seem not to work when stored as af::seq. This is a workarround
	for(size_t i=0;i<sliceDims;i++)
	{
		dim[s.size()-i-1] = af::seq(begin[i], begin[i]+size[i]-1);
		s[i] = size[i];
	}

	af::array arr;
	if(tensorDims == 1)
		arr = ptr->arr_(dim[0], af::span, af::span, af::span);
	if(tensorDims == 2)
		arr = ptr->arr_(dim[0], dim[1], af::span, af::span);
	if(tensorDims == 3)
		arr = ptr->arr_(dim[0], dim[1], dim[2], af::span);
	if(tensorDims == 4)
		arr = ptr->arr_(dim[0], dim[1], dim[2], dim[3]);
	return new AFTensorImpl(arr, s, this);
}

TensorImpl* ArrayFireBackend::sum(const TensorImpl* impl, intmax_t axis)
{
	auto ptr = (AFTensorImpl*)impl;
	Shape s = shapeFromDim4(ptr->arr_.dims(), ptr->arrShape_.size());
	return new AFTensorImpl(af::sum(ptr->arr_, ptr->arrShape_.size()-axis-1), {s[axis]}, this);
}

void ArrayFireBackend::host(const TensorImpl* impl, float* ptr) const
{
	auto& arr = ((const AFTensorImpl*)impl)->arr_;
	copyToHost(arr, ptr);
}

void ArrayFireBackend::host(const TensorImpl* impl, double* ptr) const
{
	auto& arr = ((const AFTensorImpl*)impl)->arr_;
	copyToHost(arr, ptr);
}

void ArrayFireBackend::host(const TensorImpl* impl, int32_t* ptr) const
{
	auto& arr = ((const AFTensorImpl*)impl)->arr_;
	copyToHost(arr, ptr);
}

void ArrayFireBackend::host(const TensorImpl* impl, int16_t* ptr) const
{
	auto& arr = ((const AFTensorImpl*)impl)->arr_;
	copyToHost(arr, ptr);
}

void ArrayFireBackend::host(const TensorImpl* impl, bool* ptr) const
{
	auto& arr = ((const AFTensorImpl*)impl)->arr_;
	copyToHost(arr, ptr);
}

void ArrayFireBackend::device(TensorImpl* impl, const float* ptr)
{
	auto& arr = ((AFTensorImpl*)impl)->arr_;
	writeToDevice(arr, ptr);
}

void ArrayFireBackend::device(TensorImpl* impl, const double* ptr)
{
	auto& arr = ((AFTensorImpl*)impl)->arr_;
	writeToDevice(arr, ptr);
}

void ArrayFireBackend::device(TensorImpl* impl, const int32_t* ptr)
{
	auto& arr = ((AFTensorImpl*)impl)->arr_;
	writeToDevice(arr, ptr);
}

void ArrayFireBackend::device(TensorImpl* impl, const int16_t* ptr)
{
	auto& arr = ((AFTensorImpl*)impl)->arr_;
	writeToDevice(arr, ptr);
}

void ArrayFireBackend::device(TensorImpl* impl, const bool* ptr)
{
	auto& arr = ((AFTensorImpl*)impl)->arr_;
	writeToDevice(arr, ptr);
}

TensorImpl* ArrayFireBackend::greaterThan(const TensorImpl* impl,float val)
{
	auto ptr = (const AFTensorImpl*)impl;
	return new AFTensorImpl(ptr->arr_ > val, ptr->arrShape_, this);
}

TensorImpl* ArrayFireBackend::lesserThan(const TensorImpl* impl,float val)
{
	auto ptr = (const AFTensorImpl*)impl;
	return new AFTensorImpl(ptr->arr_ < val, ptr->arrShape_, this);
}

TensorImpl* ArrayFireBackend::greaterOrEqual(const TensorImpl* impl,float val)
{
	auto ptr = (const AFTensorImpl*)impl;
	return new AFTensorImpl(ptr->arr_ >= val, ptr->arrShape_, this);
}

TensorImpl* ArrayFireBackend::lesserOrEqual(const TensorImpl* impl,float val)
{
	auto ptr = (const AFTensorImpl*)impl;
	return new AFTensorImpl(ptr->arr_ <= val, ptr->arrShape_, this);
}

TensorImpl* ArrayFireBackend::equalTo(const TensorImpl* impl,float val)
{
	auto ptr = (const AFTensorImpl*)impl;
	return new AFTensorImpl(ptr->arr_ == val, ptr->arrShape_, this);
}
