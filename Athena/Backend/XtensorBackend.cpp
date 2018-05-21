#include <Athena/Tensor.hpp>
#include <Athena/DType.hpp>
#include <Athena/Backend/XtensorBackend.hpp>
#include <Athena/Backend/TensorImpl.hpp>

#include <xtensor/xarray.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xvectorize.hpp>
//#include <xtensor/xstrided_view.hpp> //Crashes clang

#include <xtensor-blas/xlinalg.hpp>

#include <random>
#include <chrono>

#include <string.h>

#define _unused(x) ((void)(x))

using namespace At;

//Converts between different containers
template<typename ResType, typename InType>
inline ResType as(const InType& shape)
{
	return ResType(shape.begin(), shape.end());
}

namespace
{

class Xarr
{
public:
	Xarr() = default;
	Xarr(void* ptr, DType dtype) : tensorPtr_(ptr), dataType(dtype){}
	Xarr(const Xarr& other)
	{
		dataType = other.dataType;
		tensorPtr_ = other.run<void*>([](auto arr){return new decltype(arr)(arr);});
	}

	Xarr(Xarr&& other) noexcept
	{
		dataType = other.dataType;
		tensorPtr_ = other.tensorPtr_;

		other.tensorPtr_ = nullptr;
		other.dataType = DType::unknown;
	}

	Xarr operator= (const Xarr& other)
	{
		if(this == &other)
			return *this;
		release();
		dataType = other.dataType;
		tensorPtr_ = other.run<void*>([](auto arr){return new decltype(arr)(arr);});
		return *this;
	}

	template <typename T>
	Xarr(xt::xarray<T> arr)
	{
		setInternalData(arr);
	}

	template <typename T>
	void setInternalData(xt::xarray<T> array)
	{
		if(dtype() == typeToDtype<T>() && tensorPtr_ != nullptr)
		{

			get<T>() = array;
			return;
		}
		release();

		tensorPtr_ = new xt::xarray<T>(array);
		dataType = typeToDtype<T>();
	}
	
	void release()
	{
		//WORKARROUND: if(...) delete ... else if(...) delete... does not work and produces NaN
		// for no apparent reason.  Maybe a compiler bug?
		delete static_cast<xt::xarray<float>*>(dtype() == DType::float32 ? tensorPtr_ : nullptr);
		delete static_cast<xt::xarray<double>*>(dtype() == DType::float64 ? tensorPtr_ : nullptr);
		delete static_cast<xt::xarray<int16_t>*>(dtype() == DType::int16 ? tensorPtr_ : nullptr);
		delete static_cast<xt::xarray<int32_t>*>(dtype() == DType::int32 ? tensorPtr_ : nullptr);
		delete static_cast<xt::xarray<bool>*>(dtype() == DType::bool8 ? tensorPtr_ : nullptr);
		tensorPtr_ = nullptr;
		dataType = DType::unknown;
	}

	virtual ~Xarr()
	{
		release();
	}

	size_t size() const
	{
		return run<size_t>([](const auto& arr){return arr.size();});
	}

	inline xt::svector<size_t> shape() const
	{
		return run<xt::svector<size_t>>([](const auto& arr){return arr.shape();});
	}

	DType dtype() const
	{
		return dataType;
	}

	template <typename T>
	xt::xarray<T>& get()
	{
		if(dtype() != typeToDtype<T>())
			throw AtError(std::string("Requesting xarray<") + dtypeToName(typeToDtype<T>()) + "> from xarray<" + dtypeToName(dtype())+">");
		return *reinterpret_cast<xt::xarray<T>*>(tensorPtr_);
	}

	template <typename T>
	const xt::xarray<T>& get() const
	{
		if(dtype() != typeToDtype<T>())
			throw AtError(std::string("Requesting xarray<") + dtypeToName(typeToDtype<T>()) + "> from xarray<" + dtypeToName(dtype())+">");
		return *reinterpret_cast<xt::xarray<T>*>(tensorPtr_);
	}

	template <typename T>
	Xarr operator+ (const T& val) const
	{
		static_assert(std::is_scalar<T>::value == true); //Just to be on the safe side
		return run<Xarr>([val](const auto& a){return a+val;});
	}

	template <typename T>
	void operator+= (const T& val)
	{
		return run<void>([this, val](auto& a){return this->setInternalData(xt::eval(a+val));});
	}

	template <typename T>
	Xarr operator* (const T& val) const
	{
		static_assert(std::is_scalar<T>::value == true); //Just to be on the safe side
		return run<Xarr>([val](const auto& a){return a*val;});
	}

	template <typename T>
	void operator*= (const T& val)
	{
		return run<void>([this, val](auto& a){return this->setInternalData(xt::eval(a*val));});
	}

	template <typename T>
	Xarr operator- (const T& val) const
	{
		static_assert(std::is_scalar<T>::value == true); //Just to be on the safe side
		return run<Xarr>([val](const auto& a){return a-val;});
	}

	template <typename T>
	void operator-= (const T& val)
	{
		return run<void>([this, val](auto& a){return this->setInternalData(xt::eval(a-val));});
	}

	template <typename T>
	Xarr operator/ (const T& val) const
	{
		static_assert(std::is_scalar<T>::value == true); //Just to be on the safe side
		return run<Xarr>([val](const auto& a){return a/val;});
	}

	template <typename T>
	void operator/= (const T& val)
	{
		return run<void>([this, val](auto& a){return this->setInternalData(xt::eval(a/val));});
	}

	Xarr operator< (float val) const
	{
		return run<Xarr>([val](const auto& a){return xt::eval(a<val);});
	}

	Xarr operator> (float val) const
	{
		return run<Xarr>([val](const auto& a){return xt::eval(a>val);});
	}

	Xarr operator<= (float val) const
	{
		return run<Xarr>([val](const auto& a){return xt::eval(a<=val);});
	}

	Xarr operator>= (float val) const
	{
		return run<Xarr>([val](const auto& a){return xt::eval(a>=val);});
	}

	Xarr operator== (float val) const
	{
		return run<Xarr>([val](const auto& a){return xt::eval(xt::equal(a, val));});
	}


	void reciprocate()
	{
		return run<void>([this](auto& a){setInternalData(xt::eval(1.f/a));});
	}

	void reshape(xt::svector<size_t> s)
	{
		return run<void>([this, s](auto& a){a.reshape(s); setInternalData(a);});
	}

	Xarr dot(const Xarr& other) const
	{
		return run<Xarr>(other, [](const auto& a, const auto& b){
			using ValueType = typename std::decay<decltype(a)>::type::value_type;
			xt::xarray<ValueType> res = xt::linalg::dot(a, b);
			return res;
		});
	}

	Xarr sqrt() const
	{
		return run<Xarr>([](const auto& a){return xt::eval(xt::sqrt(a));});
	}

	Xarr transpose() const
	{
		return run<Xarr>([](const auto& a){return xt::eval(xt::transpose(a));});
	}

	Xarr transpose(xt::svector<size_t> axises) const
	{
		return run<Xarr>([&axises](const auto& a){return xt::eval(xt::transpose(a, axises));});
	}

	Xarr sum(xt::svector<size_t> axises) const
	{
		return run<Xarr>([&axises](const auto& a){
			using ValueType = typename std::decay<decltype(a)>::type::value_type;
			return xt::xarray<ValueType>(xt::sum(a, axises));
		});
	}

	Xarr abs() const
	{
		return run<Xarr>([](const auto& a){return xt::eval(xt::abs(a));});
	}

	Xarr exp() const
	{
		return run<Xarr>([](const auto& a){return xt::eval(xt::exp(a));});
	}

	Xarr log() const
	{
		return run<Xarr>([](const auto& a){return xt::eval(xt::log(a));});
	}

	Xarr pow(float val) const
	{
		return run<Xarr>([val](const auto& a){return xt::eval(xt::pow(a, val));});
	}

	inline Xarr stack(const Xarr& arr, size_t axis) const
	{
		if(dtype() != arr.dtype())
			throw AtError("Cannot stack tensors of different type" + dtypeToName(dtype()) + " vs " + dtypeToName(arr.dtype()));
		return run<Xarr>([&arr, axis](const auto& a){
			using ValueType = typename std::decay<decltype(a)>::type::value_type;
			const auto b = arr.get<ValueType>();
			return (xt::xarray<ValueType>)(xt::stack(xt::xtuple(a, b)), axis);
		});

	}

	inline Xarr chunk(xt::slice_vector sv) const
	{
		return run<Xarr>([&sv](const auto& a){
			using DataType = typename std::decay<decltype(a)>::type::value_type;
			return (xt::xarray<DataType>)xt::strided_view(a, sv);
		});
	}

	inline Xarr concatenate(const Xarr& arr, size_t axis) const
	{
		if(dtype() != arr.dtype())
			throw AtError("Cannot concatenate tensors of different types. " + dtypeToName(dtype()) + " vs " + dtypeToName(arr.dtype()));
		return concatenate({this, &arr}, axis);
	}

	inline Xarr concatenate(const std::vector<Xarr const*>& arr, size_t axis) const
	{
		assert(arr.size() > 1);
		auto type = arr[0]->dtype();
		for(auto a : arr)
		{
			if(a->dtype() != type)
				throw AtError("Cannot concatenate tensors of different types. ");
			if(a->shape().size() < axis)
				throw AtError("Concatenate axis out of bounds");
		}

		return run<Xarr>([&arr, axis](const auto& a){
			using ValueType = typename std::decay<decltype(a)>::type::value_type;
			auto s = arr[0]->get<ValueType>().shape();

			s[axis] = 0;
			for(auto a : arr)
				s[axis] += a->get<ValueType>().shape()[axis];
			auto res = xt::xarray<ValueType>::from_shape(s);
			size_t currentLoc = 0;
			xt::slice_vector sv;
			for(size_t i=0;i<s.size();i++)
				sv.push_back(xt::all());

			for(auto a : arr)
			{
				auto& xtarr = a->get<ValueType>();
				auto arrShape = xtarr.shape()[axis];
				sv[axis] = xt::range((int)currentLoc, (int)currentLoc+arrShape);
				currentLoc += arrShape;
				xt::strided_view(res, sv) = xtarr;
			}
			return (xt::xarray<ValueType>)res;
		});
	}

	template <typename T>
	T* data()
	{
		return (T*)run<void*>([](auto& a){return &a[0];});
	}

	template <typename T>
	const T* data() const
	{
		return (const T*)run<const void*>([](const auto& a){return &a[0];});
	}



protected:

	template <typename Ret, typename Op>
	inline Ret run(Op op) const
	{
		if(dtype() == DType::float32)
			return op(get<float>());
		else if(dtype() == DType::float64)
			return op(get<double>());
		else if(dtype() == DType::int32)
			return op(get<int32_t>());
		else if(dtype() == DType::int16)
			return op(get<int16_t>());
		else if(dtype() == DType::bool8)
			return op(get<bool>());
		throw AtError("Operand type not supported. This is a bug.");
	}

	template <typename Ret, typename Op>
	inline Ret run(Op op)
	{
		if(dtype() == DType::float32)
			return op(get<float>());
		else if(dtype() == DType::float64)
			return op(get<double>());
		else if(dtype() == DType::int32)
			return op(get<int32_t>());
		else if(dtype() == DType::int16)
			return op(get<int16_t>());
		else if(dtype() == DType::bool8)
			return op(get<bool>());
		throw AtError("Operand type not supported. This is a bug.");
	}

	template <typename Ret, typename Op>
	inline Ret run(const Xarr& arr, Op op) const
	{
		return run<Ret>([&op, &arr](const auto& a){
			return arr.run<Ret>([&op, &a](const auto& b)->Ret{
				return op(a, b);
			});
		});
	}

	template <typename Ret, typename Op>
	inline Ret run(const Xarr& arr, Op op)
	{
		return run<Ret>([&op, &arr](auto& a){
			return arr.run<Ret>([&op, &a](const auto& b)->Ret{
				return op(a, b);
			});
		});
	}

	void* tensorPtr_ = nullptr;
	DType dataType = DType::unknown;
};

template <>
Xarr Xarr::operator+ (const Xarr& other) const
{
	return run<Xarr>(other, [](const auto& a, const auto& b){return xt::eval(a+b);});
}

template <>
Xarr Xarr::operator* (const Xarr& other) const
{
	return run<Xarr>(other, [](const auto& a, const auto& b){return xt::eval(a*b);});
}

template <>
Xarr Xarr::operator- (const Xarr& other) const
{
	return run<Xarr>(other, [](const auto& a, const auto& b){return xt::eval(a-b);});
}

template <>
Xarr Xarr::operator/ (const Xarr& other) const
{
	return run<Xarr>(other, [](const auto& a, const auto& b){return xt::eval(a/b);});
}

template <>
void Xarr::operator+= (const Xarr& val)
{
	return run<void>(val, [](auto& a, const auto& b){a += b;});
}

template <>
void Xarr::operator*= (const Xarr& val)
{
	return run<void>(val, [](auto& a, const auto& b){a *= b;});
}

template <>
void Xarr::operator/= (const Xarr& val)
{
	return run<void>(val, [](auto& a, const auto& b){a /= b;});
}

template <>
void Xarr::operator-= (const Xarr& val)
{
	return run<void>(val, [](auto& a, const auto& b){a -= b;});
}


template <typename T>
void copyToPtr(const Xarr& arr, T* dest)
{
	if(arr.dtype() != typeToDtype<T>())
		throw AtError("Cannot copy data from xarray to pointer. Incompatible data type. " + dtypeToName(typeToDtype<T>()) + " vs " + dtypeToName(arr.dtype()));
	const auto& array = arr.get<T>();
	std::copy(array.begin(), array.end(), dest);
}

template <typename T>
void copyFromPtr(Xarr& arr, const T* src)
{
	if(arr.dtype() != typeToDtype<T>())
		throw AtError("Cannot copy data from pointer to xarray. Incompatible data type. " + dtypeToName(arr.dtype()) + " vs " + dtypeToName(typeToDtype<T>()));
	auto& array = arr.get<T>();
	memcpy(&array[0], src, array.size()*sizeof(T));
}

}

class XtensorTensorImpl : public TensorImpl
{
public:
	XtensorTensorImpl(XtensorBackend* backend) : TensorImpl(backend)
	{
	}

	template <typename T>
	XtensorTensorImpl(xt::xarray<T> arr, XtensorBackend* backend) : TensorImpl(backend)
	{
		arr_.setInternalData(arr);
	}

	XtensorTensorImpl(Xarr arr, XtensorBackend* backend) : TensorImpl(backend)
	{
		arr_ = arr;
	}

	const Xarr& xarr() const
	{
		return arr_;
	}

	template <typename T>
	const xt::xarray<T>& get() const
	{
		return arr_.get<T>();
	}

	template <typename T>
	xt::xarray<T>& get()
	{
		return arr_.get<T>();
	}

	virtual void host(float* ptr) const override
	{
		copyToPtr(arr_, ptr);
	}

	virtual void host(double* ptr) const override
	{
		copyToPtr(arr_, ptr);
	}

	virtual void host(int32_t* ptr) const override
	{
		copyToPtr(arr_, ptr);
	}

	virtual void host(int16_t* ptr) const override
	{
		copyToPtr(arr_, ptr);
	}

	virtual void host(bool* ptr) const override
	{
		copyToPtr(arr_, ptr);
	}

	virtual void device(const float* ptr) override
	{
		copyFromPtr(arr_, ptr);
	}

	virtual void device(const double* ptr)
	{
		copyFromPtr(arr_, ptr);
	}

	virtual void device(const int32_t* ptr)
	{
		copyFromPtr(arr_, ptr);
	}

	virtual void device(const int16_t* ptr)
	{
		copyFromPtr(arr_, ptr);
	}

	virtual void device(const bool* ptr)
	{
		copyFromPtr(arr_, ptr);
	}

	virtual size_t size() const override
	{
		return arr_.size();
	}

	virtual Shape shape() const override
	{
		return as<Shape>(arr_.shape());
	}

	virtual void add(float val) override
	{
		arr_ += val;
	}

	virtual void mul(float val) override
	{
		arr_ *= val;
	}

	//TODO: Check if the incoming impl is a XtensorTensorImpl
	virtual void add(const TensorImpl* other) override
	{
		auto impl = (const XtensorTensorImpl*)other;
		arr_ += impl->xarr();
	}

	virtual void mul(const TensorImpl* other) override
	{
		auto impl = (const XtensorTensorImpl*)other;
		arr_ *= impl->xarr();
	}

	virtual void subtract(const TensorImpl* other) override
	{
		auto impl = (const XtensorTensorImpl*)other;
		arr_ -= impl->xarr();
	}

	virtual void divide(const TensorImpl* other) override
	{
		auto impl = (const XtensorTensorImpl*)other;
		arr_ /= impl->xarr();
	}

	virtual void reciprocate() override
	{
		arr_.reciprocate();
	}

	virtual TensorImpl* clone() const override
	{
		return new XtensorTensorImpl(arr_, (XtensorBackend*) backend());
	}

	virtual void resize(const Shape& wantedShape) override
	{
		auto s = as<xt::svector<size_t>>(wantedShape);
		arr_.reshape(s);
	}

	virtual TensorImpl* reshape(const Shape& wantedShape) const override
	{
		auto s = as<xt::svector<size_t>>(wantedShape);
		Xarr arr = arr_;
		arr.reshape(s);
		return new XtensorTensorImpl(arr, (XtensorBackend*)backend());
	}

	virtual TensorImpl* dot(const TensorImpl* other) const override
	{
		auto impl = (const XtensorTensorImpl*)other;
		return new XtensorTensorImpl(arr_.dot(impl->xarr()), (XtensorBackend*)backend());
	}

	virtual TensorImpl* sqrt() const override
	{
		return new XtensorTensorImpl(arr_.sqrt(), (XtensorBackend*)backend());
	}

	virtual TensorImpl* transpose() const override
	{
		return new XtensorTensorImpl(arr_.transpose(), (XtensorBackend*)backend());
	}

	virtual TensorImpl* transpose(const std::vector<intmax_t>& axis) const override
	{
		auto a = as<xt::svector<size_t>>(axis);
		return new XtensorTensorImpl(arr_.transpose(a), (XtensorBackend*)backend());
	}


	virtual TensorImpl* sum(intmax_t axis) const override
	{
		return new XtensorTensorImpl(arr_.sum({(size_t)axis}), (XtensorBackend*)backend());
	}

	virtual TensorImpl* sum(const std::vector<intmax_t>& axis) const override
	{
		auto s = as<xt::svector<size_t>>(axis);
		return new XtensorTensorImpl(arr_.sum(s), (XtensorBackend*)backend());
	}

	virtual TensorImpl* pow(float val) const override
	{
		return new XtensorTensorImpl(arr_.pow(val), (XtensorBackend*)backend());
	}

	virtual TensorImpl* slice(const Shape& begin, const Shape& size) const override
	{
		xt::slice_vector sv;
		for(size_t i=0;i<begin.size();i++)
			sv.push_back(xt::range((int)begin[i], (int)(begin[i]+size[i])));//Why int...?
		return new XtensorTensorImpl(arr_.chunk(sv), (XtensorBackend*)backend());
	}

	virtual TensorImpl* abs() const override
	{
		return new XtensorTensorImpl(arr_.abs(), (XtensorBackend*)backend());
	}

	TensorImpl* stack(const TensorImpl* other, int axis) const override
	{
		auto impl = (const XtensorTensorImpl*)other;
		return new XtensorTensorImpl(arr_.stack(impl->xarr(), axis), (XtensorBackend*)backend());
	}

	TensorImpl* concatenate(const TensorImpl* other, int axis) const override
	{
		auto impl = (const XtensorTensorImpl*)other;
		return new XtensorTensorImpl(arr_.concatenate(impl->xarr(), axis), (XtensorBackend*)backend());
	}

	TensorImpl* concatenate(const std::vector<TensorImpl const*>& arrs, int axis) const override
	{
		std::vector<Xarr const*> vec(arrs.size());
		for(size_t i=0;i<arrs.size();i++)
			vec[i] = &((XtensorTensorImpl const*)arrs[i])->xarr();
		return new XtensorTensorImpl(vec[0]->concatenate(vec, axis), (XtensorBackend*)backend());
	}

	virtual TensorImpl* exp() const override
	{
		return new XtensorTensorImpl(arr_.exp(), (XtensorBackend*)backend());
	}

	virtual TensorImpl* log() const override
	{
		return new XtensorTensorImpl(arr_.log(), (XtensorBackend*)backend());
	}

	virtual TensorImpl* greaterThan(float val) const override
	{
		return new XtensorTensorImpl(arr_ > val, (XtensorBackend*)backend());
	}

	virtual TensorImpl* lesserThan(float val) const override
	{
		return new XtensorTensorImpl(arr_ < val, (XtensorBackend*)backend());
	}

	virtual TensorImpl* greaterOrEqual(float val) const override
	{
		return new XtensorTensorImpl(arr_ >= val, (XtensorBackend*)backend());
	}

	virtual TensorImpl* lesserOrEqual(float val) const override

	{
		return new XtensorTensorImpl(arr_ <= val, (XtensorBackend*)backend());
	}

	virtual TensorImpl* equalTo(float val) const override
	{
		return new XtensorTensorImpl(arr_ == val, (XtensorBackend*)backend());
	}

	virtual DType dtype() const override
	{
		return arr_.dtype();
	}

	virtual void* hostPtr() override
	{
		return arr_.data<void>();
	}

	virtual const void* hostPtr() const override
	{
		return arr_.data<void>();
	}

	virtual void eval() override
	{
		//No-op for xtensor.
	}

protected:
	Xarr arr_;
};

template <typename T>
inline xt::xarray<T>& get(Tensor& t)
{
	return ((XtensorTensorImpl*)t.pimpl())->get<T>();
}

template <typename T>
inline const xt::xarray<float>& get(const Tensor& t)
{
	return ((const XtensorTensorImpl*)t.pimpl())->get<T>();
}

inline xt::xarray<float> im2col(const xt::xarray<float>& img, std::array<intmax_t, 2> window, std::array<intmax_t, 2> stride)
{
	intmax_t strideY = stride[0];
	intmax_t strideX = stride[1];

	intmax_t inputNums = img.shape()[0];
	intmax_t inputChannels = img.shape()[1];
	intmax_t inputHeight = img.shape()[2];
	intmax_t inputWidth = img.shape()[3];

	intmax_t inputImageSize = inputHeight*inputWidth;
	intmax_t inputChannelSize = inputChannels*inputImageSize;

	intmax_t filterChannels = inputChannels;
	intmax_t filterHeight = window[0];
	intmax_t filterWidth = window[1];

	intmax_t filterSufaceSize = filterHeight*filterWidth;
	intmax_t filterChannlelSize = filterHeight*filterWidth*filterChannels;

	intmax_t outputHeight = (inputHeight-filterHeight)/strideY+1;
	intmax_t outputWidth = (inputWidth-filterWidth)/strideX+1;
	intmax_t convImageSize = outputHeight*outputWidth;

	intmax_t intermedChannelSize = convImageSize*filterChannlelSize;
	xt::xarray<float> res = xt::zeros<float>({inputNums, convImageSize, filterChannlelSize});

	//im2col
	for(intmax_t n=0;n<inputNums;n++)
	{
		for(intmax_t c=0;c<inputChannels;c++)
		{
			for(int dy=0;dy<filterWidth;dy++)
			{

				for(int dx=0;dx<filterWidth;dx++)
				{
					for(intmax_t y=0;y<outputHeight;y++)
					{
						intmax_t ry = (y*strideY)+dy;
						for(intmax_t x=0;x<outputWidth;x++)
						{
							intmax_t rx = x*strideX+dx;
							res[n*intermedChannelSize+(y*outputWidth+x)*filterChannlelSize+c*filterSufaceSize+dy*filterWidth+dx]
								= img[n*inputChannelSize+c*inputImageSize+ry*inputWidth+rx];
						}
					}
				}
			}
		}
	}
	return res;
}

inline xt::xarray<float> col2im(const xt::xarray<float>& in, const Shape& imgSize, std::array<intmax_t, 2> window, std::array<intmax_t, 2> stride)
{
	intmax_t strideY = stride[0];
	intmax_t strideX = stride[1];
	xt::xarray<float> col = xt::transpose(in);

	intmax_t inputNums = imgSize[0];
	intmax_t inputChannels = imgSize[1];
	intmax_t inputHeight = imgSize[2];
	intmax_t inputWidth = imgSize[3];
	//assert(col.shape()[1]%(window[0]*window[1]) == 0);

	intmax_t inputImageSize = inputHeight*inputWidth;
	intmax_t inputChannelSize = inputChannels*inputImageSize;

	intmax_t filterChannels = inputChannels;
	intmax_t filterHeight = window[0];
	intmax_t filterWidth = window[1];

	intmax_t filterSufaceSize = filterHeight*filterWidth;
	intmax_t filterChannlelSize = filterHeight*filterWidth*filterChannels;

	intmax_t outputHeight = (inputHeight-filterHeight)/strideY+1;
	intmax_t outputWidth = (inputWidth-filterWidth)/strideX+1;
	intmax_t convImageSize = outputHeight*outputWidth;

	intmax_t intermedChannelSize = convImageSize*filterChannlelSize;
	xt::xarray<float> res = xt::zeros<float>({inputNums, inputChannels, inputHeight, inputWidth});

	//col2im
	for(intmax_t n=0;n<inputNums;n++)
	{
		for(intmax_t c=0;c<inputChannels;c++)
		{
			for(intmax_t dy=0;dy<filterWidth;dy++)
			{
				for(intmax_t y=0;y<outputHeight;y++)
				{
					for(intmax_t x=0;x<outputWidth;x++)
					{
						intmax_t ry = (y*strideY)+dy;
						for(int dx=0;dx<filterWidth;dx++)
						{
							intmax_t rx = x*strideX+dx;
							res[n*inputChannelSize+c*inputImageSize+ry*inputWidth+rx]
								+= col[n*intermedChannelSize+(y*outputWidth+x)*filterChannlelSize+c*filterSufaceSize+dy*filterWidth+dx];
						}
					}
				}
			}
		}
	}
	return res;
}

XtensorBackend::XtensorBackend()
{
	addAlgorithm<FCForwardFunction>("fullyconnectedForward",
	[this](const Tensor& in, const Tensor& weight, const Tensor& bias)->Tensor
	{
		const auto& i = get<float>(in);
		const auto& w = get<float>(weight);
		const auto& b = get<float>(bias);
		auto res = new XtensorTensorImpl(
			xt::eval(xt::linalg::dot(i,w)+b), this
		);
		return res;
	});

	addAlgorithm<FCBackwardFunction>("fullyconnectedBackward",
	[this](const Tensor& dx, const Tensor& weight)->Tensor
	{
		const auto& i = get<float>(dx);
		const auto& w = get<float>(weight);
		xt::xarray<float> arr = xt::eval(xt::linalg::dot(i,xt::transpose(w)));
		return createTensor(arr);
	});

	addAlgorithm<SigmoidForward>("sigmoidForward",
		[this](const Tensor& x)->Tensor
		{
			const auto& t = get<float>(x);
			return createTensor(xt::eval(1/(1+xt::exp(-t))));
		});

	addAlgorithm<SigmoidBackward>("sigmoidBackward",
		[this](const Tensor& a, const Tensor& b)->Tensor
		{
			const auto& dy = get<float>(a);
			const auto& y = get<float>(b);
			return createTensor(xt::eval(dy*(y*(1-y))));
		});

	addAlgorithm<TanhForward>("tanhForward",
		[this](const Tensor& x)->Tensor
		{
			const auto& t = get<float>(x);
			xt::xarray<float> res = xt::tanh(t);
			return createTensor(std::move(res));
		});

	addAlgorithm<TanhBackward>("tanhBackward",
		[this](const Tensor& a, const Tensor& b)->Tensor
		{
			const auto& dy = get<float>(a);
			const auto& y = get<float>(b);
			xt::xarray<float> res = dy * (1 - xt::pow(xt::tanh(y), 2));
			return createTensor(std::move(res));
		});

	addAlgorithm<ReluForward>("reluForward",
		[this](const Tensor& x)->Tensor
		{
			const auto& t = get<float>(x);
			xt::xarray<float> res = (t>0)*t;
			return createTensor(std::move(res));
		});

	addAlgorithm<ReluBackward>("reluBackward",
		[this](const Tensor& a, const Tensor& b)->Tensor
		{
			const auto& dy = get<float>(a);
			const auto& y = get<float>(b);
			xt::xarray<float> res = dy*(y>0);
			return createTensor(std::move(res));
		});
	
	addAlgorithm<LeakyReluForward>("leakyReluForward",
		[this](const Tensor& x, float alpha)->Tensor
		{
			const auto& t = get<float>(x);
			auto f = xt::vectorize([&](float v){return (v>0?v:v*alpha);});
			xt::xarray<float> res = f(t);
			return createTensor(std::move(res));
		});

	addAlgorithm<LeakyReluBackward>("leakyReluBackward",
		[this](const Tensor& a, const Tensor& b, float alpha)->Tensor
		{
			const auto& dy = get<float>(a);
			const auto& y = get<float>(b);
			auto f = xt::vectorize([&](float y, float dy){return dy*(y>0?1.f:alpha);});
			xt::xarray<float> res = f(y, dy);
			return createTensor(std::move(res));
		});
	
	addAlgorithm<Conv2DForward>("conv2DForward",
		[this](const Tensor& x, const Tensor& weights, const Tensor& bias, const Shape& strides)->Tensor
		{
			//assuming input format of N C H W
			const auto& t = get<float>(x);
			const auto& kernels = get<float>(weights);
			const auto& b = get<float>(bias);

			intmax_t inputNums = x.shape()[0];
			intmax_t inputChannels = x.shape()[1];
			intmax_t inputHeight = x.shape()[2];
			intmax_t inputWidth = x.shape()[3];

			intmax_t filterNums =  kernels.shape()[0];
			intmax_t filterChannels = kernels.shape()[1];
			intmax_t filterHeight = kernels.shape()[2];
			intmax_t filterWidth = kernels.shape()[3];

			AtAssert(inputChannels == filterChannels);
			_unused(inputChannels);

			intmax_t filterChannlelSize = filterHeight*filterWidth*filterChannels;

			intmax_t outputHeight = (inputHeight-filterHeight)/strides[0]+1;
			intmax_t outputWidth = (inputWidth-filterWidth)/strides[1]+1;

			xt::xarray<float> tmpBuffer = im2col(t, {{filterHeight, filterWidth}}, {{strides[0], strides[1]}});

			xt::xarray<float> convKernel = kernels;
			convKernel.reshape({(size_t)filterNums, (size_t)filterChannlelSize});
			convKernel = xt::transpose(convKernel);
			xt::xarray<float> res = xt::linalg::dot(tmpBuffer, convKernel);
			res = xt::transpose(res);
			res.reshape({(size_t)inputNums, (size_t)filterNums, (size_t)outputHeight, (size_t)outputWidth});

			intmax_t outputCubeSize = filterNums*outputHeight*outputWidth;
			intmax_t outputSurfaceSize = outputHeight*outputWidth;

			//Apply bias
			for(intmax_t n=0;n<inputNums;n++)
			{
				for(intmax_t c=0;c<filterNums;c++)
				{
					for(intmax_t i=0;i<outputSurfaceSize;i++)
						res[n*outputCubeSize+c*outputSurfaceSize+i] += b[c];
				}
			}

			return createTensor(res);

		});

		//TODO: Optimize this function. It is super slow now.
		addAlgorithm<Conv2DBackward>("conv2DBackward",
			[this](const Tensor& prevOut, const Tensor& kernel, Tensor& dW, Tensor& db , const Tensor& currDelta,
				const Shape& strides)->Tensor
		{
			intmax_t batchSize = prevOut.shape()[0];
			intmax_t numFilters = kernel.shape()[0];

			const auto& dout = get<float>(currDelta);
			const auto& x = get<float>(prevOut);
			const auto& w = get<float>(kernel);

			db = currDelta.sum({0, 2, 3});
			db.resize({db.shape()[0], db.shape().volume()/db.shape()[0]});

			xt::xarray<float> xCol = im2col(x, {{kernel.shape()[2], kernel.shape()[3]}}, {{strides[0], strides[1]}});

			xt::xarray<float> doutReshaped = xt::transpose(dout, {1, 2, 3, 0});
			doutReshaped.reshape({(size_t)batchSize, (size_t)numFilters, (size_t)currDelta.size()/(numFilters*batchSize)});
			xt::xarray<float> tmp = xt::linalg::dot(doutReshaped, xt::transpose(xCol));
			tmp.reshape(w.shape());
			dW = createTensor(tmp);

			xt::xarray<float> wReshape = w;
			wReshape.reshape({(size_t)numFilters, w.size()/numFilters});
			xt::xarray<float> dxCol = xt::linalg::dot(xt::transpose(wReshape), doutReshaped);
			xt::xarray<float>  res = col2im(dxCol, prevOut.shape()
				,{{kernel.shape()[2], kernel.shape()[3]}}, {{strides[0], strides[1]}});

			return createTensor(res);
		});


	setType("xtensor");
}

template <typename T>
inline xt::xarray<T> makeXarray(const T* ptr, Shape shape)
{
	auto s = as<typename xt::xarray<T>::shape_type>(shape);
	xt::xarray<T> t(s);
	std::copy(ptr, ptr+shape.volume(), t.begin());
	return t;
}

template <typename Ret, typename Op>
Ret run(DType dtype, Op op)
{
	if(dtype == DType::float32)
		return op(0.f);
	else if(dtype == DType::float64)
		return op(0.0);
	else if (dtype == DType::int32)
		return op(0);
	else if (dtype == DType::int16)
		return op((int16_t)0);
	else if (dtype == DType::bool8)
		return op(false);
	AtAssert(dtype!=DType::unknown, "Reaching end of internal function, This is a bug");

	throw AtError("Reaching after assert failed. Something is really wrong");
	return Ret();
}

TensorImpl* XtensorBackend::createTensor(const Shape& dims)
{
	std::vector<size_t> size(dims.size());
	std::copy(dims.begin(), dims.end(), size.begin());
	return createTensor(xt::xarray<float>::from_shape(size));
}

TensorImpl* XtensorBackend::createTensor(const std::vector<float>& vec, const Shape& shape)
{
	assert(vec.size() == (size_t)shape.volume());
	return new XtensorTensorImpl(makeXarray(&vec[0], shape), this);
}

TensorImpl* XtensorBackend::createTensor(const std::vector<double>& vec, const Shape& shape)
{
	assert(vec.size() == (size_t)shape.volume());
	return new XtensorTensorImpl(makeXarray(&vec[0], shape), this);
}

TensorImpl* XtensorBackend::createTensor(const std::vector<int32_t>& vec, const Shape& shape)
{
	assert(vec.size() == (size_t)shape.volume());
	return new XtensorTensorImpl(makeXarray(&vec[0], shape), this);
}

TensorImpl* XtensorBackend::createTensor(const std::vector<int16_t>& vec, const Shape& shape)
{
	assert(vec.size() == (size_t)shape.volume());
	return new XtensorTensorImpl(makeXarray(&vec[0], shape), this);
}

TensorImpl* XtensorBackend::createTensor(const std::vector<bool>& vec, const Shape& shape)
{
	assert(vec.size() == (size_t)shape.volume());

	//std::vector<bool> is compressed, decomressing
	std::vector<int8_t> data(vec.size());
	for(size_t i=0;i<vec.size();i++)
		data[i] = vec[i];
	return new XtensorTensorImpl(makeXarray((bool*)data.data(), shape), this);
}

TensorImpl* XtensorBackend::createTensor(const xt::xarray<float>& arr)
{
	xt::xarray<float> t(arr);
	return new XtensorTensorImpl(std::move(t), this);
}

void XtensorBackend::destoryTensor(TensorImpl* impl)
{
	delete impl;
}

TensorImpl* XtensorBackend::zeros(const Shape& shape, DType dtype)
{
	return new XtensorTensorImpl(run<Xarr>(dtype, [&shape](auto dummy){return xt::eval(xt::zeros<decltype(dummy)>(shape));}), this);
}


TensorImpl* XtensorBackend::ones(const Shape& shape, DType dtype)
{
	return new XtensorTensorImpl(run<Xarr>(dtype, [&shape](auto dummy){return xt::eval(xt::ones<decltype(dummy)>(shape));}), this);
}

TensorImpl* XtensorBackend::rand(float lEdge, float rEdge, const Shape& shape)
{
	auto s = as<xt::xarray<float>::shape_type>(shape);
	return createTensor(std::move(xt::random::rand<float>(s, lEdge, rEdge)));
}

TensorImpl* XtensorBackend::normal(float mean, float stddev, const Shape& shape)
{
	auto s = as<xt::xarray<float>::shape_type>(shape);
	return createTensor(std::move(xt::random::rand<float>(s, mean, stddev)));
}
