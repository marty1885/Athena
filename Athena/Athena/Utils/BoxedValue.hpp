#pragma once

#include <map>
#include <typeinfo>
#include <memory>

namespace At
{

struct BoxedValueBase
{
	virtual ~BoxedValueBase(){}
	virtual BoxedValueBase* allocateCopy() const = 0;
};

template<typename T>
struct BoxedValue : public BoxedValueBase
{
	BoxedValue() = default;
	BoxedValue(const T& value):value_(value){}
	T& value() {return value_;}
	const T& value() const {return value_;}
	virtual BoxedValueBase* allocateCopy() const override {return new BoxedValue<T>(value_);};
	T value_;
};

//TODO: Should I use shared_ptr instead of allocating and coping everything?
class BoxedValues : public std::map<std::string, BoxedValueBase*>
{
public:
	BoxedValues() = default;

	virtual ~BoxedValues()
	{
		for(auto& e : *this)
			delete e.second;
	}

	BoxedValues(const BoxedValues& other)
		: std::map<std::string, BoxedValueBase*>(other)
	{
		//Overrite all the keys
		for(const auto& [key, ptr] : other)
			operator[](key) = ptr->allocateCopy();
	}

	template<typename T>
	inline void set(const std::string& name, const T& value)
	{
		operator[](name) = new BoxedValue<T>(value);
	}

	template<typename T>
	const T& get(const std::string& name) const
	{
		auto it = find(name);
		if(it == end())
			throw AtError("Cannot find variable \"" + name + "\"");
		BoxedValue<T>* ptr = dynamic_cast<BoxedValue<T>*>(it->second);
		if(ptr == nullptr)
			throw AtError("Variable \"" + name + "\" does not have type " + typeid(T).name());
		return ptr->value();
	}
};

template <typename T>
BoxedValue<T>* box_cast(BoxedValueBase* ptr)
{
	return dynamic_cast<BoxedValue<T>*>(ptr);
}

template <typename T>
const BoxedValue<T>* box_cast(const BoxedValueBase* ptr)
{
	return dynamic_cast<const BoxedValue<T>*>(ptr);
}

template <typename T>
T& boxed_cast(BoxedValueBase* ptr)
{
	if(ptr == nullptr)
		throw AtError("Variable does not exist");
	BoxedValue<T>* res = dynamic_cast<BoxedValue<T>*>(ptr);
	if(res == nullptr)
		throw AtError("Variable isn't typed as " + std::string(typeid(T).name()));
	return res->value();
}

template <typename T>
const T& boxed_cast(const BoxedValueBase* ptr)
{
	if(ptr == nullptr)
		throw AtError("Variable does not exist");
	const BoxedValue<T>* res = dynamic_cast<const BoxedValue<T>*>(ptr);
	if(res == nullptr)
		throw AtError("Variable isn't typed as " + std::string(typeid(T).name()));
	return res->value();
}

}