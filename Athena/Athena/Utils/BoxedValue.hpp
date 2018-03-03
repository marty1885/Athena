#pragma once

#include <map>
#include <typeinfo>

namespace At
{

struct BoxedValueBase
{
	virtual ~BoxedValueBase(){}
};

template<typename T>
struct BoxedValue : public BoxedValueBase
{
	BoxedValue(){}
	BoxedValue(const T& value):value_(value){}
	T& value() {return value_;}
	const T& value() const {return value_;}
	T value_;
};

class BoxedValues : public std::map<std::string, BoxedValueBase*>
{
public:
	virtual ~BoxedValues()
	{
		for(auto& e : *this)
			delete e.second;
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
T& boxed_cast(BoxedValueBase* ptr)
{
	if(ptr == nullptr)
		throw AtError("Variable does not exist");
	BoxedValue<T>* res = dynamic_cast<BoxedValue<T>*>(ptr);
	if(res == nullptr)
		throw AtError("Variable isn't typed as " + std::string(typeid(T).name()));
	return res->value();
}

}