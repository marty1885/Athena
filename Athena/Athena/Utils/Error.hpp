#pragma once

#include <exception>
#include <string>

namespace At
{
class AtError : public std::exception {
public:
	explicit AtError(const std::string &msg) : msg_(msg) {}
	const char *what() const throw() override { return msg_.c_str(); }

private:
	std::string msg_;
};

#define AtAssertWithMessage(expression, msg) do{if((expression) == false) throw At::AtError(msg);}while(0)
#define AtAssertNoMessage(expression) do{if((expression) == false) throw At::AtError(#expression);}while(0)

#define GetAtAssrtyMacro(_1,_2,NAME,...) NAME
#define AtAssert(...) GetAtAssrtyMacro(__VA_ARGS__ ,AtAssertWithMessage, AtAssertNoMessage)(__VA_ARGS__)
}