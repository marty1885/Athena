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
}