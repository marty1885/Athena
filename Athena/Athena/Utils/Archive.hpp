#pragma once

#include <Athena/Utils/BoxedValue.hpp>

#include <string>

namespace At
{

class Archiver
{
public:

	static void save(const BoxedValues& states, std::string path);
	static BoxedValues load(std::string path);
};


}
