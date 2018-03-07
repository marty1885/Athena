#pragma once

#include <Athena/Utils/BoxedValue.hpp>
#include <Athena/Utils/Shape.hpp>
#include <Athena/Utils/Error.hpp>

#include <string>
#include <iostream>

#include <fstream>

#include <nlohmann/json_fwd.hpp>

namespace At
{

class Archiver
{
public:

	static void save(const BoxedValues& states, std::string path);
	static void boxToJson(nlohmann::json& j, const BoxedValues& states);
	static void jsonToBox(const nlohmann::json& j, BoxedValues& states);
	static BoxedValues load(std::string path);
};


}