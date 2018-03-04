#pragma once

#include <Athena/Utils/BoxedValue.hpp>
#include <Athena/Utils/Shape.hpp>
#include <Athena/Utils/Error.hpp>

#include <string>
#include <iostream>

#include <nlohmann/json.hpp>
#include <fstream>

namespace At
{

template <typename T>
nlohmann::json makeChild(std::string type, const T& val)
{
	nlohmann::json j;
	j["__type"] = type;
	j["__value"] = val;
	return j;
}

void boxToJson(nlohmann::json& j, const BoxedValues& states)
{
	using json = nlohmann::json;
	for(const auto& [key, elem] : states)
	{
		if(auto ptr = box_cast<BoxedValues>(elem); ptr != nullptr)
		{
			json child;
			boxToJson(child, ptr->value());
			j[key] = child;
		}
		else if(auto ptr = box_cast<std::vector<float>>(elem); ptr != nullptr)
		{
			j[key] = makeChild("FloatVector", ptr->value());
		}
		else if(auto ptr = box_cast<Shape>(elem); ptr != nullptr)
		{
			j[key] = makeChild("Shape", ptr->value());
		}
		else if(auto ptr = box_cast<std::string>(elem); ptr != nullptr)
		{
			j[key] = ptr->value();
		}
		else
		{
			throw AtError("Not supported type");
		}
	}
}

bool save(const BoxedValues& states, std::string path)
{
	nlohmann::json j;
	boxToJson(j, states);

	std::ofstream out(path);
	out << j.dump();
	//std::cout << j.dump(4) << std::endl;
	out.close();
	return true;
}

void jsonToBox(const nlohmann::json& j, BoxedValues& states)
{
	for (auto it=j.begin(); it!=j.end(); ++it)
	{
		const nlohmann::json& elem = it.value();
		std::string key = it.key();
		if(elem.is_object() == true)
		{
			std::string type = elem["__type"];
			if(type == "FloatVector")
			{
				states.set<std::vector<float>>(key, elem["__value"]);
			}
			else if(type == "Shape")
			{
				states.set<Shape>(key, elem["__value"]);
			}
			else
			{
				BoxedValues params;
				jsonToBox(elem, params);
				states.set<BoxedValues>(key, params);
			}
		}
		else if(elem.is_string())
		{
			states.set<std::string>(key, elem);
		}
	}
}

BoxedValues load(std::string path)
{
	std::ifstream in(path);
	nlohmann::json j;
	in >> j;
	in.close();

	BoxedValues vals;
	jsonToBox(j, vals);
	return vals;
}

}