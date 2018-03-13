#include <nlohmann/json.hpp>

#include <Athena/Utils/Archive.hpp>
#include <Athena/Utils/Error.hpp>
using namespace At;

template <typename T>
nlohmann::json makeChild(std::string type, const T& val)
{
	nlohmann::json j;
	j["__type"] = type;
	j["__value"] = val;
	return j;
}

void Archiver::save(const BoxedValues& states, std::string path)
{
	nlohmann::json j;
	boxToJson(j, states);

	std::ofstream out(path);
	if(out.good() == false)
		throw AtError("Cannot write to file " + path);
	out << j.dump(2);
	out.close();
}

void Archiver::boxToJson(nlohmann::json& j, const BoxedValues& states)
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
		else if(auto ptr = box_cast<float>(elem); ptr != nullptr)
		{
			j[key] = makeChild("Float32", ptr->value());
		}
		else
		{
			throw AtError("Not supported type");
		}
	}
}


void Archiver::jsonToBox(const nlohmann::json& j, BoxedValues& states)
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
			else if(type == "Float32")
			{
				states.set<float>(key, elem["__value"]);
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

BoxedValues Archiver::load(std::string path)
{
	std::ifstream in(path);
	if(in.good() == false)
		throw AtError("Can't read file " + path);
	nlohmann::json j;
	in >> j;
	in.close();

	BoxedValues vals;
	jsonToBox(j, vals);
	return vals;
}