#pragma once

#include <iostream>
#include <vector>

std::chrono::high_resolution_clock::time_point printDebug(const std::string &arg, bool ret = true)
{
	if (std::getenv("DEBUG"))
	{
		if (ret)
		{
			std::cout << "DEBUG: " << arg << " " << std::flush;
			return std::chrono::high_resolution_clock::now();
		} else std::cout << arg << std::endl;
	}

	return {};
}

std::vector<std::string> split(const std::string &s, const char &c)
{
	std::string buff;
	std::vector<std::string> v;

	for (auto n: s)
	{
		if (n != c) buff += n;
		else if (n == c && !buff.empty())
		{
			v.push_back(buff);
			buff = "";
		}
	}

	if (!buff.empty()) v.push_back(buff);
	return v;
}

std::string replace(std::string str, const std::string &from, const std::string &to)
{
	size_t start_pos = 0;
	while ((start_pos = str.find(from, start_pos)) != std::string::npos)
	{
		str.replace(start_pos, from.length(), to);
		start_pos += to.length();
	}

	return str;
}
