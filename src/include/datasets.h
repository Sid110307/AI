#pragma once

#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <utility>
#include <regex>

#include "utils.h"

const std::string DATASET_ROOT = std::filesystem::current_path().string() + "/" +
                                 (std::getenv("DATASET_ROOT") ? std::getenv("DATASET_ROOT") : ".");

class Dataset
{
public:
    virtual ~Dataset() = default;

    virtual std::vector<std::string> load() = 0;
    virtual std::vector<std::string> preprocess(const std::vector<std::string> &) = 0;
};

class CornellDataset : public Dataset
{
public:
    std::vector<std::string> load() override
    {
        auto start = printDebug("Loading the Cornell movie dialogs dataset...");
        std::vector<std::string> lines;

        std::ifstream file(DATASET_ROOT + "/movie_lines.txt");
        if (!file.is_open())
        {
            std::cerr << "The file " << DATASET_ROOT << "/movie_lines.txt could not be opened." << std::endl;
            exit(EXIT_FAILURE);
        }

        std::string line;
        while (std::getline(file, line)) lines.push_back(line);

        file.close();
        printDebug("Done in " + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - start).count()) + "ms.", false);

        return lines;
    }

    std::vector<std::string> preprocess(const std::vector<std::string> &lines) override
    {
        std::unordered_map<std::string, std::string> replacements = {
                {"i'm",    "i am"},
                {"'s",     " is"},
                {"'ll",    " will"},
                {"'ve",    " have"},
                {"'re",    " are"},
                {"'d",     " not"},
                {"'re",    " are"},
                {"n't",    " not"},
                {"'bout",  "about"},
                {"'til",   "until"},
                {"'cause", "because"},
        };

        auto preprocessLine = [&](const std::string &_line)
        {
            std::string text = _line;
            for (const auto &[key, value]: replacements)
                text = replace(text, key, value);

            std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) { return std::tolower(c); });
            return text;
        };

        auto start = printDebug("Preprocessing the lines...");
        std::vector<std::string> conversations;

        for (const auto &line: lines)
        {
            auto parts = split(line, '$');
            parts[4].erase(std::remove(parts[4].begin(), parts[4].end(), '+'), parts[4].end());

            if (parts[4].find("+++$+++") != std::string::npos)
                for (const auto &conversation: split(parts[4], '+'))conversations.push_back(conversation);
            else conversations.push_back(parts[4]);
        }

        std::vector<std::string> _lines;
        for (const auto &line: conversations) _lines.push_back(preprocessLine(line));

        printDebug("Done in " + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - start).count()) + "ms.", false);

        return _lines;
    }
};

class GSMDataset : public Dataset
{
public:
    std::vector<std::string> load() override
    {
        auto start = printDebug("Loading the OpenAI GSM dataset...");
        std::vector<std::string> lines;

        std::ifstream file(DATASET_ROOT + "/train_socratic.jsonl");
        if (!file.is_open())
        {
            std::cerr << "The file " << DATASET_ROOT << "/train_socratic.jsonl could not be opened." << std::endl;
            exit(EXIT_FAILURE);
        }

        std::string line;
        while (std::getline(file, line)) lines.push_back(line);

        file.close();
        printDebug("Done in " + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - start).count()) + "ms.", false);

        return lines;
    }

    std::vector<std::string> preprocess(const std::vector<std::string> &lines) override
    {
        auto preprocessLine = [&](const std::string &_line)
        {
            std::string text = _line;
            std::smatch match;

            while (std::regex_search(text, match, std::regex(R"(\*\* (.+?) \*\*)")))
            {
                std::string question = match[1].str();
                std::string answer = question.substr(question.find("#### ") + 5);
                question = question.substr(0, question.find("#### "));
                text = replace(text, match[0].str(), question + " " + answer);
            }

            return text;
        };

        auto start = printDebug("Preprocessing the lines...");
        std::vector<std::string> conversations;

        for (const auto &line: lines)
        {
            std::string text = preprocessLine(line);
            text = text.substr(text.find("#### ") + 5);
            conversations.push_back(text);
        }

        std::vector<std::string> _lines;
        for (const auto &line: conversations) _lines.push_back(preprocessLine(line));

        printDebug("Done in " + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - start).count()) + "ms.", false);

        return _lines;
    }
};
