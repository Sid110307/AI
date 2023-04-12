#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <utility>
#include <vector>
#include <regex>
#include <filesystem>

#include <curl/curl.h>
#include <zip.h>

#include "threadPool.h"
#include "utils.h"

const std::string DATASET_ROOT = std::filesystem::current_path().string() + "/" +
								 (std::getenv("DATASET_ROOT") ? std::getenv("DATASET_ROOT") : ".");

class Dataset
{
public:
	virtual ~Dataset() = default;

	virtual void validate() = 0;
	virtual std::vector<std::string> load() = 0;
	virtual std::vector<std::string> preprocess(const std::vector<std::string> &) = 0;
};

class CornellDataset : public Dataset
{
public:
	void validate() override
	{
		auto writeCallback = [](char* contents, size_t size, size_t nmemb, FILE* buffer)
		{
			return fwrite(contents, size, nmemb, buffer);
		};

		std::chrono::high_resolution_clock::time_point start = printDebug(
				"Checking if " + DATASET_ROOT + "/movie_lines.txt is available...");

		// FIXME: curl_easy_perform() segfault.
		if (!std::ifstream(DATASET_ROOT + "/movie_lines.txt"))
		{
			std::cout << "\nThe file " << DATASET_ROOT << "/movie_lines.txt could not be resolved. "
					  << "Downloading from the Cornell Movie-Dialogs Corpus..." << std::endl;

			CURL* curl = curl_easy_init();

			if (curl)
			{
				FILE* file = fopen("cornell_movie_dialogs_corpus.zip", "wb");
				curl_easy_setopt(curl, CURLOPT_URL,
								 "https://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip");
				curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, false);
				curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, false);
				curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, true);
				curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writeCallback);
				curl_easy_setopt(curl, CURLOPT_WRITEDATA, &file);

				CURLcode res = curl_easy_perform(curl);
				if (res != CURLE_OK)
					std::cerr << "curl_easy_perform() failed: " << curl_easy_strerror(res) << std::endl;

				curl_easy_cleanup(curl);
				fclose(file);
			}

			std::cout << "Download complete. Extracting..." << std::endl;
			struct zip* archive = zip_open("cornell_movie_dialogs_corpus.zip", 0, nullptr);

			if (archive)
			{
				for (int i = 0; i < (int) zip_get_num_entries(archive, 0); i++)
				{
					struct zip_stat stat = {};
					zip_stat_init(&stat);
					zip_stat_index(archive, i, 0, &stat);

					std::string name = stat.name;
					if (name.find("movie_lines.txt") != std::string::npos)
					{
						struct zip_file* file = zip_fopen_index(archive, i, 0);
						if (file)
						{
							std::ofstream out(DATASET_ROOT + "/movie_lines.txt", std::ios::binary);
							char buffer[1024];
							int bytesRead;

							while ((bytesRead = (int) zip_fread(file, buffer, 1024)) > 0) out.write(buffer, bytesRead);
							out.close();
							zip_fclose(file);
						}
					}
				}

				zip_close(archive);
			}

			std::cout << "Extraction complete." << std::endl;

			std::filesystem::rename(DATASET_ROOT + "/cornell movie-dialogs corpus/movie_lines.txt",
									DATASET_ROOT + "/movie_lines.txt");
			std::filesystem::remove("cornell_movie_dialogs_corpus.zip");
			std::filesystem::remove_all(DATASET_ROOT + "/cornell movie-dialogs corpus");
			std::filesystem::remove_all(DATASET_ROOT + "/__MACOSX");
		}

		printDebug("Done in " + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(
				std::chrono::high_resolution_clock::now() - start).count()) + "ms.", false);
	}

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
				{"'cause", "because"}
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

		ThreadPool pool;
		for (const auto &line: conversations)
		{
			printDebug("Preprocessing the line: " + line, false);
			pool.queueJob([&]() { _lines.push_back(preprocessLine(line)); });
		}

		pool.stop();
		printDebug("Done in " + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(
				std::chrono::high_resolution_clock::now() - start).count()) + "ms.", false);

		return _lines;
	}
};

// TODO: Add more datasets
