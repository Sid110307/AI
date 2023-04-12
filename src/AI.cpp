#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <chrono>

#include <torch/torch.h>

#include "include/datasets.h"
#include "include/utils.h"

const int MAX_LENGTH = 20;
const double LEARNING_RATE = 0.01;
const int EPOCHS = 100;

struct Chatbot : torch::nn::Module
{
public:
	explicit Chatbot(int vocabSize, int embeddingDim = 50, int hiddenDim = 50)
	{
		embedding = register_module("embedding", torch::nn::Embedding(vocabSize, embeddingDim));
		lstm = register_module("lstm", torch::nn::LSTM(embeddingDim, hiddenDim));
		fc = register_module("fc", torch::nn::Linear(hiddenDim, vocabSize));
	}

	torch::Tensor forward(const torch::Tensor &input)
	{
		auto embedded = embedding->forward(input);
		auto output = std::get<0>(lstm->forward(embedded));
		auto decoded = fc->forward(output);

		return decoded;
	}

private:
	torch::nn::Embedding embedding = nullptr;
	torch::nn::LSTM lstm = nullptr;
	torch::nn::Linear fc = nullptr;
};

void testChatbot(const std::string &inputSequence, const std::map<std::string, int> &wordToIndex,
				 const std::map<int, std::string> &indexToWord, Chatbot &chatbot)
{
	auto start = printDebug("Testing the chatbot by passing it a sample input sequence...");

	std::vector<int> inputSequenceIndices;
	std::istringstream iss(inputSequence);

	std::string token;
	while (std::getline(iss, token, ' ')) inputSequenceIndices.push_back(wordToIndex.at(token));

	auto inputSequenceTensor = torch::tensor(inputSequenceIndices).unsqueeze(0);
	auto output = chatbot.forward(inputSequenceTensor);

	auto outputData = output.argmax(2).squeeze(0).data_ptr<int>();
	std::vector<int> outputIndices(outputData, outputData + output.size(1));

	std::cout << "Input: " << inputSequence << std::endl;
	std::cout << "Output: ";

	for (const auto &index: outputIndices) std::cout << indexToWord.at(index) << " ";
	std::cout << std::endl;

	printDebug("Done in " + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(
			std::chrono::high_resolution_clock::now() - start).count()) + "ms.", false);
}

int main()
{
	CornellDataset cornellDataset;
	cornellDataset.validate();

	std::vector<std::string> cleanedLines = cornellDataset.preprocess(cornellDataset.load());
	auto start = printDebug("Generating word-to-index and index-to-word dictionaries...");

	std::map<std::string, int> wordToIndex;
	std::map<int, std::string> indexToWord;

	for (const auto &line: cleanedLines)
	{
		std::istringstream iss(line);
		std::string token;

		while (std::getline(iss, token, ' '))
		{
			if (wordToIndex.find(token) == wordToIndex.end())
			{
				int index = (int) wordToIndex.size();

				wordToIndex[token] = index;
				indexToWord[index] = token;
			}
		}
	}

	printDebug("Done in " + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(
			std::chrono::high_resolution_clock::now() - start).count()) + "ms.", false);
	start = printDebug("Encoding the cleaned conversations into sequences of indices...");

	std::vector<std::vector<int>> encodedConversations;
	for (const auto &conversation: cleanedLines)
	{
		std::vector<int> encodedConversation;
		std::istringstream iss(conversation);
		std::string token;

		while (std::getline(iss, token, ' ')) encodedConversation.push_back(wordToIndex[token]);
		encodedConversations.push_back(encodedConversation);
	}

	printDebug("Done in " + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(
			std::chrono::high_resolution_clock::now() - start).count()) + "ms.", false);
	start = printDebug("Creating training pairs of input-output sequences...");

	std::vector<std::pair<std::vector<int>, int >> pairs;
	for (const auto &conversation: encodedConversations)
	{
		for (int i = 1; i < (int) conversation.size(); i++)
		{
			std::vector<int> inputSequence(conversation.begin(), conversation.begin() + i);
			int outputSequence = conversation[i];

			pairs.emplace_back(inputSequence, outputSequence);
		}
	}

	printDebug("Done in " + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(
			std::chrono::high_resolution_clock::now() - start).count()) + "ms.", false);
	start = printDebug("Padding the input and output sequences to " + std::to_string(MAX_LENGTH) + " tokens...");

	std::vector<std::pair<std::vector<int>, int >> paddedPairs;
	for (const auto &pair: pairs)
	{
		if (pair.first.size() <= MAX_LENGTH)
		{
			std::vector<int> inputSequence = pair.first;
			inputSequence.insert(inputSequence.end(), MAX_LENGTH - inputSequence.size(), 0);

			paddedPairs.emplace_back(inputSequence, pair.second);
		}
	}

	printDebug("Done in " + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(
			std::chrono::high_resolution_clock::now() - start).count()) + "ms.", false);
	start = printDebug("Creating the training and validation datasets...");

	std::vector<std::pair<std::vector<int>, int >> trainingPairs;
	std::vector<std::pair<std::vector<int>, int >> validationPairs;

	for (int i = 0; i < (int) paddedPairs.size(); i++)
	{
		if (i % 10 == 0) validationPairs.push_back(paddedPairs[i]);
		else trainingPairs.push_back(paddedPairs[i]);
	}

	printDebug("Done in " + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(
			std::chrono::high_resolution_clock::now() - start).count()) + "ms.", false);
	start = printDebug("Converting the pairs into PyTorch tensors...");

	std::vector<torch::Tensor> trainingInputs;
	std::vector<torch::Tensor> trainingTargets;
	std::vector<torch::Tensor> validationInputs;
	std::vector<torch::Tensor> validationTargets;

	for (const auto &pair: trainingPairs)
	{
		trainingInputs.push_back(torch::tensor(pair.first, torch::kInt64));
		trainingTargets.push_back(torch::tensor(pair.second, torch::kInt64));
	}

	for (const auto &pair: validationPairs)
	{
		validationInputs.push_back(torch::tensor(pair.first, torch::kInt64));
		validationTargets.push_back(torch::tensor(pair.second, torch::kInt64));
	}

	printDebug("Done in " + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(
			std::chrono::high_resolution_clock::now() - start).count()) + "ms.", false);
	start = printDebug("Initializing the chatbot model...");

	int vocabularySize = (int) wordToIndex.size();
	auto* chatbot = new Chatbot(vocabularySize);

	printDebug("Done in " + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(
			std::chrono::high_resolution_clock::now() - start).count()) + "ms.", false);
	start = printDebug("Defining the loss function and the optimizer...");

	torch::nn::CrossEntropyLoss loss;
	torch::optim::Adam optimizer(chatbot->parameters(), torch::optim::AdamOptions(LEARNING_RATE));

	printDebug("Done in " + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(
			std::chrono::high_resolution_clock::now() - start).count()) + "ms.", false);
	start = printDebug("Training the chatbot model...");

	std::cout << std::endl;
	for (int epoch = 0; epoch < EPOCHS; epoch++)
	{
		torch::Tensor totalLoss = torch::zeros({1});

		for (int i = 0; i < (int) trainingInputs.size(); i++)
		{
			optimizer.zero_grad();

			torch::Tensor outputs = chatbot->forward(trainingInputs[i].view({-1, 1})).view(trainingTargets[i].sizes());
			// FIXME: shape '[]' is invalid for input of size 3291760. Possibly trainingTargets is empty?
			torch::Tensor lossValue = loss(outputs, trainingTargets[i].view({-1}));

			lossValue.backward();
			optimizer.step();

			totalLoss += lossValue;
		}

		std::cout << "Epoch " << epoch + 1 << "/" << EPOCHS << " - Loss: " << totalLoss.item<float>() << std::endl;
	}

	printDebug("Done in " + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(
			std::chrono::high_resolution_clock::now() - start).count()) + "ms.", false);
	start = printDebug("Evaluating the chatbot on the validation set...");

	int correct = 0;
	for (int i = 0; i < (int) validationInputs.size(); i++)
	{
		torch::Tensor outputs = chatbot->forward(validationInputs[i].view({-1, 1}));
		torch::Tensor predicted = outputs.argmax(1);

		if (predicted.item<int>() == validationTargets[i].item<int>()) correct++;
	}

	std::cout << std::endl;
	printDebug("Accuracy: " + std::to_string((float) correct / (float) validationInputs.size() * 100.0f) + "%", false);

	printDebug("Done in " + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(
			std::chrono::high_resolution_clock::now() - start).count()) + "ms.", false);
	start = printDebug("Saving the chatbot model...");

	// TODO: Save the chatbot model.
	printDebug("Done in " + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(
			std::chrono::high_resolution_clock::now() - start).count()) + "ms.", false);

	testChatbot("how are you doing today", wordToIndex, indexToWord, *chatbot);
	return EXIT_SUCCESS;
}