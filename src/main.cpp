#include <iostream>
#include <string>
#include <torch/nn/modules/loss.h>
#include <vector>
#include <unordered_map>
#include <chrono>

#include <torch/torch.h>

#include "include/datasets.h"
#include "include/utils.h"

constexpr int MAX_LENGTH = 50;
constexpr double LEARNING_RATE = 0.1;
constexpr int EPOCHS = 1000;

struct Chatbot : torch::nn::Module
{
public:
    explicit Chatbot(int vocabularySize)
    {
        encoder = torch::nn::Embedding(torch::nn::EmbeddingOptions(vocabularySize, 256));
        decoder = torch::nn::Embedding(torch::nn::EmbeddingOptions(vocabularySize, 256));
        lstm = torch::nn::LSTM(torch::nn::LSTMOptions(256, 1024).num_layers(2));
        linear = torch::nn::Linear(torch::nn::LinearOptions(1024, vocabularySize));
    }

    torch::Tensor forward(const torch::Tensor &inputSequence)
    {
        auto encoderOutput = encoder->forward(inputSequence);
        auto lstmOutput = lstm->forward(encoderOutput);
        auto decoderOutput = decoder->forward(inputSequence);
        auto output = linear->forward(decoderOutput);

        return output;
    }

private:
    torch::nn::Embedding encoder = nullptr, decoder = nullptr;
    torch::nn::LSTM lstm = nullptr;
    torch::nn::Linear linear = nullptr;
};

void testChatbot(const std::string &inputSequence, const std::map<std::string, int> &wordToIndex,
                 const std::map<int, std::string> &indexToWord, Chatbot &chatbot)
{
    auto start = printDebug("Testing the chatbot by passing it an input sequence...");

    std::vector<int> inputSequenceIndices;
    std::istringstream iss(inputSequence);

    std::string token;
    while (std::getline(iss, token, ' ')) inputSequenceIndices.push_back(wordToIndex.at(token));

    auto inputSequenceTensor = torch::tensor(inputSequenceIndices).unsqueeze(0);
    auto output = chatbot.forward(inputSequenceTensor);

    auto outputData = output.argmax(2).squeeze(0).data_ptr<int>();
    std::vector<int> outputIndices(outputData, outputData + output.size(1));

    std::cout << "Input: " << inputSequence << "\nOutput: ";

    for (const auto &index: outputIndices) std::cout << indexToWord.at(index) << " ";
    std::cout << std::endl;

    printDebug("Done in " + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - start).count()) + "ms.", false);
}

int main()
{
    CornellDataset cornellDataset;

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
                int index = static_cast<int>(wordToIndex.size());

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

    std::vector<std::pair<std::vector<int>, int>> pairs;
    for (const auto &conversation: encodedConversations)
        for (int i = 1; i < static_cast<int>(conversation.size()); ++i)
        {
            std::vector<int> inputSequence(conversation.begin(), conversation.begin() + i);
            int outputSequence = conversation[i];

            pairs.emplace_back(inputSequence, outputSequence);
        }

    printDebug("Done in " + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - start).count()) + "ms.", false);
    start = printDebug("Padding the input and output sequences to " + std::to_string(MAX_LENGTH) + " token" +
                       (MAX_LENGTH == 1 ? "" : "s") + "...");

    std::vector<std::pair<std::vector<int>, int>> paddedPairs;
    for (const auto &pair: pairs)
    {
        std::vector<int> inputSequence = pair.first;
        int outputSequence = pair.second;

        while (inputSequence.size() < MAX_LENGTH) inputSequence.push_back(0);
        paddedPairs.emplace_back(inputSequence, outputSequence);
    }

    printDebug("Done in " + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - start).count()) + "ms.", false);
    start = printDebug("Creating the training and validation datasets...");

    std::vector<std::pair<std::vector<int>, int>> trainingPairs, validationPairs;

    for (int i = 0; i < static_cast<int>(paddedPairs.size()); ++i)
    {
        if (i % 10 == 0) validationPairs.push_back(paddedPairs[i]);
        else trainingPairs.push_back(paddedPairs[i]);
    }

    printDebug("Done in " + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - start).count()) + "ms.", false);
    start = printDebug("Converting the pairs into PyTorch tensors...");

    std::vector<torch::Tensor> trainingInputs, trainingTargets, validationInputs, validationTargets;

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
    start = printDebug("Defining the loss function and the optimizer...");

    int vocabularySize = static_cast<int>(wordToIndex.size());
    auto* chatbot = new Chatbot(vocabularySize);

    torch::nn::CrossEntropyLoss loss;
    torch::optim::Adam optimizer(chatbot->parameters(), torch::optim::AdamOptions(LEARNING_RATE));

    printDebug("Done in " + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - start).count()) + "ms.", false);
    start = printDebug("Training the chatbot model...");

    std::cout << std::endl;
    for (int epoch = 0; epoch < EPOCHS; ++epoch)
    {
        int correct = 0;
        float totalLoss = 0.0f;

        for (int i = 0; i < static_cast<int>(trainingInputs.size()); ++i)
        {
            torch::Tensor outputs = chatbot->forward(trainingInputs[i].view({-1, 1}));
            torch::Tensor predicted = outputs.argmax(1);

            if (predicted.allclose(trainingTargets[i].view({-1}))) correct++;

            torch::Tensor lossValue = loss(outputs, trainingTargets[i].view({-1}));
            totalLoss += lossValue.item<float>();

            optimizer.zero_grad();
            lossValue.backward();
            optimizer.step();
        }

        std::cout << "\rEpoch " << epoch + 1 << "/" << EPOCHS << " - Loss: " << totalLoss << " - Accuracy: "
                  << static_cast<float>(correct) / static_cast<float>(trainingInputs.size()) * 100.0f << "%\033[2K\r"
                  << std::flush;
    }

    printDebug("Done in " + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - start).count()) + "ms.", false);
    start = printDebug("Evaluating the chatbot on the validation set...");

    int correct = 0;
    for (int i = 0; i < static_cast<int>(validationInputs.size()); ++i)
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
