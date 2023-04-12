#pragma once

#include <vector>
#include <functional>
#include <thread>
#include <queue>
#include <condition_variable>

class ThreadPool
{
public:
	void start();
	void queueJob(const std::function<void()> &);
	void stop();
	void busy();

private:
	void threadLoop();
	bool shouldTerminate = false;

	std::mutex queueMutex;
	std::condition_variable mutexCondition;

	std::vector<std::thread> threads;
	std::queue<std::function<void()>> jobs;
};
