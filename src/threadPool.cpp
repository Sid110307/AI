#include "include/threadPool.h"

void ThreadPool::start()
{
	const unsigned int numberOfThreads = std::thread::hardware_concurrency();
	threads.resize(numberOfThreads);

	for (unsigned int i = 0; i < numberOfThreads; i++) threads.at(i) = std::thread(&ThreadPool::threadLoop, this);
}

void ThreadPool::queueJob(const std::function<void()> &job)
{
	{
		std::unique_lock<std::mutex> lock(queueMutex);
		jobs.push(job);
	}

	mutexCondition.notify_one();
}

void ThreadPool::stop()
{
	{
		std::unique_lock<std::mutex> lock(queueMutex);
		shouldTerminate = true;
	}
	mutexCondition.notify_all();

	for (auto &thread: threads) thread.join();
	threads.clear();
}

void ThreadPool::busy()
{
	while (true)
	{
		std::unique_lock<std::mutex> lock(queueMutex);
		if (jobs.empty()) break;
	}
}

void ThreadPool::threadLoop()
{
	while (true)
	{
		std::function<void()> job;
		{
			std::unique_lock<std::mutex> lock(queueMutex);
			mutexCondition.wait(lock, [this] { return !jobs.empty() || shouldTerminate; });

			if (shouldTerminate) return;
			job = jobs.front();
			jobs.pop();
		}

		job();
	}
}
