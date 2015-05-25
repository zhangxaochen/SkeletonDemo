#ifndef __logCat_h_
#define __logCat_h_

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <list>
#include <vector>
#include <time.h>
#include <stdlib.h>

using namespace std;

#define LOG_LEVEL_ALL           0
#define LOG_LEVEL_IMPORT        1
#define LOG_LEVEL_ERROR         2
#define LOG_LEVEL_WARNING       3

#define MSG(msg) LogCat::getInstancePtr()->out(msg);
#define PLAIN_MSG(msg, id) LogCat::getInstancePtr()->out(msg, id);
#define ERROR_MSG(msg, id) LogCat::getInstancePtr()->out(msg, id, LOG_LEVEL_ERROR);
#define IMPORT_MSG(msg, id) LogCat::getInstancePtr()->out(msg, id, LOG_LEVEL_IMPORT);
#define WARNING_MSG(msg, id) LogCat::getInstancePtr()->out(msg, id, LOG_LEVEL_WARNING);

class LogCatListener;
class LogCat
{
public:
	static LogCat* getInstancePtr();
	static LogCat& getInstance();

	~LogCat();

	void out(const char* msg, int id = -1, int level = LOG_LEVEL_ALL);

	void out(string msg, int id = -1, int level = LOG_LEVEL_ALL){
		out(msg.data(), id, level);
	}

	void addProcessBar(const char* title, int processTotal);
	void setProcessBar(float percent);
	void removeProcessBar();

	void addListener(LogCatListener* l){
		_listeners.push_back(l);
	}

	void removeListener(LogCatListener* l){
		_listeners.remove(l);
	}

	int registerModule(string name){
		_moduleNames.push_back(name);
		return _moduleNames.size()-1;
	}

	template <class T>
	static inline string to_string (const T& t)
	{
		stringstream ss;
		ss << t;
		return ss.str();
	}

	static inline int to_int(string s)
	{
		return atoi(s.data());
	}

	static inline float to_float(string s)
	{
		return atof(s.data());
	}

	static inline std::string getTimeString()
	{
		time_t curtime=time(0);
		tm tim = *localtime(&curtime);
		std::string timeString;
		timeString.append(to_string(tim.tm_hour)+":"+
			to_string(tim.tm_min)+":"+to_string(tim.tm_sec));
		return timeString;
	}

private:
	LogCat();

	string findModuleName(int id){
		if(id < _moduleNames.size())
			return _moduleNames[id];
		else return string();
	}

private:
	static LogCat* _instance;

	fstream _logFile;

	list<LogCatListener*> _listeners;

	vector<string> _moduleNames;

	int _process;
	int _processTotal;

};

class LogCatListener
{
public:
	LogCatListener(){}
	~LogCatListener(){}

	virtual void newMsg(string msg){}
};

#endif
