#include "logCat.h"
#include "Macros.h"

#define PRINT_SIELENT

LogCat* LogCat::_instance = 0;

LogCat* LogCat::getInstancePtr(){
	if(_instance == 0)
		_instance = new LogCat();
	return _instance;
}

LogCat& LogCat::getInstance(){
	if(_instance == 0)
		_instance = new LogCat();
	return *_instance;
}

LogCat::LogCat()
{
#if defined(ANDROID)
#define LOGCAT_PATH "/data/data/com.motioninteractive.zte/bpr.log"
#else 
#define LOGCAT_PATH "bpr.log"
#endif

#ifndef LINC_DEBUG
	_logFile.open(LOGCAT_PATH, ios::out | ios::trunc);
#else
	_logFile.open(LOGCAT_PATH, ios::out | ios::trunc);
#endif
	if(!_logFile.is_open()){
		CapgPrintf("%s\n", "!!!Error to open log file");
	}else{
		_logFile.clear();
		_logFile << "Body Part Recognition Log" << endl << endl;
		_logFile.flush();
	}
	_process = 0;
}

LogCat::~LogCat()
{
	_logFile.flush();
	_logFile.close();

	_moduleNames.clear();
	_listeners.clear();
}

void LogCat::out(const char* msg, int id, int level)
{
	if(!_logFile.is_open()) return;

	string outString;

	switch(level){
		case LOG_LEVEL_ALL:
#ifdef PRINT_SIELENT
			return;
#endif
			break;
		case LOG_LEVEL_IMPORT:
			outString.append("<I>");
			break;
		case LOG_LEVEL_ERROR:
			outString.append("<ERROR>");
			break;
		case LOG_LEVEL_WARNING:
			outString.append("<W>");
			break;
	}

	outString.append(getTimeString());

	if(id >=0 ){
		string moduleName = findModuleName(id);
		if(!moduleName.empty()){
			outString.append(" #");
			outString.append(moduleName);
		}
	}

	outString.append("   ");
	outString.append(msg);

	_logFile << outString << endl;
	cout << outString << endl;

	_logFile.flush();

	for(list<LogCatListener*>::iterator it=_listeners.begin();
		it!=_listeners.end(); it++){
			(*it)->newMsg(outString);
	}
}

void LogCat::addProcessBar(const char* title, int processTotal)
{
	_processTotal = processTotal;
	_process = 0;

	cout << string(title) << endl;
	for(list<LogCatListener*>::iterator it=_listeners.begin();
		it!=_listeners.end(); it++){
			(*it)->newMsg(string(title));
	}

	string processBar;
	for(int i=0; i<_processTotal; i++) processBar.append("=");
	processBar.append("   0%");

	cout << processBar;
	fflush(stdout);

	for(list<LogCatListener*>::iterator it=_listeners.begin();
		it!=_listeners.end(); it++){
			(*it)->newMsg(processBar);
	}
}

void LogCat::setProcessBar(float percent)
{
	int p = percent*_processTotal;
	int sub_p = p - _process;
	_process = p;

	if(sub_p > 0){
		string pBar;
		for(int i=0; i< _process; i++)
			pBar.append(">");
		for(int i=_process; i<_processTotal; i++)
			pBar.append("=");
		int percentage = percent*100;
		if(percentage < 10)
			pBar.append("   ");
		else if(percentage < 100)
			pBar.append("  ");
		else pBar.append(" ");
		pBar.append(to_string(percentage));
		pBar.append("%");

		cout << "\r" << pBar;
		fflush(stdout);

		for(list<LogCatListener*>::iterator it=_listeners.begin();
			it!=_listeners.end(); it++){
				(*it)->newMsg(string("\r"));
				(*it)->newMsg(pBar);
		}
	}
}

void LogCat::removeProcessBar()
{
	setProcessBar(1);
	cout << endl;
}
