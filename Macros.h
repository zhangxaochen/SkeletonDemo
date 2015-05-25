#ifndef MACROS_H
#define MACROS_H

#include <exception>
#include <stdlib.h>

#define IMAGE_WIDTH 320
#define IMAGE_HEIGHT 240
//zhangxaochen:
// #define IMAGE_WIDTH 640
// #define IMAGE_HEIGHT 480

/// windows platform
#if defined(_WIN32)
	//#define CapgPrintf(format, ...) printf("")
	//zhangxaochen:
	#define CapgPrintf(format, ...) printf(format, ##__VA_ARGS__)
	#define LOGI(format, ...) printf(format, ##__VA_ARGS__)
	#define LOGD(format, ...) printf(format, ##__VA_ARGS__)

/// android platform
#elif (linux && ANDROID)
#include <android/log.h>
	#define CapgPrintf(format, ...) __android_log_print(ANDROID_LOG_DEBUG, "MotionRec", format, ##__VA_ARGS__)
	#define LOGI(format, ...) __android_log_print(ANDROID_LOG_INFO, "MotionRec", format, ##__VA_ARGS__)
	#define LOGD(format, ...) __android_log_print(ANDROID_LOG_INFO, "MotionRec", format, ##__VA_ARGS__)

/// linux platform
#elif (linux && (i386 || __x86_64 || __arm__))
	#define CapgPrintf(format, ...) printf(format, ##__VA_ARGS__)
	#define LOGI(format, ...) printf(format, ##__VA_ARGS__)
	#define LOGD(format, ...) printf(format, ##__VA_ARGS__)

#else
	#error MotionTracker  - Unsupported Platform!
#endif

#define ASSERT_RETURN(ret)	\
	if (!(ret)) return (ret);

#define ASSERT_EXCEPTION(ret)	\
	if (!(ret)) throw std::exception();

/// debug util
#ifdef _DEBUG
#define DebugPrintf(format, ...)			\
	do														\
	{															\
		CapgPrintf("%s, %s, %d\t", __FILE__, __FUNCTION__, __LINE__);	\
		CapgPrintf(format, __VA_ARGS__);	\
	}while(0)

#else
#define DebugPrintf(format, ...)
#endif

#endif
