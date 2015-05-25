#pragma once
#include "Vector3.h"
#include <vector>
#include <fstream>

typedef Vector3 Joint;
typedef std::vector<Joint> CapgSkeleton;

//************************************
// Method:    debug the skeleton to file
// FullName:  debugOutSkeleton
// Access:    public 
// Returns:   void
// Qualifier:
// Parameter: const Skeleton & s
//************************************
void debugOutSkeleton(const CapgSkeleton& s,const char* filename);