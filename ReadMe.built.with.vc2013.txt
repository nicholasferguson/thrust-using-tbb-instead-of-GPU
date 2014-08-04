Essentially this project should prove that cuda files can be tested with tbb, then moved into cuda env.

This example works with cudatoolkit and thrust, using tbb only.

Built using Visual Studio Express 2013, with VC++ Compiler Nov 2103 CTP ( increased c++11 friendly)

VC project will work out of box if you put the contents
as in this directory structure

<some directory>/Thrust.Test ( files in its thrust does not appear in cuda v4.1
		This is a clone from github/thrust/thrust
		I pruned files that were in cuda.v4.1
		Also each *.h file was edited
		#ifdef MSVC
		#include "stdafx.h"
		#endif

		In stdafx.h, there is
		#ifdef WIN32
		#include <Windows.h>
		#define __host__ 
		#define __device__ 
		#endif 
		
<some directory>/cuda.v4.1  ( this is a reduced version)
		I pruned files that were not needed for this test.
		for cuda v4.1, one of its files must be updated:
		backend_iterator_spaces.h
<some directory>/BlueTBB_tbb41_20130314oss
		https://wiki.alcf.anl.gov/parts/index.php/BlueTBB





