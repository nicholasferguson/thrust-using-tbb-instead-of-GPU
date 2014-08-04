Essentially this project should prove that cuda files can be tested with tbb, then moved into cuda env.
I could not find this type of example...so I developed one.

	Example.1 is an exe  ( it runs a Monte Carlo simulation to calc pi...included as an exmple with Thrust files)
		This exe during build pulls in Thrust.Test.lib ... a static lib with thrust files. 
		Thrust files are in github Thrust.
		Thrust.Test.lib pulls in tbb libs.
		and Thrust.Test.lib pulls in files from cuda tool kit.
		
		TBB is a cross platform parallel lib from INTEL.
		https://www.threadingbuildingblocks.org/

		Thrust files are designed to work with parallel backbone devices of either 
			- CUDA
			- OMP
			- TBB

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


<some directory>/Example.1
		This is the monte_carlo.u from thrust examples

======================		
Changes to cuda.v4.1
======================
<some directory>/cuda.v4.1
added file: 
include/thrust/iterator/detailbackend_iterator_spaces.h  ( why did I have to do this????.. )
and added code:
#elif THRUST_DEVICE_BACKEND == THRUST_DEVICE_SYSTEM_TBB
typedef omp_device_space_tag  default_device_space_tag

added code ( review if necessary...) to
device_system.h
host_system.h

======================
Preprocessors for Visual Studio 2013
======================
THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_TBB
THRUST_DEVICE_BACKEND=THRUST_DEVICE_SYSTEM_TBB
======================
Compiler optimization
======================
Optimization:  Maximum Speed(/02)
  Otherwise most debug apps run too slow.
  But cannot debug with this setting.
  Good just to see its possible performance



