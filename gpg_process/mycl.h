#ifndef GPU_OPENCL_H
#define GPU_OPENCL_H

#include <CL/cl_platform.h>
#include <CL/cl.h>
#include <vector>
using namespace std;

struct  Cl_Resource
{
	cl_context context;
	cl_command_queue queue;
	cl_device_id device;
	cl_program program;
};

int init_GPU_Environment(Cl_Resource& cl_res, const char* filename);

int init_GPU_Environment(Cl_Resource& cl_res, const char* source, size_t src_size);

void release_GPU_Environment(Cl_Resource& cl_res);

cl_int createKernel(const Cl_Resource& cl_res, const char* functionName, cl_kernel& kernel);

cl_int createGPUmemery(const Cl_Resource& cl_res, cl_mem_flags flags, void* src, size_t size, cl_mem& dst);

cl_int runKernel(const cl_kernel& kernel, const Cl_Resource& cl_res, const vector<pair<size_t, void*>>& args,
	const cl_uint& work_dims, const size_t* global_work_size, const size_t* local_work_size, cl_ulong& execTime);

cl_int exportGPUmemery(const Cl_Resource& cl_res, const cl_mem& gpu_mem, const size_t& size, void* dst);

#endif

