#include "mycl.h"
#include <iostream>

#pragma warning( disable : 4996 )

#define OPENCL_CHECK_ERRORS(ERR)		   \
    if(ERR != CL_SUCCESS)                  \
    {                                      \
    cerr                                   \
    << "OpenCL error with code " << ERR    \
    << " happened in file " << __FILE__    \
    << " at line " << __LINE__             \
    << ". Exiting...\n";                   \
    return ERR;                            \
    }


cl_int findPlatformID(cl_platform_id* &platforms, cl_int& selected_platform_index)
{
	platforms = NULL;
	selected_platform_index = -1;

	cl_int error = CL_SUCCESS;
	cl_uint num_of_platforms = 0;	//OpenCL平台数量
									// 得到平台数目
	error = clGetPlatformIDs(0, 0, &num_of_platforms);
	OPENCL_CHECK_ERRORS(error);
	cout << "可用平台数: " << num_of_platforms << endl;

	// 得到所有平台的ID
	platforms = new cl_platform_id[num_of_platforms];
	error = clGetPlatformIDs(num_of_platforms, platforms, 0);
	OPENCL_CHECK_ERRORS(error);


	if (num_of_platforms <= 0)
	{
		cerr << "no platform is found " << endl;
		return 1;
	}

	//遍历平台，选择一个Intel平台的
	for (cl_uint i = 0; i < num_of_platforms; ++i)
	{
		size_t platform_name_length = 0;
		error = clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 0, 0, &platform_name_length);				//调用第一次，得到名称的长度
		OPENCL_CHECK_ERRORS(error);

		char* platform_name = new char[platform_name_length];
		error = clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, platform_name_length, platform_name, 0);	//调用第二次，得到名称字符串
		OPENCL_CHECK_ERRORS(error);
		cout << "[" << i << "] " << platform_name << endl;

		if (true)															// have not selected yet	
		{
			selected_platform_index = i;
			cout << "Selected [" << selected_platform_index << "], the " << platform_name << " platform." << endl;
			delete[] platform_name;
			break;
		}
		delete[] platform_name;
	}
	return error;
}

cl_int createResource(/*const char* platformName,*/ Cl_Resource& resource)
{
	cl_int error = CL_SUCCESS;

	//获取AMD平台
	cl_platform_id* platforms = NULL;
	cl_int selected_platform_index = -1;
	error = findPlatformID(/*platformName,*/ platforms, selected_platform_index);
	OPENCL_CHECK_ERRORS(error)

		//创建 Device
		error = clGetDeviceIDs(platforms[selected_platform_index], CL_DEVICE_TYPE_GPU, 1, &resource.device, NULL);
	OPENCL_CHECK_ERRORS(error)

		//创建 Context
		resource.context = clCreateContext(0, 1, &resource.device, NULL, NULL, &error);
	OPENCL_CHECK_ERRORS(error)

		//创建 Command-queue CL_QUEUE_PROFILING_ENABLE开启才能计时
		resource.queue = clCreateCommandQueue(resource.context, resource.device, CL_QUEUE_PROFILING_ENABLE, &error);
	OPENCL_CHECK_ERRORS(error)
		return error;
}

cl_int createGPUmemery(const Cl_Resource& cl_res, cl_mem_flags flags, void* src, size_t size, cl_mem& dst)
{
	cl_int error = CL_SUCCESS;
	dst = clCreateBuffer(cl_res.context, flags, size, src, &error);
	OPENCL_CHECK_ERRORS(error)
		return error;
}

cl_int importProgramSource(const char* source, size_t src_size, Cl_Resource& cl_res, bool showLog)
{
	cl_int error = CL_SUCCESS;
	//创建编译运行kernel函数
	cl_res.program = clCreateProgramWithSource(cl_res.context, 1, &source, &src_size, &error);
	OPENCL_CHECK_ERRORS(error)

	// Builds the program
	error = clBuildProgram(cl_res.program, 1, &cl_res.device, NULL, NULL, NULL);
	OPENCL_CHECK_ERRORS(error)

		if (showLog)
		{
			// Shows the log
			char* build_log;
			size_t log_size;
			// First call to know the proper size
			clGetProgramBuildInfo(cl_res.program, cl_res.device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
			build_log = new char[log_size + 1];
			// Second call to get the log
			clGetProgramBuildInfo(cl_res.program, cl_res.device, CL_PROGRAM_BUILD_LOG, log_size, build_log, NULL);
			build_log[log_size] = '\0';
			cout << build_log << endl;
			delete[] build_log;
		}
	return error;
}

cl_int importProgramSource(const char* fileName, Cl_Resource& cl_res, bool showLog)
{
	cl_int error = CL_SUCCESS;
	FILE* fp = fopen(fileName, "rb");
	fseek(fp, 0, SEEK_END);
	size_t src_size = ftell(fp);
	fseek(fp, 0, SEEK_SET);
	if (src_size == 0)
	{
		return 1;
	}
	const char* source = new char[src_size];
	fread((void*)source, 1, src_size, fp);
	fclose(fp);

	//创建编译运行kernel函数
	cl_res.program = clCreateProgramWithSource(cl_res.context, 1, &source, &src_size, &error);
	OPENCL_CHECK_ERRORS(error)
		delete[] source;

	// Builds the program
	error = clBuildProgram(cl_res.program, 1, &cl_res.device, NULL, NULL, NULL);
	OPENCL_CHECK_ERRORS(error)

		if (showLog)
		{
			// Shows the log
			char* build_log;
			size_t log_size;
			// First call to know the proper size
			clGetProgramBuildInfo(cl_res.program, cl_res.device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
			build_log = new char[log_size + 1];
			// Second call to get the log
			clGetProgramBuildInfo(cl_res.program, cl_res.device, CL_PROGRAM_BUILD_LOG, log_size, build_log, NULL);
			build_log[log_size] = '\0';
			cout << build_log << endl;
			delete[] build_log;
		}
	return error;
}

cl_int createKernel(const Cl_Resource& cl_res, const char* functionName, cl_kernel& kernel)
{
	cl_int error = CL_SUCCESS;
	kernel = clCreateKernel(cl_res.program, functionName, &error);
	OPENCL_CHECK_ERRORS(error)
		return error;
}

cl_int setKernelArgs(const cl_kernel& kernel, const vector<pair<size_t, void*>>& args)
{
	cl_int error = CL_SUCCESS;
	for (size_t i = 0, length = args.size(); i < length; i++)
	{
		error |= clSetKernelArg(kernel, i, args[i].first, args[i].second);
	}
	OPENCL_CHECK_ERRORS(error)
		return error;
}

cl_int runKernel(const cl_kernel& kernel, const Cl_Resource& cl_res, const vector<pair<size_t, void*>>& args,
	const cl_uint& work_dims, const size_t* global_work_size, const size_t* local_work_size, cl_ulong& execTime)
{
	cl_int error = CL_SUCCESS;
	error = setKernelArgs(kernel, args);
	OPENCL_CHECK_ERRORS(error)

		// 启动kernel
#if 1
		cl_event ev;
	error = clEnqueueNDRangeKernel(cl_res.queue, kernel, work_dims, NULL, global_work_size, local_work_size, 0, NULL, &ev);
	OPENCL_CHECK_ERRORS(error)
		clFinish(cl_res.queue);

	//计算kerenl执行时间 
	cl_ulong startTime, endTime;
	clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &startTime, NULL);
	clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &endTime, NULL);
	execTime = endTime - startTime;
#else
		error = clEnqueueNDRangeKernel(cl_res.queue, kernel, work_dims, NULL, global_work_size, NULL, 0, NULL, NULL);
	OPENCL_CHECK_ERRORS(error)
		clFinish(cl_res.queue);
#endif
	return error;
}

cl_int exportGPUmemery(const Cl_Resource& cl_res, const cl_mem& gpu_mem, const size_t& size, void* dst)
{
	cl_int error = CL_SUCCESS;
	clEnqueueReadBuffer(cl_res.queue, gpu_mem, CL_TRUE, 0, size, dst, 0, NULL, NULL);
	OPENCL_CHECK_ERRORS(error)
		return error;
}

int init_GPU_Environment(Cl_Resource& cl_res, const char* filename)
{
	cl_int error = CL_SUCCESS;
	error = createResource(cl_res);
	error |= importProgramSource(filename, cl_res, true);
	return error;
}

int init_GPU_Environment(Cl_Resource& cl_res, const char* source, size_t src_size)
{
	cl_int error = CL_SUCCESS;
	error = createResource(cl_res);
	error |= importProgramSource(source, src_size, cl_res, true);
	return error;
}

void release_GPU_Environment(Cl_Resource& cl_res)
{
	if (cl_res.queue)
	{
		clReleaseCommandQueue(cl_res.queue);
		cl_res.queue = nullptr;
	}

	if (cl_res.context)
	{
		clReleaseContext(cl_res.context);
		cl_res.context = nullptr;
	}
}