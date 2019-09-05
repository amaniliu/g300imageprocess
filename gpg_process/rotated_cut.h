//
// Created by ht on 2019/3/5.
//

#ifndef MY_OPENCV_LIBS_JNI_TEST_ROTATED_CUT_H
#define MY_OPENCV_LIBS_JNI_TEST_ROTATED_CUT_H

#ifndef MY_OPENCV_LIBS_JNI_TEST_ROTATED_CUT_DLL
#define MY_OPENCV_LIBS_JNI_TEST_ROTATED_CUT_DLL //extern "C" __declspec(dllexport)
#endif // !MY_OPENCV_LIBS_JNI_TEST_ROTATED_CUT_DLL

#include <opencv2/opencv.hpp>
using namespace cv;
	//
	enum RC_TYPE
	{
		RC_INVALID          = 0x0,
		RC_ROTATED          = 0x1,
		RC_CUT		        = 0x2,
		RC_BLACK_BACKGROUD  = 0x4,
		RC_BLANK_PAGE       = 0x8
	};

MY_OPENCV_LIBS_JNI_TEST_ROTATED_CUT_DLL	int init_GPU_Environment();

MY_OPENCV_LIBS_JNI_TEST_ROTATED_CUT_DLL void release_GPU_Environment();

MY_OPENCV_LIBS_JNI_TEST_ROTATED_CUT_DLL void rotated_and_cut(const Mat& src, Mat& dst, int flags = RC_ROTATED | RC_CUT | RC_BLACK_BACKGROUD | RC_BLANK_PAGE, double threshold = 40, int noise = 7, int indent = 5);

#endif //MY_OPENCV_LIBS_JNI_TEST_ROTATED_CUT_H
