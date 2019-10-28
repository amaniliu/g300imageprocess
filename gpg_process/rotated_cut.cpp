#include <stdio.h>
#include <time.h>
#include <string>
#include "rotated_cut.h"
#include "mycl.h"
#include "ImageProcDiscardBlank.h"
using namespace std;

Cl_Resource cl_res;
char cl_code_src[] =
"__kernel void transforColor_threshold_opencl(__global uchar* src,\r\n"
"\t__global uchar* dst, uchar thre)\r\n"
"{\r\n"
"\tint i = get_global_id(0);\r\n"
"\tuchar b = src[i * 3 + 0];\r\n"
"\tuchar g = src[i * 3 + 1];\r\n"
"\tuchar r = src[i * 3 + 2];\r\n"
"\tdst[i] = max(r, max(b, g)) > thre ? 255 : 0;\r\n"
"}\r\n\r\n"
"__kernel void warpAffine_opencl(__global uchar* src, __global uchar* dst,\r\n"
"\tfloat a, float b, float c, float d, float e, float f,\r\n"
"\tint swidth, int sheight, int dwidth, int dheight, int numOfChannels)\r\n"
"{\r\n"
"\tint x_d = get_global_id(0);\r\n"
"\tint y_d = get_global_id(1);\r\n"
"\tint x_s = (int)(a * x_d + b * y_d + c);\r\n"
"\tint y_s = (int)(d * x_d + e * y_d + f);\r\n"
"\tfor (int i = 0; i < numOfChannels; i++)\r\n"
"\t{\r\n"
"\t	dst[(y_d * dwidth + x_d) * numOfChannels + i] = src[(y_s * swidth + x_s) * numOfChannels + i];\r\n"
"\t}\r\n"
"}";

int init_GPU_Environment()
{
#if 0
	return init_GPU_Environment(cl_res, "cl_code.cl");
#else
	return init_GPU_Environment(cl_res, cl_code_src, sizeof(cl_code_src));
#endif
}

void release_GPU_Environment()
{
	release_GPU_Environment(cl_res);
}

void MyConvexHull(const vector<Point>& src, vector<Point>& dst, bool clockwise = false)
{
	CvMemStorage* storage = cvCreateMemStorage(0);	//申请内存空间，用于存放源数据和结果数据
	CvSeq* ptseq = cvCreateSeq(CV_SEQ_KIND_GENERIC | CV_32SC2, sizeof(CvContour), sizeof(CvPoint), storage);	//ptseq作为storage的迭代器

	//填充源数据
	for (const Point& item : src)
	{
		CvPoint p;
		p.x = item.x;
		p.y = item.y;
		cvSeqPush(ptseq, &p);
	}

	//凸多边形计算，得到结果数据的迭代器hull，结果数据依然保存在storage分配的内存空间中
	CvSeq* hull = cvConvexHull2(ptseq, 0, clockwise ? CV_CLOCKWISE : CV_COUNTER_CLOCKWISE, 0);

	//填充结果数据到dst中
	dst.clear();
	int hullCount = hull->total;
	for (size_t i = 0; i < hullCount; i++)
	{
		dst.push_back(Point(**CV_GET_SEQ_ELEM(CvPoint*, hull, i)));
	}

	//释放storage
	cvClearMemStorage(storage);
}

Mat transforColor(const Mat& src)
{
	if (src.channels() == 1)
	{
		return src.clone();
	}

	vector<Mat> channels(3);
	cv::split(src, channels);

	Mat temp, dst;
	bitwise_or(channels[0], channels[1], temp);
	bitwise_or(channels[2], temp, dst);

	temp.release();
	for (Mat& index : channels)
	{
		index.release();
	}
	return dst;
}

void transforColor_threshold_opencl(const Mat& src, Mat& dst, unsigned char thre)
{
	cl_int error = CL_SUCCESS;

	cl_kernel run_kernel;
	error |= createKernel(cl_res, "transforColor_threshold_opencl", run_kernel);

	cl_mem mem_src;
	error |= createGPUmemery(cl_res, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, src.data, src.total() * src.channels(), mem_src);

	cl_mem mem_dst;
	error |= createGPUmemery(cl_res, CL_MEM_WRITE_ONLY, NULL, src.total(), mem_dst);

	size_t global_total[1] = { src.total() };

	cl_ulong time = 0;
	vector<pair<size_t, void*>> args;
	args.push_back(pair<size_t, void*>(sizeof(cl_mem), &mem_src));
	args.push_back(pair<size_t, void*>(sizeof(cl_mem), &mem_dst));
	args.push_back(pair<size_t, void*>(sizeof(unsigned char), &thre));
	error |= runKernel(run_kernel, cl_res, args, 1, global_total, NULL, time);
	std::cout << "run_kernel " << " time = " << time << "ns" << endl;

	exportGPUmemery(cl_res, mem_dst, src.total(), dst.data);
	clReleaseMemObject(mem_src);
	clReleaseMemObject(mem_dst);
}

void threshold_Mat(const Mat& src, Mat& dst, double thre, int noise)
{
	if (src.channels() == 3)
	{
		if (cl_res.context)
		{
			transforColor_threshold_opencl(src, dst, (unsigned char)thre);
			//imwrite("dst.bmp", dst);
		}
		else
		{
			//LOGD("threshold_Mat 000");
			Mat gray = transforColor(src);
			//LOGD("threshold_Mat 1111");
			threshold(gray, dst, thre, 255, THRESH_BINARY);
			gray.release();
		}
#if 0
		imwrite("gray.bmp", gray);
#endif
	}
	else
	{
		threshold(src, dst, thre, 255, THRESH_BINARY);
	}
	Mat element = getStructuringElement(MORPH_RECT, Size(noise, noise));
	morphologyEx(dst, dst, MORPH_OPEN, element);
#if 0
	imwrite("thre.bmp", dst);
#endif
}

void findContours(const Mat& src, vector<vector<Point>>& contours, vector<Vec4i>& hierarchy,
	int retr = RETR_LIST, int method = CHAIN_APPROX_SIMPLE, Point offset = Point(0, 0))
{
	CvMat c_image = src;
	MemStorage storage(cvCreateMemStorage());
	CvSeq* _ccontours = nullptr;
	cvFindContours(&c_image, storage, &_ccontours, sizeof(CvContour), retr, method, CvPoint(offset));

	if (!_ccontours)
	{
		contours.clear();
		return;
	}
	Seq<CvSeq*> all_contours(cvTreeToNodeSeq(_ccontours, sizeof(CvSeq), storage));
	int total = (int)all_contours.size();
	contours.resize(total);

	SeqIterator<CvSeq*> it = all_contours.begin();
	for (int i = 0; i < total; i++, ++it)
	{
		CvSeq* c = *it;
		((CvContour*)c)->color = (int)i;
		int count = (int)c->total;
		int* data = new int[count * 2];
		cvCvtSeqToArray(c, data);
		for (int j = 0; j < count; j++)
		{
			contours[i].push_back(Point(data[j * 2], data[j * 2 + 1]));
		}
		delete[] data;
	}

	hierarchy.resize(total);
	it = all_contours.begin();
	for (int i = 0; i < total; i++, ++it)
	{
		CvSeq* c = *it;
		int h_next = c->h_next ? ((CvContour*)c->h_next)->color : -1;
		int h_prev = c->h_prev ? ((CvContour*)c->h_prev)->color : -1;
		int v_next = c->v_next ? ((CvContour*)c->v_next)->color : -1;
		int v_prev = c->v_prev ? ((CvContour*)c->v_prev)->color : -1;
		hierarchy[i] = Vec4i(h_next, h_prev, v_next, v_prev);
	}

	cvClearMemStorage(storage);
}

vector<Point> getMaxContour(const vector<vector<Point>>& contours, const vector<Vec4i>& hierarchy)
{
	vector<Point> maxContour;
	if (contours.empty())
	{
		return maxContour;
	}

	if (contours.size() == 1)
	{
		maxContour = contours[0];
		return maxContour;
	}

	for (int i = 0, length = hierarchy.size(); i < length; i++)
	{
		if (hierarchy[i][3] == -1)
		{
			for (const auto &item : contours[i])
			{
				maxContour.push_back(item);
			}
		}
	}

	return maxContour;
}

RotatedRect getBoundingRect(const vector<Point>& contour)
{
	if (contour.empty())
	{
		return {};
	}

	RotatedRect rect = minAreaRect(contour);
	if (rect.angle < -45)
	{
		rect.angle += 90;
		double temp = rect.size.width;
		rect.size.width = rect.size.height;
		rect.size.height = temp;
	}

	return rect;
}

void fill_poly(Mat& src, const vector<vector<Point>>& contours, const Scalar& color, int lineType = 8, int shift = 0, Point offset = Point())
{
	int ncontours = contours.size();
	Point** ptsptr = new Point*[ncontours];
	int* npts = new int[ncontours];

	for (size_t i = 0; i < ncontours; i++)
	{
		ptsptr[i] = new Point[contours[i].size()];
		npts[i] = contours[i].size();
		for (size_t j = 0; j < npts[i]; j++)
		{
			ptsptr[i][j] = contours[i][j];
		}
	}

	fillPoly(src, (const Point**)ptsptr, (const int*)npts, ncontours, color, lineType, shift, offset);

	for (size_t i = 0; i < ncontours; i++)
	{
		delete[] ptsptr[i];
	}
	delete[] ptsptr;
	delete[] npts;
}

void polyIndent(vector<Point>& points, float indent)
{
	static Point zero(0, 0);
	Point center = getBoundingRect(points).center;
	for (Point& item : points)
	{
		Point vec = item - center;
		if (vec != zero)
		{
			float length = vec.x * vec.x + vec.y * vec.y;
			float x = sqrt((float)(vec.x * vec.x) / length) * indent;
			float y = sqrt((float)(vec.y * vec.y) / length) * indent;

			if (vec.x < 0)
			{
				x *= -1.0f;
			}
			if (vec.y < 0)
			{
				y *= -1.0f;
			}
			item.x -= x;
			item.y -= y;
		}
	}

	//convexHull(points, points);					//用于排序，确保轮廓随逆时针排序
	MyConvexHull(points, points);
}

#define R_COLOR 255
void fillBlackBackGround_cpu(Mat& src, vector<Point> points)
{
	int index_top = 0;
	int index_bottom = 0;
	for (size_t i = 0, length = points.size(); i < length; i++)
	{
		if (points[i].y < points[index_top].y)
		{
			index_top = i;
		}
		if (points[i].y > points[index_bottom].y)
		{
			index_bottom = i;
		}
	}

	vector<Point> edge_left;
	int temp = index_top;
	while (temp != index_bottom)
	{
		edge_left.push_back(points[temp]);
		temp = (temp + points.size() - 1) % points.size();
	}
	edge_left.push_back(points[index_bottom]);

	vector<Point> edge_right;
	temp = index_top;
	while (temp != index_bottom)
	{
		edge_right.push_back(points[temp]);
		temp = (temp + points.size() + 1) % points.size();
	}
	edge_right.push_back(points[index_bottom]);

	vector<int> left_edge;
	for (size_t i = 0, length = edge_left.size() - 1; i < length; i++)
	{
		int y_top = edge_left[i].y;
		int x_top = edge_left[i].x;
		int y_bottom = edge_left[i + 1].y;
		int x_bottom = edge_left[i + 1].x;
		for (size_t y = y_top; y < y_bottom; y++)
		{
			if (y_top != y_bottom && y < src.rows)
			{
				left_edge.push_back(((x_bottom - x_top) * y + x_top * y_bottom - x_bottom * y_top) / (y_bottom - y_top));
			}
		}
	}
	int step = src.step;
	unsigned char* ptr = src.data + edge_left[0].y * step;
	for (size_t i = 0, length = left_edge.size(); i < length; i++)
	{
		int offset = left_edge[i];
		memset(ptr + i * step, R_COLOR, (offset + 1) * src.channels());
	}
	vector<int> right_edge;
	for (size_t i = 0, length = edge_right.size() - 1; i < length; i++)
	{
		int y_top = edge_right[i].y;
		int x_top = edge_right[i].x;
		int y_bottom = edge_right[i + 1].y;
		int x_bottom = edge_right[i + 1].x;
		for (size_t y = y_top; y < y_bottom; y++)
		{
			if (y_top != y_bottom && y < src.rows)
			{
				right_edge.push_back(((x_bottom - x_top) * y + x_top * y_bottom - x_bottom * y_top) / (y_bottom - y_top));
			}
		}
	}

	ptr = src.data + edge_right[0].y * step;
	for (size_t i = 0, length = right_edge.size(); i < length; i++)
	{
		int offset = right_edge[i];
		memset(ptr + i * step + offset * src.channels(), R_COLOR, step - offset * src.channels());
	}

	if (edge_left[0].y > 0)
	{
		memset(src.data, R_COLOR, edge_left[0].y * step);
	}

	if (edge_left.back().y < src.rows)
	{
		memset(src.data + edge_left.back().y * step, R_COLOR, (src.rows - edge_left.back().y) * step);
	}
}

void fillBlackBackGround(Mat& src, vector<Point> points, float indent)
{
	polyIndent(points, indent);

	clock_t start, end;
	start = clock();
	fillBlackBackGround_cpu(src, points);
	end = clock();
	printf("fillBlackBackGround_cpu = %f\n", (double)(end - start) / CLOCKS_PER_SEC);
	/*
	clock_t start, end;
	start = clock();
	Mat mask(src.size(), CV_8UC1);
	memset(mask.data, 255, mask.total());
	end = clock();
	printf("zeros = %f\n", (double)(end - start) / CLOCKS_PER_SEC);

	start = clock();
	fillConvexPoly(mask, points, Scalar(0, 0, 0, 0));
	end = clock();
	printf("fillConvexPoly = %f\n", (double)(end - start) / CLOCKS_PER_SEC);

	start = clock();
	if (src.channels() == 3)
	{
	cvtColor(mask, mask, COLOR_GRAY2BGR);
	}
	bitwise_or(src, mask, src);
	end = clock();
	printf("bitwise_or = %f\n", (double)(end - start) / CLOCKS_PER_SEC);
	*/
}

Point warpPoint(Point p, const Mat& warp_mat)
{
	double src_data[3] = { (double)p.x, (double)p.y, 1 };
	Mat src(3, 1, warp_mat.type(), src_data);	//warp_mat.type() == CV_64FC1

												//int type = warp_mat.type();
	Mat dst = warp_mat * src;
	double* ptr = (double*)dst.data;

	return Point(ptr[0], ptr[1]);
}

void warpAffine_kernel_cpu(const Mat& src, Mat& dst, Mat& m, Size dSize)
{
	unsigned char* ptr_src = src.data;
	unsigned char* ptr_dst = dst.data;

	double* ptr_m = (double*)m.data;

	int dStep = dst.step;
	int sStep = src.step;
	double a = ptr_m[0];
	double b = ptr_m[1];
	double c = ptr_m[2];
	double d = ptr_m[3];
	double e = ptr_m[4];
	double f = ptr_m[5];
	int numOfChannels = src.channels();
	int swidth = src.cols;
	int sheight = src.rows;
	int dwidth = dSize.width;
	int dheight = dSize.height;
	for (size_t y_d = 0; y_d < dheight; y_d++)
	{
		for (size_t x_d = 0; x_d < dwidth; x_d++)
		{
			int x_s = (int)(a * x_d + b * y_d + c);
			int y_s = (int)(d * x_d + e * y_d + f);

			for (int i = 0; i < numOfChannels; i++)
			{
				ptr_dst[(y_d * dwidth + x_d) * numOfChannels + i] = ptr_src[(y_s * swidth + x_s) * numOfChannels + i];
			}
		}
	}
}

void my_warpAffine(const Mat& src, Mat& dst, Mat& m, Size dSize)
{
	if (/*cl_res.context*/false)
	{
#if 1
		cl_int error = CL_SUCCESS;

		cl_kernel run_kernel;
		error |= createKernel(cl_res, "warpAffine_opencl", run_kernel);

		cl_mem mem_src;
		error |= createGPUmemery(cl_res, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, src.data, src.total() * src.channels(), mem_src);

		cl_mem mem_dst;
		error |= createGPUmemery(cl_res, CL_MEM_WRITE_ONLY, NULL, dSize.width * dSize.height * src.channels(), mem_dst);

		float m_[6];
		double* ptr_m = (double*)m.data;
		m_[0] = ptr_m[0];
		m_[1] = ptr_m[1];
		m_[2] = ptr_m[2];
		m_[3] = ptr_m[3];
		m_[4] = ptr_m[4];
		m_[5] = ptr_m[5];

		size_t global_total[2] = { (size_t)dSize.width, (size_t)dSize.height };
		int swidth = src.cols;
		int sheight = src.rows;
		int dwidth = dSize.width;
		int dheight = dSize.height;
		int numOfChannels = src.channels();
		cl_ulong time = 0;
		vector<pair<size_t, void*>> args;
		args.push_back(pair<size_t, void*>(sizeof(cl_mem), &mem_src));
		args.push_back(pair<size_t, void*>(sizeof(cl_mem), &mem_dst));
		args.push_back(pair<size_t, void*>(sizeof(float), &m_[0]));
		args.push_back(pair<size_t, void*>(sizeof(float), &m_[1]));
		args.push_back(pair<size_t, void*>(sizeof(float), &m_[2]));
		args.push_back(pair<size_t, void*>(sizeof(float), &m_[3]));
		args.push_back(pair<size_t, void*>(sizeof(float), &m_[4]));
		args.push_back(pair<size_t, void*>(sizeof(float), &m_[5]));
		args.push_back(pair<size_t, void*>(sizeof(int), &swidth));
		args.push_back(pair<size_t, void*>(sizeof(int), &sheight));
		args.push_back(pair<size_t, void*>(sizeof(int), &dwidth));
		args.push_back(pair<size_t, void*>(sizeof(int), &dheight));
		args.push_back(pair<size_t, void*>(sizeof(int), &numOfChannels));
		error |= runKernel(run_kernel, cl_res, args, 2, global_total, NULL, time);
		std::cout << "warpAffine_opencl " << " time = " << time << "ns" << endl;

		exportGPUmemery(cl_res, mem_dst, dst.total() * numOfChannels, dst.data);
		clReleaseMemObject(mem_src);
		clReleaseMemObject(mem_dst);
#else
		warpAffine_kernel_cpu(src, dst, m, dSize);
#endif
	}
	else
	{
		warpAffine(src, dst, m, dSize, INTER_LINEAR);
	}
}


void points_zoom(vector<Point>& points, double scale_h, double scale_v)
{
	for (Point& p : points)
	{
		p.x *= scale_h;
		p.y *= scale_v;
	}
}
#define RE 1

bool isBlankPage(Mat& src, const vector<Point>& contours)
{
#if 1
	if (src.channels() == 3)
	{
		cvtColor(src, src, COLOR_BGR2GRAY);
	}
	CImageProcDiscardBlank blank;
	blank.apply(src, contours);
	return blank.getResult();
#else

	imwrite("src.bmp", src);
	//得到图像合适的尺寸，以便得到最快的处理速度
	int comfRows = getOptimalDFTSize(src.rows);
	int comfCols = getOptimalDFTSize(src.cols);

	//剪裁到合适的尺寸
	Mat dstImage;
	copyMakeBorder(src, dstImage, 0, comfRows - src.rows, 0, comfCols - src.cols, BORDER_CONSTANT, Scalar::all(0));

	//因为傅里叶变换结果是一个复数，在这里就是输出两个Mat类型的变量，所以需要用容器去存放这两个变量
	Mat plannes[] = { Mat_<float>(dstImage), Mat::zeros(dstImage.size(), CV_32F) };
	Mat complexImage;
	merge(plannes, 2, complexImage);

	//进行离散傅里叶变换
	dft(complexImage, complexImage);

	//对计算计算出来的复数 求 幅值
	split(complexImage, plannes);
	//将结果保存在Mat 类型的变量 中
	Mat magnitudeImage;
	magnitude(plannes[0], plannes[1], magnitudeImage);

	//为了显示出幅值，进行以下操作

	//1、进行对数尺度缩放
	magnitudeImage += Scalar::all(1);
	log(magnitudeImage, magnitudeImage);

	//2、之前的步骤中，对图像进行了扩展，现在，需要将扩展的截去
	magnitudeImage = magnitudeImage(Rect(0, 0, magnitudeImage.cols & -2, magnitudeImage.rows & -2));

	//3、重新分布幅度图 象限
	//中心
	int center_x = magnitudeImage.cols / 2;
	int center_y = magnitudeImage.rows / 2;
	//定义象限
	Mat q0 = magnitudeImage(Rect(0, 0, center_x, center_y));
	Mat q1 = magnitudeImage(Rect(center_x, 0, center_x, center_y));
	Mat q2 = magnitudeImage(Rect(0, center_y, center_x, center_y));
	Mat q3 = magnitudeImage(Rect(center_x, center_y, center_x, center_y));

	//4、交换象限
	Mat temp;
	q0.copyTo(temp);
	q3.copyTo(q0);
	temp.copyTo(q3);

	q1.copyTo(temp);
	q2.copyTo(q1);
	temp.copyTo(q2);

	//5、归一化处理
	//normalize(magnitudeImage, magnitudeImage, 0, 1, CV_MINMAX);

	imwrite("magnitudeImage.bmp", magnitudeImage);
	//imshow("【傅里叶变换】", magnitudeImage);
	return false;
#endif
}

void rotated_and_cut(const Mat& src, Mat& dst, int flags, double threshold, int noise, int indent)
{
	if (flags == RC_INVALID)
	{
		dst = src.clone();
		return;
	}

	clock_t start, end;
	Mat src_resize;
	resize(src, src_resize, Size(src.cols / RE, src.rows / RE));
	Mat scale_mat;
	Mat thre(src_resize.size(), CV_8UC1);
	//LOGD("rotated_and_cut111111");
	start = clock();
	threshold_Mat(src_resize, thre, threshold, noise);
	end = clock();
	printf("threshold_Mat = %f\n", (double)(end - start) / CLOCKS_PER_SEC);
	//LOGD("rotated_and_cut22222");

	vector<Vec4i> hierarchy;
	vector<vector<Point>> contours;

	start = clock();
	findContours(thre, contours, hierarchy, RETR_EXTERNAL);
	end = clock();
	printf("findContours = %f\n", (double)(end - start) / CLOCKS_PER_SEC);
	vector<Point> maxContour = getMaxContour(contours, hierarchy);

	if (flags & RC_BLANK_PAGE)
	{
		start = clock();
		bool blank = isBlankPage(src_resize, maxContour);
		end = clock();
		printf("isBlankPage = %f\n", (double)(end - start) / CLOCKS_PER_SEC);
		if (blank)
		{
			return;
		}
	}

	for (Point& item : maxContour)
	{
		item.x = item.x * RE + RE / 2;
		item.y = item.y * RE + RE / 2;
	}

	//LOGD("rotated_and_cut33333");
	//LOGD("maxContour:%d", maxContour.size());
	if (maxContour.size() == 0) {
		thre.release();
		dst = src.clone();
		return;
	}
	start = clock();
	MyConvexHull(maxContour, maxContour);
	end = clock();
	printf("MyConvexHull = %f\n", (double)(end - start) / CLOCKS_PER_SEC);
	thre.release();
	dst.release();
	//LOGD("rotated_and_cut44444");

	vector<Point> temp(maxContour);
	points_zoom(temp, 1, 1.5);
	RotatedRect rect = getBoundingRect(temp);
	Rect bounding_rect = rect.boundingRect();
	bounding_rect.y /= 1.5;
	bounding_rect.height /= 1.5;
	//LOGD("rotated_and_cut55555");
	if (flags & RC_CUT)
	{
		if (flags & RC_ROTATED)
		{
			dst = Mat::zeros(Size(rect.size.width, rect.size.height / 1.5), src.type());
		}
		else
		{
			dst = src(bounding_rect).clone();
		}
	}
	else
	{
		if (flags & RC_BLACK_BACKGROUD)
		{
			dst = src.clone();
		}
		else
		{
			dst = Mat::zeros(src.size(), src.type());
		}
	}

	//LOGD("rotated_and_cut66666");

	Mat warp_mat/*, warp_mat_n*/;
	if (flags & RC_ROTATED)
	{
		Point2f dstTri[3];
		Point2f srcTri[4];
		rect.points(srcTri);
		for (Point2f& p : srcTri)
		{
			p.y /= 1.5;
		}
		if (flags & RC_CUT)
		{
			dstTri[0] = Point2f(0, rect.size.height / 1.5 - 1);
			dstTri[1] = Point2f(0, 0);
			dstTri[2] = Point2f(rect.size.width - 1, 0);
		}
		else
		{
			float left = (src.cols - rect.size.width) / 2;
			float right = left + rect.size.width - 1;
			float top = (src.rows - rect.size.height / 1.5f) / 2;
			float bottom = top + rect.size.height / 1.5f - 1;
			dstTri[0] = Point2f(left, bottom);
			dstTri[1] = Point2f(left, top);
			dstTri[2] = Point2f(right, top);
		}

		warp_mat = getAffineTransform(srcTri, dstTri);
		/*warp_mat_n = getAffineTransform(dstTri, srcTri);*/

		Size dSize = (flags & RC_CUT) ? Size(rect.size.width, rect.size.height / 1.5) : dst.size();

		start = clock();
		my_warpAffine(src, dst, /*cl_res.context ? warp_mat_n : */warp_mat, dSize);
		end = clock();
		printf("my_warpAffine = %f\n", (double)(end - start) / CLOCKS_PER_SEC);
	}
	//LOGD("rotated_and_cut77777");
	if (flags & RC_BLACK_BACKGROUD)
	{
		if (flags & RC_ROTATED)
		{
			for (Point& item : maxContour)
			{
				item = warpPoint(item, warp_mat);
			}
		}
		else
		{
			if (flags & RC_CUT)
			{
				Point offset = bounding_rect.tl();
				for (Point& item : maxContour)
				{
					item -= offset;
				}
			}
		}

		start = clock();
		fillBlackBackGround(dst, maxContour, indent);
		end = clock();
		printf("fillBlackBackGround = %f\n", (double)(end - start) / CLOCKS_PER_SEC);
	}
	//LOGD("rotated_and_cut888888");
	//warp_mat.release();
	//warp_mat_n.release();
}

#if 0
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <complex>
#include <string>
#include <io.h>
void getFiles(string path, vector<string>& files)
{
	//文件句柄  
	long   hFile = 0;
	//文件信息  
	struct _finddata_t fileinfo;
	string p;
	if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1)
	{
		do
		{
			//如果是目录,迭代之  
			//如果不是,加入列表  
			if ((fileinfo.attrib &  _A_SUBDIR))
			{
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
					getFiles(p.assign(path).append("\\").append(fileinfo.name), files);
			}
			else
			{
				//files.push_back(p.assign(path).append("\\").append(fileinfo.name));
				files.push_back(fileinfo.name);
			}
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
}
#endif

#include <fstream>

void encode_mat2tiff(const Mat& src, vector<uchar>& encode)
{
	imencode(".jpg", src, encode);

	uchar* buffer = new uchar[encode.size()];
	for (size_t i = 0, length = encode.size(); i < length; i++)
	{
		buffer[i] = encode[i];
	}
	ofstream  file;
	file.open("tttt.jpg", ios::app | ios::out);
	file.write((char*)buffer, encode.size());
	delete[] buffer;
	file.close();
}
/*
void a(uchar* data, int im_width, int im_height, vector<Mat>& mv)
{
	cv::Mat mat(im_height, im_width * 5, CV_8UC1, data);

	cv::Mat mat_R		= mat(Rect(im_width * 0, 0, im_width, im_height)).clone();
	cv::Mat mat_G		= mat(Rect(im_width * 1, 0, im_width, im_height)).clone();
	cv::Mat mat_B		= mat(Rect(im_width * 2, 0, im_width, im_height)).clone();
	cv::Mat mat_Block   = mat(Rect(im_width * 3, 0, im_width, im_height)).clone();
	cv::Mat mat_UV		= mat(Rect(im_width * 4, 0, im_width, im_height)).clone();

	mv.push_back(mat_R);
	mv.push_back(mat_G);
	mv.push_back(mat_B);
	mv.push_back(mat_Block);
	mv.push_back(mat_UV);
}

void join(const vector<Mat>& mv, Mat& dst)
{
	int im_width = mv[0].cols;
	int im_height = 0;
	for (const Mat& item : mv)
	{
		im_height += item.cols;
	}

	dst = Mat(im_height, im_width, CV_8UC(mv[0].channels()));
	int offset_y = 0;
	for (int i = 0; i < mv.size(); i++)
	{
		Mat roi = dst(Rect(Point(0, offset_y), mv[i].size()));
		mv[i].copyTo(roi);

		offset_y += mv[i].rows;
	}
}
*/
int main()
{
	int b = init_GPU_Environment();

	Mat img = imread("0.jpg", IMREAD_COLOR);
	//vector<uchar> encode;
	//encode_mat2tiff(img, encode);

	clock_t start, end;
	start = clock();
	Mat dst;
	rotated_and_cut(img, dst, RC_ROTATED | RC_CUT | 0 | 0, 40, 1, 2);
	end = clock();
	printf("time = %f\n", (double)(end - start) / CLOCKS_PER_SEC);
	imwrite("0.bmp", dst);
#if 1
#else
	vector<string> filenames;
	getFiles(".\\RC_ROTATED_ONLY\\123", filenames);

	int index = 0;
	for (const string& name : filenames)
	{
		Mat img = imread(name, IMREAD_GRAYSCALE);
		Mat front = img(Rect(0, 0, img.cols / 2, img.rows));
		Mat back = img(Rect(img.cols / 2, 0, img.cols / 2, img.rows));

		string front_name = ".\\img_gray\\" + to_string(index) + ".bmp";
		imwrite(front_name, front);
		index++;
		string back_name = ".\\img_gray\\" + to_string(index) + ".bmp";
		imwrite(back_name, back);
		index++;
	}

	///
	getFiles(".\\img_src", filenames);

	//string filename = "104.bmp";
	for (const string& filename : filenames)
	{
		Mat img = imread(".\\img_src\\" + filename, IMREAD_COLOR);
		Mat dst;
		clock_t start, end;
		start = clock();
		rotated_and_cut(img, dst, 0 | 0 | 0 | RC_BLANK_PAGE, 40, 7, 4);
		end = clock();
		printf("time = %f\n", (double)(end - start) / CLOCKS_PER_SEC);
		if (dst.rows != 0)
		{
			imwrite(".\\you\\" + filename, img);
		}
		else
		{
			imwrite(".\\wu\\" + filename, img);
		}
	}
#endif

	release_GPU_Environment();
	getchar();
	return 0;
}