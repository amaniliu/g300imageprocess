#include "ImageProcDiscardBlank.h"

using namespace cv;

CImageProcDiscardBlank::CImageProcDiscardBlank(bool isnormal)
	:m_grap(12),
	m_thresh(50),
	m_blobSize(5)
{
}

CImageProcDiscardBlank::~CImageProcDiscardBlank(void)
{
}

void CImageProcDiscardBlank::points_zoom(vector<Point>& points, double scale_h, double scale_v)
{
	for (Point& p : points)
	{
		p.x *= scale_h;
		p.y *= scale_v;
	}
}

RotatedRect CImageProcDiscardBlank::getBoundingRect(const vector<Point>& contour)
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

#define V_SCALE 1.5

void CImageProcDiscardBlank::apply(Mat& src, const vector<Point>& contour)
{
	vector<Point> contour_temp(contour);
	points_zoom(contour_temp, 1, V_SCALE);
	RotatedRect rect = getBoundingRect(contour_temp); 

	Mat warp_mat;
	Point2f dstTri[3];
	Point2f srcTri[4];
	rect.points(srcTri);
	for (Point2f& p : srcTri)
	{
		p.y /= V_SCALE;
	}

	dstTri[0] = Point2f(0, rect.size.height / V_SCALE - 1);
	dstTri[1] = Point2f(0, 0);
	dstTri[2] = Point2f(rect.size.width - 1, 0);

	warp_mat = getAffineTransform(srcTri, dstTri);

	Size dSize = Size(rect.size.width, rect.size.height / V_SCALE);
	Mat dst;
	cv::warpAffine(src, dst, warp_mat, dSize, INTER_AREA);

	if (dst.cols < m_grap * 2 + m_blobSize || dst.rows < m_grap * 2 + m_blobSize)
	{
		m_res = true;
		return;
	}

	Mat dst_temp = dst(Rect(m_grap, m_grap, dst.cols - m_grap * 2, dst.rows - m_grap * 2));

	int col_size = dst_temp.cols / m_blobSize;
	int row_size = dst_temp.rows / m_blobSize;

	double mean_min = 255;
	double mean_max = 0;
	for (size_t i = 0; i < row_size; i++)
	{
		for (size_t j = 0; j < col_size; j++)
		{
			Rect roi(j * m_blobSize, i * m_blobSize, m_blobSize, m_blobSize);
			
			double mean_value = mean(dst_temp(roi))[0];
			mean_min = min(mean_min, mean_value);
			mean_max = max(mean_max, mean_value);

			if (abs(mean_min - mean_max) > 20)
			{
				m_res = false;
				return;
			}
		}
	}
	m_res = true;
}