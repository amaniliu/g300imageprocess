#pragma once
#include <math.h>
#include <vector>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

class CImageProcDiscardBlank
{
public:
	CImageProcDiscardBlank(bool isnormal = true);

	virtual ~CImageProcDiscardBlank(void);

	void apply(Mat& src, const vector<Point>& contour);

	inline bool getResult() { return m_res; }

	inline int getGrap() { return m_grap; }
	inline double getThresh() { return m_thresh; }

	inline void setGrap(int value) { m_grap = value; }
	inline void setThresh(double value) { m_thresh = value; }

private:
	RotatedRect getBoundingRect(const vector<Point>& contour);
	void points_zoom(vector<Point>& points, double scale_h, double scale_v);
private:
	int dSize;
	
	bool m_res;

	int m_grap;
	double m_thresh;
	int m_blobSize;
};
