__kernel void transforColor_threshold_opencl(__global uchar* src, __global uchar* dst, uchar thre)
{
	int i = get_global_id(0);
	uchar b = src[i * 3 + 0];
	uchar g = src[i * 3 + 1];
	uchar r = src[i * 3 + 2];
	dst[i] = max(r, max(b, g)) > thre ? 255 : 0;
}

__kernel void warpAffine_opencl(__global uchar* src, __global uchar* dst,
float a, float b, float c, float d, float e, float f,
int swidth, int sheight, int dwidth, int dheight, int numOfChannels)
{
	int x_d = get_global_id(0);
	int y_d = get_global_id(1);
	
	int x_s = (int)(a * x_d + b * y_d + c);
	int y_s = (int)(d * x_d + e * y_d + f);
	
	for (int i = 0; i < numOfChannels; i++)
	{
		dst[(y_d * dwidth + x_d) * numOfChannels + i] = src[(y_s * swidth + x_s) * numOfChannels + i];
	}
}