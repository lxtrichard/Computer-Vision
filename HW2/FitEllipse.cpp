#include <iostream>
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
using namespace std;
using namespace cv;


void Track_ellipse(Mat src_image, string path, int Threshold);

int main(int argc, char** argv)
{
	int Threshold = 125;
	string path = string(*(argv + 1));
	Mat src_image = imread(path, 0);
	if (src_image.empty())
		return 0;
	path = path.substr(0, path.find_last_of("\\"));
	imshow("Input", src_image);
	Track_ellipse(src_image, path, Threshold);
	waitKey(0);
	return 0;
}

void Track_ellipse(Mat src_image, string path, int Threshold)
{
	vector<vector<Point>> contours;
	Mat edge_image;
	Mat compare_img = src_image;
	Canny(src_image, edge_image, 150, 180);
	findContours(edge_image, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);//提取图片轮廓

	Mat point;
	Mat output_image = Mat::zeros(edge_image.size(), CV_8UC3);
	for (int i = 0; i < contours.size(); i++)
	{
		int count = contours[i].size();
		if (count < 6)
			continue;
		else
		{
			Mat(contours[i]).convertTo(point, CV_32F);
			RotatedRect box = fitEllipse(point);//椭圆拟合
			ellipse(output_image, box, Scalar(255, 255, 255), 1, CV_AA);//绘制椭圆
			ellipse(compare_img, box, Scalar(0, 255, 255), 1, LINE_AA);//在原图上绘制椭圆
		}
	}
	imshow("Output", output_image);
	imwrite("./output.png", output_image);
	imshow("Compare", compare_img);
	imwrite("./compare.png", compare_img);
}