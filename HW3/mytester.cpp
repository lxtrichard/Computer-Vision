#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int PREV = 100;
Mat Equalize(Mat src);
void GetFiles(std::vector<Mat>& src);

int main(int argc, const char *argv[])
{
	if (argc < 3) {
		cout << "argument lacks." << endl;
		return 1;
	}
	int K;
	vector<Mat> src;
	GetFiles(src);
	int M = src[0].rows, N = src[0].cols;
	//get file list
	K = (int)src.size();
	Mat eigenvectors;
	//read eigenvectors
	FileStorage fs(argv[2], FileStorage::READ);
	fs["Eigenvectors"] >> eigenvectors;
	//Mat
	Mat average, coordinate;
	fs["Average"] >> average;
	fs["Coordinate"] >> coordinate;
	vector<Mat> eigenfaces;
	//read eigenfaces
	PREV = coordinate.rows;

	for (int t = 0; t < PREV; t++) {
		Mat tmpface;
		char str[15];
		sprintf_s(str, "eface%dth", t);
		fs[str] >> tmpface;
		eigenfaces.push_back(tmpface);
	}
	//测试部分
	Mat gray = imread(argv[1]);
	//show
	imshow("Origin", gray);
	Mat qry;
	cvtColor(gray, qry, CV_BGR2GRAY);
	Mat line_qry(M * N, 1, CV_64FC1);
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++) {
			line_qry.at<double>(i * N + j, 0) = qry.at<uchar>(i, j);
		}
	}
	//Projecting the query image into the PCA subspace.
	Mat query(PREV, 1, CV_64FC1);
	query = eigenvectors.t() * (line_qry - average);
	Mat result = Mat::zeros(M, N, CV_64FC1);
	for (int t = 0; t < PREV; t++) {
		double weight = query.at<double>(t, 0);
		result += weight * eigenfaces[t];
	}
	result.convertTo(result, CV_8UC1);
	imwrite("result.png", 0.5 * (qry + result));
	int min_id = 0;
	double min = norm(query - coordinate.col(0), NORM_L2);
	for (int i = 1; i < K; i++) {
		double form = norm(query - coordinate.col(i), NORM_L2);
		if (form < min) {
			min = form;
			min_id = i;
		}
	}
	imshow("Eigenface", src[min_id]);
	fs.release();
	waitKey();
	return 0;
}

Mat Equalize(Mat src)
{
	Mat g, e;
	cvtColor(src, g, CV_BGR2GRAY);
	equalizeHist(g, e);
	return e;
}

void GetFiles(std::vector<Mat>& src)
{
	string path = "JAFFE/";
	std::vector<cv::String> fn;
	cv::String pattern = "./JAFFE/*.tiff";
	glob(pattern, fn, false);
	int count = fn.size();
	for (int j = 0; j < count; j++) {
		src.push_back(Equalize(imread(fn[j])));
	}
}