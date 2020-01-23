#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int PREV;
Mat Equalize(Mat src);
void GetFiles(std::vector<Mat>& src);

int main(int argc, const char *argv[])
{
	if (argc < 3) {
		cout << "Arguments lacks." << endl;
		return 1;
	}
	double percent;
	percent = atof(argv[1]);
	std::vector<Mat> src;
	GetFiles(src);

	int K = (int)src.size();
	PREV = K * percent;
	int M = src[0].rows, N = src[0].cols;

	//transfer the total matrix into one column
	Mat X_matrix(M * N, K, CV_64FC1);
	for (int i = 0; i < M * N; i++) {
		for (int j = 0; j < K; j++) {
			X_matrix.at<double>(i, j) = src[j].at<uchar>(i / N, i % N);//(row, col)
		}
	}
	//get miu
	Mat average;
	X_matrix.col(0).copyTo(average);
	for (int i = 1; i < K; i++) {
		average += X_matrix.col(i);
	}
	average /= K;
	FileStorage fs(argv[2], FileStorage::WRITE);
	fs << "Average" << average;

	//X-u
	for (int i = 0; i < K; i++) {
		X_matrix.col(i) -= average;
	}
	Mat eigenvalues(K, 1, CV_64FC1), eigenvectors(K, K, CV_64FC1);
	//eigen(cov_matrix, eigenvalues, eigenvectors);
	eigen(X_matrix.t() * X_matrix, eigenvalues, eigenvectors);
	eigenvectors = eigenvectors.t();

	//calculate real eigen vectors(d*100) d=M*N
	Mat real_eig_vectors(M * N, PREV, CV_64FC1);
	for (int i = 0; i < PREV; i++) {
		real_eig_vectors.col(i) = X_matrix * eigenvectors.col(i);
		real_eig_vectors.col(i) /= norm(real_eig_vectors.col(i), NORM_L2);
	}
	fs << "Eigenvectors" << real_eig_vectors;

	Mat total_eigenface(2 * M, 5 * N, CV_8UC1);
	std::vector<Mat> eigenfaces;

	for (int t = 0; t < PREV; t++) {
		Mat tmpface(M, N, CV_64FC1);
		//Ttransfer 1D to 2D
		for (int i = 0; i < M; i++) {
			for (int j = 0; j < N; j++) {
				tmpface.at<double>(i, j) = real_eig_vectors.at<double>(i * N + j, t);
			}
		}
		Mat dst, eigenface;
		normalize(tmpface, dst, 0, 255, NORM_MINMAX);
		dst.convertTo(eigenface, CV_8UC1);
		eigenfaces.push_back(tmpface);
		char str[15];
		sprintf_s(str, "eface%dth", t);
		fs << str << tmpface;
		//start x, y
		if (t >= 10) {
			continue;
		}
		int startX = t / 5 * M, startY = (t % 5) * N;
		for (int i = 0; i < M; i++) {
			for (int j = 0; j < N; j++) {
				total_eigenface.at<uchar>(startX + i, startY + j) = eigenface.at<uchar>(i, j);
			}
		}
	}

	//ten eigenfaces shown in one big window
	imshow("Eigenfaces", total_eigenface);
	//write eigenfaces into files
	imwrite("Eigenfaces.jpg", total_eigenface);

	//calculate the coordinates
	Mat coordinate(PREV, K, CV_64FC1);
	for (int i = 0; i < K; i++) {
		coordinate.col(i) = real_eig_vectors.t() * (X_matrix.col(i) - average);
	}
	//store the coordinate
	fs << "Coordinate" << coordinate;
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