#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

using std::vector;
using std::cout;
using std::cerr;
using std::endl;

int main(int argc, char *argv[]) {
	int n_boards = 0;           // will be set by input list
	float image_sf = 0.5f;      // image scaling factor
	float delay = 1.f;
	int board_w = 0;
	int board_h = 0;
	string folders = "./calibration";
	string file_name = "./test.jpg";

	if (argc < 4 || argc > 8) {
		cout << "\nERROR: Wrong number of input parameters\n";
		return -1;
	}

	board_w = atoi(argv[1]);
	board_h = atoi(argv[2]);
	n_boards = atoi(argv[3]);

	if (argc > 4) {
		delay = atof(argv[4]);
	}
	if (argc > 5) {
		image_sf = atof(argv[5]);
	}
	if (argc > 6) {
		folders = argv[6];
	}
	int board_n = board_w * board_h;
	cv::Size board_sz = cv::Size(board_w, board_h);
	String path = folders+"/*.jpg";
	vector<String> filenames;
	glob(path, filenames, false);

	// ALLOCATE STORAGE
	//
	vector<vector<cv::Point2f> > image_points;
	vector<vector<cv::Point3f> > object_points;
	int corner_count;
	// Capture corner views: loop until we've got n_boards successful
	// captures (all corners on the board are found).
	//
	double last_captured_timestamp = 0;
	cv::Size image_size;
	int board_count = 0;

	for (size_t i = 0; (i < filenames.size()) && (board_count < n_boards); ++i) {
		cv::Mat image, image0 = cv::imread(filenames[i]);
		board_count += 1;
		if (!image0.data) {  // protect against no file
			cerr << filenames[i] << ", file #" << i << ", is not an image" << endl;
			continue;
		}
		image_size = image0.size();
		cv::resize(image0, image, cv::Size(), image_sf, image_sf, cv::INTER_LINEAR);

		// Find the board
		//
		vector<cv::Point2f> corners;
		bool found = cv::findChessboardCorners(image, board_sz, corners);

		// Draw it
		//
		drawChessboardCorners(image, board_sz, corners, found);  // will draw only if found

		// If we got a good board, add it to our data
		//
		if (found) {
			image ^= cv::Scalar::all(255);
			cv::Mat mcorners(corners);

			// do not copy the data
			mcorners *= (1.0 / image_sf);

			// scale the corner coordinates
			image_points.push_back(corners);
			object_points.push_back(vector<cv::Point3f>());
			vector<cv::Point3f> &opts = object_points.back();

			opts.resize(board_n);
			for (int j = 0; j < board_n; j++) {
				opts[j] = cv::Point3f(static_cast<float>(j / board_w),
					static_cast<float>(j % board_w), 0.0f);
			}
			cout << "Collected " << static_cast<int>(image_points.size())
				<< "total boards. This one from chessboard image #"
				<< i << ", " << filenames[i] << endl;
		}
		cv::imshow("Calibration", image);

		// show in color if we did collect the image
		if ((cv::waitKey(delay) & 255) == 27) {
			return -1;
		}
	}

	// END COLLECTION WHILE LOOP.
	cv::destroyWindow("Calibration");
	cout << "\n\n*** CALIBRATING THE CAMERA...\n" << endl;

	// CALIBRATE THE CAMERA!
	//
	cv::Mat intrinsic_matrix, distortion_coeffs;
	double err = cv::calibrateCamera(
		object_points, image_points, image_size, intrinsic_matrix,
		distortion_coeffs, cv::noArray(), cv::noArray(),
		cv::CALIB_ZERO_TANGENT_DIST | cv::CALIB_FIX_PRINCIPAL_POINT);

	// SAVE THE INTRINSICS AND DISTORTIONS
	cout << " *** DONE!\n\nReprojection error is " << err
		<< "\nStoring Intrinsics.xml and Distortions.xml files\n\n";
	cv::FileStorage fs("intrinsics.xml", cv::FileStorage::WRITE);
	fs << "image_width" << image_size.width << "image_height" << image_size.height
		<< "camera_matrix" << intrinsic_matrix << "distortion_coefficients"
		<< distortion_coeffs;
	fs.release();

	// EXAMPLE OF LOADING THESE MATRICES BACK IN:
	fs.open("intrinsics.xml", cv::FileStorage::READ);
	cout << "\nimage width: " << static_cast<int>(fs["image_width"]);
	cout << "\nimage height: " << static_cast<int>(fs["image_height"]);
	cv::Mat intrinsic, distortion;
	fs["camera_matrix"] >> intrinsic;
	fs["distortion_coefficients"] >> distortion;
	cout << "\nintrinsic matrix:" << intrinsic;
	cout << "\ndistortion coefficients: " << distortion << endl;

	// Build the undistort map which we will use for all
	// subsequent frames.
	//
	cv::Mat map1, map2;
	cv::initUndistortRectifyMap(intrinsic, distortion,
		cv::Mat(), intrinsic, image_size,
		CV_16SC2, map1, map2);

	// Just run the camera to the screen, now showing the raw and
	// the undistorted image.

	if (argc > 7) {
		file_name = argv[7];
	}
	cv::Mat gray_image, image, image0 = cv::imread(file_name, 1);
	if (image0.empty()) {
		cout << "Error: Couldn't load image " << file_name << endl;
		return -1;
	}
	cv::undistort(image0, image, intrinsic, distortion, intrinsic);
	cv::cvtColor(image, gray_image, cv::COLOR_BGRA2GRAY);

	vector<cv::Point2f> corners;
	bool found = cv::findChessboardCorners( // True if found
		image,                              // Input image
		board_sz,                           // Pattern size
		corners,                            // Results
		cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FILTER_QUADS);
	if (!found) {
		cout << "Couldn't acquire checkerboard on " << file_name << ", only found "
			<< corners.size() << " of " << board_n << " corners\n";
		return -1;
	}
	cv::cornerSubPix(
		gray_image,       // Input image
		corners,          // Initial guesses, also output
		cv::Size(11, 11), // Search window size
		cv::Size(-1, -1), // Zero zone (in this case, don't use)
		cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 30,
			0.1));
	cv::Point2f objPts[4], imgPts[4];
	objPts[0].x = 0;
	objPts[0].y = 0;
	objPts[1].x = board_w - 1;
	objPts[1].y = 0;
	objPts[2].x = 0;
	objPts[2].y = board_h - 1;
	objPts[3].x = board_w - 1;
	objPts[3].y = board_h - 1;
	imgPts[0] = corners[0];
	imgPts[1] = corners[board_w - 1];
	imgPts[2] = corners[(board_h - 1) * board_w];
	imgPts[3] = corners[(board_h - 1) * board_w + board_w - 1];
	// DRAW THE POINTS in order: B,G,R,YELLOW
//
	cv::circle(image, imgPts[0], 9, cv::Scalar(255, 0, 0), 3);
	cv::circle(image, imgPts[1], 9, cv::Scalar(0, 255, 0), 3);
	cv::circle(image, imgPts[2], 9, cv::Scalar(0, 0, 255), 3);
	cv::circle(image, imgPts[3], 9, cv::Scalar(0, 255, 255), 3);

	cv::drawChessboardCorners(image, board_sz, corners, found);
	cv::imshow("Checkers", image);
	cv::imwrite("Checkers.jpg", image);
	// FIND THE HOMOGRAPHY
  //
	cv::Mat H = cv::getPerspectiveTransform(objPts, imgPts);
	// LET THE USER ADJUST THE Z HEIGHT OF THE VIEW
 //
	cout << "\nPress 'd' for lower birdseye view, and 'u' for higher (it adjusts the apparent 'Z' height), Esc to exit" << endl;
	double Z = 15;
	cv::Mat birds_image;
	for (;;) {
		// escape key stops
		H.at<double>(2, 2) = Z;
		// USE HOMOGRAPHY TO REMAP THE VIEW
		//
		cv::warpPerspective(image,			// Source image
			birds_image, 	// Output image
			H,              // Transformation matrix
			image.size(),   // Size for output image
			cv::WARP_INVERSE_MAP | cv::INTER_LINEAR,
			cv::BORDER_CONSTANT, cv::Scalar::all(0) // Fill border with black
		);
		cv::imshow("Birds_Eye", birds_image);
		cv::imwrite("Birds_Eye.jpg", birds_image);
		int key = cv::waitKey() & 255;
		if (key == 'u')
			Z += 0.5;
		if (key == 'd')
			Z -= 0.5;
		if (key == 27)
			break;
	}
	// SHOW ROTATION AND TRANSLATION VECTORS
  //
	vector<cv::Point2f> image_points_1;
	vector<cv::Point3f> object_points_1;
	for (int i = 0; i < 4; ++i) {
		image_points_1.push_back(imgPts[i]);
		object_points_1.push_back(cv::Point3f(objPts[i].x, objPts[i].y, 0));
	}
	cv::Mat rvec, tvec, rmat;
	cv::solvePnP(object_points_1, 	// 3-d points in object coordinate
		image_points_1,  	// 2-d points in image coordinates
		intrinsic,     	// Our camera matrix
		cv::Mat(),     	// Since we corrected distortion in the
						 // beginning,now we have zero distortion
						 // coefficients
		rvec, 			// Output rotation *vector*.
		tvec  			// Output translation vector.
	);
	cv::Rodrigues(rvec, rmat);

	// PRINT AND EXIT
	cout << "rotation matrix: " << rmat << endl;
	cout << "translation vector: " << tvec << endl;
	cout << "homography matrix: " << H << endl;
	cout << "inverted homography matrix: " << H.inv() << endl;
	return 0;
}
