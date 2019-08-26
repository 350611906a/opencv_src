#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char** argv) 
{
	String cascadeFilePath = "D:/opencv/opencv/build/etc/haarcascades/haarcascade_frontalface_alt.xml";
	CascadeClassifier face_cascade;
	if (!face_cascade.load(cascadeFilePath)) 
	{
		printf("could not load haar data...\n");
		return -1;
	}
	Mat src,src1, src2, gray_src;
	src = imread("./tuanjian.jpg");
	resize(src, src1, Size(src.rows/1, src.cols/1));
	cvtColor(src1, gray_src, COLOR_BGR2GRAY);
	equalizeHist(gray_src, gray_src);     //直方图均衡化，，用于提高图像的质量
	//imshow("input image", src1);

	vector<Rect> faces;
	face_cascade.detectMultiScale(gray_src, faces, 1.1, 3, 0, Size(20, 20));
	for (size_t t = 0; t < faces.size(); t++) 
	{
		rectangle(src1, faces[t], Scalar(0, 0, 255), 2, 8, 0);
	}
	namedWindow("output", CV_WINDOW_AUTOSIZE);
	resize(src1, src2, Size(src.rows / 2, src.cols / 2));
	imshow("output", src2);

	waitKey(0);
	return 0;
}
