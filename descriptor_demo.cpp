#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <iostream>

using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;

int main(int argc, char** argv) 
{
	Mat img1 = imread("./box.png", IMREAD_GRAYSCALE);
	Mat img2 = imread("./box_in_scene.png", IMREAD_GRAYSCALE);
	if (!img1.data || !img2.data) 
	{
		return -1;
	}
	imshow("image1", img1);
	imshow("image2", img2);

	//1、定义SURF特征提取方法，检测并计算特征点和描述符
	int minHessian = 400;
	Ptr<SURF> detector = SURF::create(minHessian);
	vector<KeyPoint> keypoints_1;
	vector<KeyPoint> keypoints_2;

	Mat descriptor_1, descriptor_2;
	detector->detectAndCompute(img1, Mat(), keypoints_1, descriptor_1);
	detector->detectAndCompute(img2, Mat(), keypoints_2, descriptor_2);

	//2、定义暴力匹配方法,匹配特征描述符，得匹配结果matches
	BFMatcher matcher(NORM_L2);
	vector<DMatch> matches;

	double time_start = getTickCount();
	matcher.match(descriptor_1, descriptor_2, matches);
	double time_end = getTickCount();
	double time_use = (time_end - time_start) / getTickFrequency();
	printf("time = %f\n", time_use);

	//3、绘制匹配结果
	Mat matchesImg;
	drawMatches(img1, keypoints_1, img2, keypoints_2, matches, matchesImg);
	imshow("Descriptor Demo", matchesImg);

	waitKey(0);
	return 0;
}