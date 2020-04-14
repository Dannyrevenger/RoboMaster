#include<opencv2/opencv.hpp>

#include<iostream>

// #include"../include/ArmorDetector.h"

using namespace cv;
using namespace std;
/*自定义消息类新*/
#define red 0
#define blue 1
#define _enemy_color 1
#define light_min_area 10
#define light_max_ratio  1.0

/*全局变量*/
float light_color_detect_extend_ratio = 1.1;
float light_contour_min_solidity = 0.5;
int g_nThresh = 60;
int g_nThresh_max = 255;
int lowThreshold;
int const max_lowThreshold = 100;

/*自定义函数申明*/
int OTSU(cv::Mat& img);
// void HYAdaptiveFindThreshold(Mat *dx, Mat *dy, double *low, double *high);

class LightDescriptor
{
	public:
	cv::RotatedRect rec() const
	{
		return cv::RotatedRect(center, cv::Size2f(width, length), angle);
	}
	public:
	float width;
	float length;
	cv::Point2f center;
	float angle;
	float area;
};

int main(int argc, char** argv) 
{
    Mat srcImage = imread("/home/daniel/catkin_ws/src/abb_detection/src/pic/1.png");
    if (srcImage.empty())
    {
        cout<<"can not load image \n"<<endl;
        return -1;
    }
    Mat channels[3];
    Mat _grayImg;        
    // 把一个3通道图像转换成3个单通道图像
    split(srcImage,channels);//分离色彩通道
    //预处理删除己方装甲板颜色
    if(_enemy_color == red)
    _grayImg=channels[2]-channels[0];//Get red-blue image;
    else _grayImg=channels[0]-channels[2];//Get blue-red image;

    Mat binBrightImg;
    int brightness_threshold = OTSU(_grayImg);
    
    //阈值化
    threshold( _grayImg, binBrightImg, brightness_threshold, 255, 2);
    //膨胀
    Mat element = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
    dilate(binBrightImg, binBrightImg, element);

	Canny( binBrightImg, binBrightImg, 50, 255, 3);
	imshow("test",binBrightImg);
    //定义轮廓
	vector<vector<Point>> lightContours;
	//定义椭圆
	vector<Rect> Rect( lightContours.size()); 
	//找轮廓
	findContours(binBrightImg, lightContours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
	for( int i = 0; i < lightContours.size(); i++)
    {
		//得到面积
    	float lightContourArea = contourArea(lightContours[i]);
    	//面积太小的不要
		// cout << "area" <<lightContourArea << endl;
    	//椭圆拟合区域得到外接矩形
		RotatedRect lightRec = fitEllipse(lightContours[i]);
		if(lightContourArea > 9)
		{
			if( lightRec.size.height > 0 && lightRec.size.width > 0)
			{
				if( lightRec.size.height / lightRec.size.width > 2)
				{
					cout << "height:" << lightRec.size.height << " " << "width:" << lightRec.size.width << endl;
					ellipse(srcImage, lightRec, Scalar(0,0,255), 1, 8);
				}		
    		}
		}
	}

    imshow("row",srcImage);
    
    waitKey(0);
	
 
	return 0;
}

int OTSU(cv::Mat& img)
{
	int nRows = img.rows;
	int nCols = img.cols;
	int threshold = 0;
 
	int nSumPix[256];
	float nProDis[256];
 
	for (int i = 0; i < nRows; i++)
	{
		for (int j = 0; j < nCols; j++)
		{
			nSumPix[(int)img.at<uchar>(i, j)]++;
		}
	}
 
	for (int i = 0; i < 256; i++)
	{
		nProDis[i] = (float)nSumPix[i] / (nCols*nRows);
	}
 
	float wb, wf; //比重. wb-背景部分； wf-前景部分
	float u0_temp, u1_temp, u0, u1;	//平均值
	float delta_temp;	//存放临时方差
	double delta_max = 0.0;	//初始化最大类间方差
	for (int i = 0; i < 256; i++)
	{
		wb = wf = u0_temp = u1_temp = u0 = u1 = delta_temp = 0;//初始化相关参数
		for (int j = 0; j < 256; j++)
		{
			//背景部分
			if (j <= i)
			{
				//当前i为分割阈值，第一类总的概率
				wb += nProDis[j];	//比重
				u0_temp += j * nProDis[j];
			}
			//前景部分
			else
			{
				//当前i为分割阈值，第一类总的概率
				wf += nProDis[i];	//比重
				u1_temp += j * nProDis[j];
			}
		}
		//------------分别计算各类的平均值------------
		u0 = u0_temp / wb;
		u1 = u1_temp / wf;
		//-----------计算最大类间方差------------
		delta_temp = (float)(wb*wf*pow((u0 - u1), 2));//形如pow(x,y);其作用是计算x的y次方。
		//------------依次找到最大类间方差下的阈值------------
		if (delta_temp > delta_max)
		{
			delta_max = delta_temp;
			threshold = i;
		}
	}//计算结束
	return threshold;	//返回OTUS计算出的阈值 
}


