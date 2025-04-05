//
// Created by Emeric on 05/04/2025.
//

#ifndef ALGORITHMS_H
#define ALGORITHMS_H
#include <opencv2/opencv.hpp>
#define MAX_ITERATIONS 50
using namespace cv;
using namespace std;
Mat bilinear_reconstruction(Mat src);
Mat bicubic_reconstruction(const Mat& src);
#endif //ALGORITHMS_H
