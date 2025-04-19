#include <iostream>
#include <opencv2/opencv.hpp>
#include "src/algorithms.h"
using namespace std;
using namespace cv;

int main() {

    Mat source = imread("X:\\Facultate\\An3\\Sem2\\PI\\Project\\Image-Restoration\\images\\gradientBrush.bmp",
                        IMREAD_COLOR);

    namedWindow("source", WINDOW_NORMAL);
    namedWindow("Restored Image bicubic", WINDOW_NORMAL);
    namedWindow("Restored Image bilinear", WINDOW_NORMAL);
    int width = source.cols ;
    int height = source.rows ;
    resizeWindow("source", width, height);
    resizeWindow("Restored Image bicubic", width, height);
    resizeWindow("Restored Image bilinear", width, height);

    imshow("source", source);

    Mat restored = bicubic_reconstruction(source);

    imshow("Restored Image bicubic", restored);
    if (!imwrite("RestoredImage.bmp", restored)) {
        cerr << "Failed to write the image to file." << endl;
    }

    // Mat restored2 = bilinear_reconstruction(source);
    //
    // imshow("Restored Image bilinear", restored2);
    // if (!imwrite("RestoredImage.bmp", restored2)) {
    //     cerr << "Failed to write the image to file." << endl;
    // }

    waitKey();

    return 0;
}