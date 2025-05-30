#include <iostream>
#include <opencv2/opencv.hpp>
#include "src/algorithms.h"
#include "src/inpainting.h"
#include "src/tests.h"
using namespace std;
using namespace cv;

int main() {
    test9_ComparePerformances();
    waitKey();

    return 0;
}