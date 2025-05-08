#include <iostream>
#include <opencv2/opencv.hpp>
#include "src/algorithms.h"
#include "src/inpainting.h"

using namespace std;
using namespace cv;
#define DEBUG 1
void runCriminisi(const String& colorFilename, const string& maskFilename) {
    // Load and prepare images
    Mat colorMat, maskMat, grayMat;
    loadInpaintingImages(colorFilename, maskFilename, colorMat, maskMat, grayMat);
    
    if (DEBUG) {
        imshow("Initial photo", colorMat);
    }

    // Initialize confidence matrix
    Mat confidenceMat;
    maskMat.convertTo(confidenceMat, CV_32F);
    confidenceMat /= 255.0f;

    // Initialize priority matrix
    Mat priorityMat(confidenceMat.size(), CV_32FC1);
    
    // Create eroded mask for patch matching
    Mat erodedMask;
    erode(maskMat, erodedMask, Mat(), Point(-1, -1), RADIUS);

    // Main inpainting loop
    const size_t totalPixels = maskMat.total();
    while (countNonZero(maskMat) != totalPixels) {
        // Display progress
        cout << "Progress: " << (double)countNonZero(maskMat) / totalPixels << endl;

        // Reset priority matrix
        priorityMat.setTo(-0.1f);

        // Find contours of holes
        contours_t contours;
        hierarchy_t hierarchy;
        getContours((maskMat == 0), contours, hierarchy);

        if (DEBUG) {
            Mat drawMat = colorMat.clone();
            imshow("Progress", drawMat);
        }

        // Compute priorities for contour points
        computePriority(contours, grayMat, confidenceMat, priorityMat);

        // Find highest priority point
        Point targetPoint;
        minMaxLoc(priorityMat, nullptr, nullptr, nullptr, &targetPoint);

        // Get patches around target point
        Mat targetColorPatch = getPatch(colorMat, targetPoint);
        Mat targetConfidencePatch = getPatch(confidenceMat, targetPoint);

        // Create template mask for patch matching
        Mat confInv = (targetConfidencePatch != 0.0f);
        confInv.convertTo(confInv, CV_32F);
        confInv /= 255.0f;
        Mat templateMask;
        Mat mergeArrays[3] = {confInv, confInv, confInv};
        merge(mergeArrays, 3, templateMask);

        // Find best matching patch
        Mat ssdResult = computeSSD(targetColorPatch, colorMat, templateMask, targetPoint);
        ssdResult.setTo(1.1f, erodedMask == 0);
        
        Point sourcePoint;
        minMaxLoc(ssdResult, nullptr, nullptr, &sourcePoint);

        // Ensure source point is within valid bounds
        sourcePoint.x = std::max(RADIUS, std::min(sourcePoint.x, colorMat.cols-RADIUS-1));
        sourcePoint.y = std::max(RADIUS, std::min(sourcePoint.y, colorMat.rows-RADIUS-1));

        assert(sourcePoint != targetPoint);

        // Transfer patches
        transferPatch(sourcePoint, targetPoint, grayMat, (maskMat == 0));
        transferPatch(sourcePoint, targetPoint, colorMat, (maskMat == 0));

        // Update confidence
        double confidence = computeConfidence(targetConfidencePatch);
        assert(0 <= confidence && confidence <= 1.0f);
        targetConfidencePatch.setTo(confidence, (targetConfidencePatch == 0.0f));

        // Update mask
        maskMat = (confidenceMat != 0.0f);

        // Blend patches for smoother transition
        Mat targetPatch = getPatch(colorMat, targetPoint);
        Mat sourcePatch = getPatch(colorMat, sourcePoint);
        Mat blended;
        addWeighted(targetPatch, 0.7, sourcePatch, 0.3, 0, blended);
        blended.copyTo(getPatch(colorMat, targetPoint), getPatch(maskMat, targetPoint));
    }

    // Show final result
    showMat("Final Result", colorMat, 0);
    waitKey();
}
void runBilinear(const String& colorFilename) {
    Mat colorMat = imread(colorFilename, IMREAD_COLOR);
    Mat resultMat = bilinear_reconstruction(colorMat);
    imshow("Initial Mat", colorMat);
    imshow("Final Result", resultMat);
    imwrite("result_bilinear.bmp", resultMat);
}
void runBicubic(const String& colorFilename) {
    Mat colorMat = imread(colorFilename, IMREAD_COLOR );
    Mat resultMat = bicubic_reconstruction(colorMat);
    imwrite("result_bicubic.bmp", resultMat);
}
int main() {
    //runCriminisi("X:\\Facultate\\An3\\Sem2\\PI\\Project\\Image-Restoration\\images\\nature_small.png", "X:\\Facultate\\An3\\Sem2\\PI\\Project\\Image-Restoration\\images\\nature_small_mask.png");
    //runBilinear("X:\\Facultate\\An3\\Sem2\\PI\\Project\\Image-Restoration\\images\\gradientBrush.bmp");
    runBicubic("X:\\Facultate\\An3\\Sem2\\PI\\Project\\Image-Restoration\\images\\gradientBrush.bmp");
    waitKey();

    return 0;
}