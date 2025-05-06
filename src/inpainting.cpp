#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "inpainting.h"

using namespace cv;

int PATCH_RADIUS = 4;



/*
 * Load the color, mask, grayscale images with a border of size
 * radius around every image to prevent boundary collisions when taking patches
 */
void loadInpaintingImages(const std::string& colorFilename,
                         const std::string& maskFilename,
                         Mat& colorMat,
                         Mat& maskMat,
                         Mat& grayMat)
{
    assert(colorFilename.length() && maskFilename.length());

    // Load images
    colorMat = imread(colorFilename, 1); // color
    maskMat = imread(maskFilename, 0);   // grayscale

    assert(!colorMat.empty() && !maskMat.empty());

    // Resize mask to match color image if sizes don't match
    if (colorMat.size() != maskMat.size()) {
        std::cout << "Resizing mask from " << maskMat.size() << " to " << colorMat.size() << std::endl;
        resize(maskMat, maskMat, colorMat.size(), 0, 0, INTER_NEAREST);
    }

    // convert colorMat to depth CV_32F for colorspace conversions
    colorMat.convertTo(colorMat, CV_32F);
    colorMat /= 255.0f;

    // add border around colorMat
    copyMakeBorder(colorMat, colorMat,
                  RADIUS, RADIUS, RADIUS, RADIUS,
                  BORDER_CONSTANT,
                  Scalar_<float>(0,0,0));

    // Convert to grayscale
    cvtColor(colorMat, grayMat, COLOR_BGR2GRAY);

    // Add border to mask and mark border as filled (255)
    copyMakeBorder(maskMat, maskMat,
                  RADIUS, RADIUS, RADIUS, RADIUS,
                  BORDER_CONSTANT,
                  255);  // Mark border as filled

    // Verify sizes
    std::cout << "Final matrix sizes:" << std::endl;
    std::cout << "Color matrix: " << colorMat.size() << std::endl;
    std::cout << "Gray matrix: " << grayMat.size() << std::endl;
    std::cout << "Mask matrix: " << maskMat.size() << std::endl;

    assert(colorMat.size() == grayMat.size() && colorMat.size() == maskMat.size());
}


/*
 * Show a Mat object quickly. For testing purposes only.
 */
void showMat(const String& winname, const Mat& mat, int time = 5)
{
    assert(!mat.empty());
    namedWindow(winname);
    imshow(winname, mat);
    waitKey(time);
    destroyWindow(winname);
}


/*
 * Extract closed boundary from mask.
 */
void getContours(const Mat& mask,
                contours_t& contours,
                hierarchy_t& hierarchy)
{
    assert(mask.type() == CV_8UC1);
    findContours(mask, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE);
}


/*
 * Get a patch of size RADIUS around point p in mat.
 */
Mat getPatch(const Mat& mat, const Point& p)
{
    // Check if point is within valid bounds
    if (RADIUS > p.x || p.x >= mat.cols-RADIUS || 
        RADIUS > p.y || p.y >= mat.rows-RADIUS) {
        std::cout << "WARNING: Point (" << p.x << ", " << p.y 
                  << ") too close to border for patch extraction" << std::endl;
        std::cout << "Image size: " << mat.size() << ", RADIUS: " << RADIUS << std::endl;
        std::cout << "Valid x range: [" << RADIUS << ", " << mat.cols-RADIUS-1 << "]" << std::endl;
        std::cout << "Valid y range: [" << RADIUS << ", " << mat.rows-RADIUS-1 << "]" << std::endl;
        throw std::out_of_range("Point too close to border for patch extraction");
    }

    return mat(Range(p.y-RADIUS, p.y+RADIUS+1),
              Range(p.x-RADIUS, p.x+RADIUS+1));
}


// get the x and y derivatives of a patch centered at patchCenter in image
// computed using a 3x3 sobel filter
void getDerivatives(const Mat& grayMat, Mat& dx, Mat& dy)
{
    assert(grayMat.type() == CV_32FC1);
    Sobel(grayMat, dx, -1, 1, 0, -1);
    Sobel(grayMat, dy, -1, 0, 1, -1);
}


/*
 * Get the unit normal of a dense list of boundary points centered around point p.
 */
Point2f getNormal(const contour_t& contour, const Point& point)
{
    int sz = (int)contour.size();
    assert(sz != 0);

    int pointIndex = (int)(std::find(contour.begin(), contour.end(), point) - contour.begin());
    assert(pointIndex != contour.size());

    if (sz == 1) {
        return Point2f(1.0f, 0.0f);
    } 
    else if (sz < 2 * BORDER_RADIUS + 1) {
        // Too few points in contour to use LSTSQ regression
        // return the normal with respect to adjacent neighbourhood
        Point adj = contour[(pointIndex + 1) % sz] - contour[pointIndex];
        return Point2f(adj.y, -adj.x) / norm(adj);
    }

    // Use least square regression
    // create X and Y mat to SVD
    Mat X(Size(2, 2*BORDER_RADIUS+1), CV_32F);
    Mat Y(Size(1, 2*BORDER_RADIUS+1), CV_32F);

    assert(X.rows == Y.rows && X.cols == 2 && Y.cols == 1 && 
           X.type() == Y.type() && Y.type() == CV_32F);

    int i = mod((pointIndex - BORDER_RADIUS), sz);
    float* Xrow;
    float* Yrow;

    int count = 0;
    int countXequal = 0;
    while (count < 2*BORDER_RADIUS+1) {
        Xrow = X.ptr<float>(count);
        Xrow[0] = contour[i].x;
        Xrow[1] = 1.0f;

        Yrow = Y.ptr<float>(count);
        Yrow[0] = contour[i].y;

        if (Xrow[0] == contour[pointIndex].x) {
            ++countXequal;
        }

        i = mod(i+1, sz);
        ++count;
    }

    if (countXequal == count) {
        return Point2f(1.0f, 0.0f);
    }

    // to find the line of best fit
    Mat sol;
    solve(X, Y, sol, DECOMP_SVD);

    assert(sol.type() == CV_32F);

    float slope = sol.ptr<float>(0)[0];
    Point2f normal(-slope, 1);

    return normal / norm(normal);
}


/*
 * Return the confidence of confidencePatch
 */
double computeConfidence(const Mat& confidencePatch)
{
    return sum(confidencePatch)[0] / (double)confidencePatch.total();
}


/*
 * Iterate over every contour point in contours and compute the
 * priority of path centered at point using grayMat and confidenceMat
 */
void computePriority(const contours_t& contours, 
                    const Mat& grayMat, 
                    const Mat& confidenceMat, 
                    Mat& priorityMat)
{
    assert(grayMat.type() == CV_32FC1 &&
           priorityMat.type() == CV_32FC1 &&
           confidenceMat.type() == CV_32FC1);

    // define some patches
    Mat confidencePatch;
    Mat magnitudePatch;

    Point2f normal;
    Point maxPoint;
    Point2f gradient;

    double confidence;

    // get the derivatives and magnitude of the greyscale image
    Mat dx, dy, magnitude;
    getDerivatives(grayMat, dx, dy);
    magnitude(dx, dy, magnitude);

    // mask the magnitude
    Mat maskedMagnitude(magnitude.size(), magnitude.type(), Scalar_<float>(0));
    magnitude.copyTo(maskedMagnitude, (confidenceMat != 0.0f));
    erode(maskedMagnitude, maskedMagnitude, Mat());

    assert(maskedMagnitude.type() == CV_32FC1);

    // for each point in contour
    Point point;

    for (int i = 0; i < contours.size(); ++i) {
        contour_t contour = contours[i];

        for (int j = 0; j < contour.size(); ++j) {
            point = contour[j];

            // Skip points that are too close to the border
            if (RADIUS > point.x || point.x >= grayMat.cols-RADIUS || 
                RADIUS > point.y || point.y >= grayMat.rows-RADIUS) {
                continue;
            }

            try {
                confidencePatch = getPatch(confidenceMat, point);

                // get confidence of patch
                confidence = sum(confidencePatch)[0] / (double)confidencePatch.total();
                assert(0 <= confidence && confidence <= 1.0f);

                // get the normal to the border around point
                normal = getNormal(contour, point);

                // get the maximum gradient in source around patch
                magnitudePatch = getPatch(maskedMagnitude, point);
                minMaxLoc(magnitudePatch, NULL, NULL, NULL, &maxPoint);
                gradient = Point2f(
                    -getPatch(dy, point).ptr<float>(maxPoint.y)[maxPoint.x],
                    getPatch(dx, point).ptr<float>(maxPoint.y)[maxPoint.x]
                );

                // set the priority in priorityMat
                priorityMat.ptr<float>(point.y)[point.x] = 
                    std::abs((float)confidence * gradient.dot(normal));
                assert(priorityMat.ptr<float>(point.y)[point.x] >= 0);
            } 
            catch (const std::exception& e) {
                std::cout << "Error processing point (" << point.x << ", " << point.y 
                          << "): " << e.what() << std::endl;
                continue;
            }
        }
    }
}


/*
 * Transfer the values from patch centered at psiHatQ to patch centered at psiHatP in
 * mat according to maskMat.
 */
void transferPatch(const Point& psiHatQ, 
                  const Point& psiHatP, 
                  Mat& mat, 
                  const Mat& maskMat)
{
    assert(maskMat.type() == CV_8U);
    assert(mat.size() == maskMat.size());
    assert(RADIUS <= psiHatQ.x && psiHatQ.x < mat.cols-RADIUS && 
           RADIUS <= psiHatQ.y && psiHatQ.y < mat.rows-RADIUS);
    assert(RADIUS <= psiHatP.x && psiHatP.x < mat.cols-RADIUS && 
           RADIUS <= psiHatP.y && psiHatP.y < mat.rows-RADIUS);

    // copy contents of psiHatQ to psiHatP with mask
    getPatch(mat, psiHatQ).copyTo(getPatch(mat, psiHatP), getPatch(maskMat, psiHatP));
}

/*
 * Runs template matching with template and mask templateMask on source.
 * Resulting Mat is stored in result.
 *
 */
Mat computeSSD(const Mat& tmplate, 
              const Mat& source, 
              const Mat& tmplateMask, 
              const Point& targetCenter)
{
    assert(tmplate.type() == CV_32FC3 && source.type() == CV_32FC3);
    assert(tmplate.rows <= source.rows && tmplate.cols <= source.cols);
    assert(tmplateMask.size() == tmplate.size() && tmplate.type() == tmplateMask.type());

    Mat result(source.rows - tmplate.rows + 1, 
              source.cols - tmplate.cols + 1, 
              CV_32F, 0.0f);

    matchTemplate(source, tmplate, result, TM_SQDIFF, tmplateMask);
    normalize(result, result, 0, 1, NORM_MINMAX);
    
    // Add border with high values to prevent selecting points too close to border
    copyMakeBorder(result, result, 
                  RADIUS, RADIUS, RADIUS, RADIUS, 
                  BORDER_CONSTANT, 1.1f);
    
    // Set border regions to high values to prevent selection
    Mat borderMask = Mat::zeros(result.size(), CV_8U);
    rectangle(borderMask, 
             Point(RADIUS, RADIUS), 
             Point(result.cols-RADIUS-1, result.rows-RADIUS-1), 
             Scalar(255), 
             -1);
    bitwise_not(borderMask, borderMask);
    result.setTo(1.1f, borderMask);

    // // Add spatial penalty to prefer local patches
    // float lambda = 0.1f; // Tune this value as needed
    // for (int y = 0; y < result.rows; ++y) {
    //     for (int x = 0; x < result.cols; ++x) {
    //         int src_x = x + RADIUS;
    //         int src_y = y + RADIUS;
    //         float dist2 = (src_x - targetCenter.x) * (src_x - targetCenter.x) +
    //                       (src_y - targetCenter.y) * (src_y - targetCenter.y);
    //         result.at<float>(y, x) += lambda * dist2;
    //     }
    // }

    return result;
}
