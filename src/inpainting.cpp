#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "inpainting.h"

using namespace cv;

int PATCH_RADIUS = 4;



/**
 * Loads and prepares images for the inpainting process.
 * 
 * @param colorFilename Path to the color image file
 * @param maskFilename Path to the mask image file (0 for holes, 255 for filled regions)
 * @param colorMat Output color image matrix (normalized to [0,1])
 * @param maskMat Output mask matrix (0 for holes, 255 for filled)
 * @param grayMat Output grayscale version of the color image
 * 
 * The function:
 * 1. Loads the color and mask images
 * 2. Ensures mask matches color image dimensions
 * 3. Normalizes color values to [0,1]
 * 4. Adds padding around images to handle boundary cases
 * 5. Converts color to grayscale for processing
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
void showMat(const String& winname, const Mat& mat, int time)
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


/**
 * Extracts a patch of size (2*RADIUS + 1) x (2*RADIUS + 1) centered at point p.
 * 
 * @param mat Source image matrix
 * @param p Center point of the patch
 * @return Mat containing the extracted patch
 * @throws std::out_of_range if point is too close to image boundary
 */
Mat getPatch(const Mat& mat, const Point& p)
{
    // Validate patch boundaries
    if (RADIUS > p.x || p.x >= mat.cols-RADIUS || 
        RADIUS > p.y || p.y >= mat.rows-RADIUS) {
        std::cerr << "Patch extraction failed: Point (" << p.x << ", " << p.y 
                  << ") too close to image boundary" << std::endl;
        std::cerr << "Image size: " << mat.size() << ", Required margin: " << RADIUS << std::endl;
        throw std::out_of_range("Point too close to boundary for patch extraction");
    }

    return mat(Range(p.y-RADIUS, p.y+RADIUS+1),
              Range(p.x-RADIUS, p.x+RADIUS+1));
}


/**
 * Computes image gradients and their magnitude.
 * Uses Sobel operators for gradient computation and combines them for magnitude.
 * 
 * @param grayMat Input grayscale image
 * @param dx Output x-direction gradient
 * @param dy Output y-direction gradient
 * @param magnitude Output gradient magnitude
 */
void getDerivatives(const Mat& grayMat, Mat& dx, Mat& dy, Mat& magnitude)
{
    assert(grayMat.type() == CV_32FC1);
    
    // Compute gradients using Sobel operators
    Sobel(grayMat, dx, -1, 1, 0, -1);  // x-direction gradient
    Sobel(grayMat, dy, -1, 0, 1, -1);  // y-direction gradient
    
    // Compute gradient magnitude
    magnitude = Mat::zeros(grayMat.size(), CV_32F);
    for(int y = 0; y < grayMat.rows; y++) {
        for(int x = 0; x < grayMat.cols; x++) {
            float gx = dx.at<float>(y, x);
            float gy = dy.at<float>(y, x);
            magnitude.at<float>(y, x) = std::sqrt(gx*gx + gy*gy);
        }
    }
}


/**
 * Computes the unit normal vector at a point on the contour using least squares regression.
 * 
 * @param contour Vector of points forming the contour
 * @param point The point at which to compute the normal
 * @return Unit normal vector as Point2f
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


/**
 * Computes the confidence value for a patch.
 * Confidence is the average of all confidence values in the patch.
 * 
 * @param confidencePatch Matrix containing confidence values
 * @return Average confidence value in range [0,1]
 */
double computeConfidence(const Mat& confidencePatch)
{
    return mean(confidencePatch)[0];
}


/**
 * Computes the priority for each point on the contour.
 * Priority = |confidence * gradient · normal|
 * 
 * @param contours Vector of contours to process
 * @param grayMat Grayscale version of the image
 * @param confidenceMat Matrix containing confidence values
 * @param priorityMat Output matrix to store computed priorities
 */
void computePriority(const contours_t& contours, 
                    const Mat& grayMat, 
                    const Mat& confidenceMat, 
                    Mat& priorityMat)
{
    assert(grayMat.type() == CV_32FC1 &&
           priorityMat.type() == CV_32FC1 &&
           confidenceMat.type() == CV_32FC1);

    // Initialize matrices
    Mat dx, dy, magnitude;
    getDerivatives(grayMat, dx, dy, magnitude);
    
    // Create masked magnitude matrix
    Mat maskedMagnitude = Mat::zeros(magnitude.size(), magnitude.type());
    magnitude.copyTo(maskedMagnitude, (confidenceMat != 0.0f));
    erode(maskedMagnitude, maskedMagnitude, Mat());

    // Process each contour point
    for (const auto& contour : contours) {
        for (const auto& point : contour) {
            // Skip boundary points
            if (RADIUS > point.x || point.x >= grayMat.cols-RADIUS || 
                RADIUS > point.y || point.y >= grayMat.rows-RADIUS) {
                continue;
            }

            try {
                // Get confidence value
                Mat confidencePatch = getPatch(confidenceMat, point);
                double confidence = computeConfidence(confidencePatch);
                
                // Get normal vector
                Point2f normal = getNormal(contour, point);
                
                // Find maximum gradient in source region
                Mat magnitudePatch = getPatch(maskedMagnitude, point);
                Point maxPoint;
                minMaxLoc(magnitudePatch, nullptr, nullptr, nullptr, &maxPoint);
                
                // Compute gradient vector
                Point2f gradient(
                    -getPatch(dy, point).at<float>(maxPoint.y, maxPoint.x),
                    getPatch(dx, point).at<float>(maxPoint.y, maxPoint.x)
                );
                
                // Compute and store priority
                priorityMat.at<float>(point.y, point.x) = 
                    std::abs(static_cast<float>(confidence) * gradient.dot(normal));
            } 
            catch (const std::exception& e) {
                std::cerr << "Error processing point (" << point.x << ", " << point.y 
                          << "): " << e.what() << std::endl;
                continue;
            }
        }
    }
}


/**
 * Transfers pixel values from source patch to target patch according to mask.
 * 
 * @param psiHatQ Center point of source patch
 * @param psiHatP Center point of target patch
 * @param mat Image matrix to modify
 * @param maskMat Binary mask (0 for holes, 255 for filled)
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

/**
 * Computes Sum of Squared Differences (SSD) between template and source image regions.
 * Includes border handling and spatial weighting for local patch preference.
 * 
 * @param tmplate Template patch to match
 * @param source Source image to search in
 * @param tmplateMask Mask for the template
 * @param targetCenter Center point of target region
 * @return Matrix of SSD values, normalized to [0,1]
 */
Mat computeSSD(const Mat& tmplate, 
              const Mat& source, 
              const Mat& tmplateMask, 
              const Point& targetCenter)
{
    assert(tmplate.type() == CV_32FC3 && source.type() == CV_32FC3);
    assert(tmplate.rows <= source.rows && tmplate.cols <= source.cols);
    assert(tmplateMask.size() == tmplate.size());

    // Compute SSD using template matching
    Mat result;
    matchTemplate(source, tmplate, result, TM_SQDIFF, tmplateMask);
    normalize(result, result, 0, 1, NORM_MINMAX);
    
    // Add border padding
    copyMakeBorder(result, result, 
                  RADIUS, RADIUS, RADIUS, RADIUS, 
                  BORDER_CONSTANT, 1.1f);
    
    // Create and apply border mask
    Mat borderMask = Mat::zeros(result.size(), CV_8U);
    rectangle(borderMask, 
             Point(RADIUS, RADIUS), 
             Point(result.cols-RADIUS-1, result.rows-RADIUS-1), 
             Scalar(255), 
             -1);
    bitwise_not(borderMask, borderMask);
    result.setTo(1.1f, borderMask);

    return result;
}
