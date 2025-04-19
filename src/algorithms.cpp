//
// Created by Emeric on 05/04/2025.
//

#include "algorithms.h"

#include <opencv2/core/mat.hpp>
bool isInside(Mat source, Point P_0) {
    return P_0.x >= 0 && P_0.x < source.cols && P_0.y >= 0 && P_0.y < source.rows;
}
Mat compute_mask(Mat src, const int threshhold) {
    Mat mask;
    Mat hsv;
    cvtColor(src, hsv, COLOR_BGR2HSV);
    vector<Mat> hsvChannels;
    split(hsv, hsvChannels);
    Mat saturation = hsvChannels[1];
    mask = Mat::zeros(src.rows, src.cols, CV_8UC1);
    inRange(src, Scalar(240, 240, 240), Scalar(255, 255, 255), mask);

    // Add to mask potential unwanted pixels based on saturation difference
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            if (mask.at<uchar>(y, x) == 0) {
                int currentSaturation = saturation.at<uchar>(y, x);
                for (int dy = -1; dy <= 1; dy++) {
                    for (int dx = -1; dx <= 1; dx++) {
                        if (dx == 0 && dy == 0) continue;

                        int nx = x + dx;
                        int ny = y + dy;

                        if (isInside(src, Point(nx, ny)) && mask.at<uchar>(nx, ny) == 0) {
                            int neighborSaturation = saturation.at<uchar>(ny, nx);

                            int satDiff = abs(currentSaturation - neighborSaturation);
                            if (satDiff > threshhold) {
                                if(currentSaturation <= neighborSaturation) {
                                    mask.at<uchar>(y, x) = 255;
                                }
                                if(neighborSaturation < currentSaturation) {
                                    mask.at<uchar>(ny, nx) = 255;
                                }
                            }
                        }
                    }
                    //if (mask.at<uchar>(y, x) == 255) break;
                }
            }
        }
    }
    imshow("mask", mask);
    return mask;
}
// === BILINEAR INTERPOLATION === //
/**
 * Reconstructs a given colored image using bilinear interpolation.
 * @param src Image to restore. Needs to be a colored image.
 * @return Restored image with white pixels interpolated with neighbors.
 */
Mat bilinear_reconstruction(Mat src) {
    // Highlight white pixels
    Mat mask = Mat::zeros(src.rows, src.cols, CV_8UC1);
    Mat initialMask = Mat::zeros(src.rows, src.cols, CV_8UC1);
    inRange(src, Scalar(240, 240, 240), Scalar(255, 255, 255), initialMask);

    bool changed = false;
    int iteration = 0;
    const int maxSatDiff = 30;
    Mat dst = src.clone();

    do {
        changed = false;
        mask = compute_mask(dst, maxSatDiff);

        for (int y = 0; y < src.rows; y++) {
            for (int x = 0; x < src.cols; x++) {
                if (mask.at<uchar>(y, x) == 255) {
                    Vec3f interpolated = Vec3f(0, 0, 0);
                    int count = 0;
                    vector<Point> neighbors;

                    for (int dy = -1; dy <= 1; ++dy) {
                        for (int dx = -1; dx <= 1; ++dx) {
                            int nx = x + dx;
                            int ny = y + dy;
                            if (nx == x && ny == y) {
                                continue;
                            }
                            neighbors.push_back(Point(nx, ny));
                        }
                    }

                    for (auto point : neighbors) {
                        if (isInside(src, point) && mask.at<uchar>(point.y, point.x) == 0 && initialMask.at<uchar>(point.y, point.x) == 0) {
                            interpolated += dst.at<Vec3b>(point.y, point.x);
                            count++;
                        }
                    }
                    if (count > 0) {
                        dst.at<Vec3b>(y, x) = interpolated / count;
                        initialMask.at<uchar>(y, x) = 0;
                        changed = true;
                    }
                }
            }
        }
        imshow(to_string(iteration), dst);
        imshow("mask", mask);
        waitKey(1);
    } while (iteration++ < MAX_ITERATIONS && changed);
    return dst;
}

// ==== BICUBIC INTERPOLATION ==== //
/**
 * Cubic interpolation function
 * @param p Array of 4 points
 * @param x Position to interpolate at (0 <= x <= 1)
 * @return Interpolated value
 */
float cubicInterpolate(float p[4], float x) {
    return p[1] + 0.5 * x * (p[2] - p[0] + x * (2.0 * p[0] - 5.0 * p[1] + 4.0 * p[2] - p[3] + x * (3.0 * (p[1] - p[2]) + p[3] - p[0])));
}
/**
 * Bicubic interpolation for a single point
 * @param p 4x4 grid of values
 * @param x X position to interpolate at (0 <= x <= 1)
 * @param y Y position to interpolate at (0 <= y <= 1)
 * @return Interpolated value
 */
float bicubicInterpolate(float p[4][4], float x, float y) {
    float arr[4];
    for (int i = 0; i < 4; i++) {
        arr[i] = cubicInterpolate(p[i], x);
    }
    return cubicInterpolate(arr, y);
}

/**
 * Gets a 4x4 kernel around a given point
 * @param src Source image
 * @param mask Mask image (to identify valid/invalid pixels)
 * @param x X coordinate of center point
 * @param y Y coordinate of center point
 * @param channel Color channel to extract
 * @param kernel Output 4x4 kernel
 * @return Number of valid pixels in the kernel
 */
int getKernel(const Mat& src, const Mat& mask, int x, int y, int channel, float kernel[4][4]) {
    int validPixels = 0;

    for (int j = 0; j < 4; j++) {
        for (int i = 0; i < 4; i++) {
            int nx = x + i - 1;
            int ny = y + j - 1;

            // Handle boundary conditions by clamping to edge
            nx = max(0, min(src.cols - 1, nx));
            ny = max(0, min(src.rows - 1, ny));

            // Check if this is a valid pixel (not white)
            if (mask.at<uchar>(ny, nx) == 0) {
                kernel[j][i] = static_cast<float>(src.at<Vec3b>(ny, nx)[channel]);
                validPixels++;
            } else {
                kernel[j][i] = -1.0f; //invalid
            }
        }
    }

    return validPixels;
}

/**
 * Process a kernel to handle invalid pixels
 * @param kernel The 4x4 kernel to process
 * @param validPixels Number of valid pixels in the kernel
 * @return True if the kernel can be used for interpolation
 */
bool processKernel(float kernel[4][4], int validPixels) {
    if (validPixels < 2) {
        // Not enough valid pixels for interpolation
        return false;
    }

    // Replace invalid pixels with the average of valid neighbors
    bool changed = true;
    while (changed) {
        changed = false;
        for (int j = 0; j < 4; j++) {
            for (int i = 0; i < 4; i++) {
                if (kernel[j][i] < 0) {
                    float sum = 0.0f;
                    int count = 0;

                    // Check 8-connected neighbors
                    for (int nj = max(0, j-1); nj <= min(3, j+1); nj++) {
                        for (int ni = max(0, i-1); ni <= min(3, i+1); ni++) {
                            if ((ni != i || nj != j) && kernel[nj][ni] >= 0) {
                                sum += kernel[nj][ni];
                                count++;
                            }
                        }
                    }

                    if (count > 0) {
                        kernel[j][i] = sum / count;
                        changed = true;
                    }
                }
            }
        }
    }
    return true;
}

/**
 * Check if a point has valid neighbors for interpolation
 * @param mask Mask image
 * @param x X coordinate
 * @param y Y coordinate
 * @return True if the point has at least one valid neighbor for interpolation
 */
bool hasValidNeighbors(Mat& mask, int x, int y) {
    for (int j = -2; j <= 2; j++) {
        for (int i = -2; i <= 2; i++) {
            if (i == 0 && j == 0) continue;

            int nx = x + i;
            int ny = y + j;

            if (isInside(mask, Point(nx, ny)) && mask.at<uchar>(ny, nx) == 0) {
                return true;
            }
        }
    }
    return false;
}

/**
 * Reconstructs a given colored image using bicubic interpolation.
 * @param src Image to restore. Needs to be a colored image.
 * @return Restored image with white pixels interpolated with neighbors.
 */
Mat bicubic_reconstruction(const Mat& src) {
    // Highlight white pixels to interpolate
    Mat mask;

    Mat dst = src.clone();
    Mat processedMask = Mat::zeros(mask.size(), CV_8U); // Track successfully processed pixels

    bool changed = false;
    int iteration = 0;
    constexpr int maxAllowedDiff = 20   ;
    Mat currentMask = compute_mask(src, maxAllowedDiff);
    Mat originalMask = currentMask.clone();
    do {
        changed = false;

        // First pass: bicubic interpolation
        for (int y = 0; y < src.rows; y++) {
            for (int x = 0; x < src.cols; x++) {
                if (currentMask.at<uchar>(y, x) == 255 && hasValidNeighbors(currentMask, x, y)) {
                    Vec3f interpolated(0, 0, 0);
                    bool validInterpolation = true;

                    // Perform bicubic interpolation for each channel
                    for (int c = 0; c < 3; c++) {
                        float kernel[4][4];
                        int validPixels = getKernel(dst, currentMask, x, y, c, kernel);

                        if (processKernel(kernel, validPixels)) {
                            float value = bicubicInterpolate(kernel, 0.5f, 0.5f);
                            interpolated[c] = value;
                        } else {
                            validInterpolation = false;
                            break;
                        }
                    }

                    if (validInterpolation) {
                        Vec3b newColor(
                            saturate_cast<uchar>(interpolated[0]),
                            saturate_cast<uchar>(interpolated[1]),
                            saturate_cast<uchar>(interpolated[2])
                        );

                        // Always apply the interpolated color
                        dst.at<Vec3b>(y, x) = newColor;
                        currentMask.at<uchar>(y, x) = 0;
                        changed = true;

                    }
                }
            }
        }
        
        // Second pass: bilinear interpolation for remaining pixels
        for (int y = 0; y < src.rows; y++) {
            for (int x = 0; x < src.cols; x++) {
                if (currentMask.at<uchar>(y, x) == 255) {
                    Vec3f interpolated = Vec3f(0, 0, 0);
                    int count = 0;

                    // Check 8-connected neighbors
                    for (int dy = -1; dy <= 1; dy++) {
                        for (int dx = -1; dx <= 1; dx++) {
                            if (dx == 0 && dy == 0) continue;

                            int nx = x + dx;
                            int ny = y + dy;

                            if (isInside(src, Point(nx, ny)) && currentMask.at<uchar>(ny, nx) == 0) {
                                interpolated += dst.at<Vec3b>(ny, nx);
                                count++;
                            }
                        }
                    }

                    if (count > 0) {
                        Vec3b newColor = interpolated / count;

                        // Always apply the interpolated color
                        dst.at<Vec3b>(y, x) = newColor;
                        currentMask.at<uchar>(y, x) = 0;
                        changed = true;


                    }
                }
            }
        }
        
        iteration++;
        if(!changed) {
            currentMask = compute_mask(dst, maxAllowedDiff);
            changed = true;
        }

        imshow("Running", dst);
        imshow("Mask", currentMask);
    } while (iteration < MAX_ITERATIONS && changed);

    return dst;
}