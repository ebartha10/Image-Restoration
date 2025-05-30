//
// Created by Emeric on 16/05/2025.
//
#include "algorithms.h"
#include "inpainting.h"
const String base_path =  "X:\\Facultate\\An3\\Sem2\\PI\\Project\\Image-Restoration\\images\\tests\\";
#define DEBUG 1

/**
 * Computes the similarity percentage between two images of the same size.
 * @param img1 First image
 * @param img2 Second image
 * @return Similarity percentage (0-100)
 */
double computeImageSimilarity(const Mat& img1, const Mat& img2) {
    if (img1.size() != img2.size() || img1.type() != img2.type()) {
        throw std::runtime_error("Images must be of the same size and type");
    }

    // Convert images to float for more accurate computation
    Mat img1_float, img2_float;
    img1.convertTo(img1_float, CV_32F);
    img2.convertTo(img2_float, CV_32F);

    // Compute absolute difference
    Mat diff;
    cv::absdiff(img1_float, img2_float, diff);
    
    // Convert to single channel by summing differences across channels
    Mat diff_single;
    cv::reduce(diff.reshape(1, diff.rows * diff.cols), diff_single, 1, cv::REDUCE_SUM);
    diff_single = diff_single.reshape(1, diff.rows);
    
    // Count pixels that are different (difference > 0)
    int differentPixels = cv::countNonZero(diff_single);
    int totalPixels = img1.total();
    
    // Calculate similarity percentage
    double similarity = (1.0 - (static_cast<double>(differentPixels) / totalPixels)) * 100.0;
    
    return similarity;
}
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
    //showMat("Final Result", colorMat, 0);
    //imwrite("criminisi_final.bmp", colorMat);
    //waitKey();
}
Mat runBilinear(const String& colorFilename) {
    Mat colorMat = imread(colorFilename, IMREAD_COLOR);
    Mat resultMat = bilinear_reconstruction(colorMat);
    //imshow("Initial Mat", colorMat);
    imshow("Result Bilinear", resultMat);
    imwrite("result_bilinear.bmp", resultMat);
    return resultMat;
}
Mat runBicubic(const String& colorFilename) {
    Mat colorMat = imread(colorFilename, IMREAD_COLOR );
    Mat resultMat = bicubic_reconstruction(colorMat);
    imshow("Result Bicubic", resultMat);
    imwrite("result_bicubic.bmp", resultMat);
    return resultMat;
}

Mat runBilinearV2(const String& colorFilename, const String& maskFilename) {
    Mat colorMat = imread(colorFilename, IMREAD_COLOR);
    Mat maskMat = imread(maskFilename, IMREAD_GRAYSCALE);
    Mat resultMat = bilinear_reconstruction_with_mask(colorMat, maskMat);
    //imshow("Initial Mat", colorMat);
    imshow("V2 - BIL", resultMat);
    imwrite("result_bilinear.bmp", resultMat);
    return resultMat;
}
Mat runBicubicV2(const String& colorFilename, const String& maskFilename) {
    Mat colorMat = imread(colorFilename, IMREAD_COLOR);
    Mat maskMat = imread(maskFilename, IMREAD_GRAYSCALE);
    Mat resultMat = bicubic_reconstruction_with_mask(colorMat, maskMat);
    imshow("V2 - BIC", resultMat);
    imwrite("result_bicubic.bmp", resultMat);
    return resultMat;
}
void test1_lines() {
   // runBilinear(base_path + "Rubens_T1.bmp");
    // runBicubic(base_path + "Rubens_T1.bmp");
    auto start_time = std::chrono::high_resolution_clock::now();
    runCriminisi(base_path + "Rubens_T1.bmp", base_path + "Rubens_T1_mask.png");
    auto end_time = std::chrono::high_resolution_clock::now();
    cout << "Criminisi time:" << std::chrono::duration<double>(end_time - start_time).count() << endl;

    //waitKey();
}
void test2_lines() {
    runBilinear(base_path + "Rubens_T2.bmp");
    runBicubic(base_path + "Rubens_T2.bmp");
    runCriminisi(base_path + "Rubens_T2.bmp", base_path + "Rubens_T2_mask.png");

}
void test3_lines() {
    auto start_time = std::chrono::high_resolution_clock::now();
    runBilinear(base_path + "Rubens_T3.bmp");
    auto end_time = std::chrono::high_resolution_clock::now();
    cout << "Time for test 3 bilinear: " << std::chrono::duration<double>(end_time - start_time).count() << std::endl;
    start_time = std::chrono::high_resolution_clock::now();
    runBicubic(base_path + "Rubens_T3.bmp");
    end_time = std::chrono::high_resolution_clock::now();
    cout << "Time for test 3 bicubic: " << std::chrono::duration<double>(end_time - start_time).count() << std::endl;
    runCriminisi(base_path + "Rubens_T3.bmp", base_path + "Rubens_T3_mask.png");

}
void test5_Gaussian() {
    auto start_time = std::chrono::high_resolution_clock::now();
    runBilinear(base_path + "Hills_T5.bmp");
    auto end_time = std::chrono::high_resolution_clock::now();
    cout << "Time for test 5 bilinear: " << std::chrono::duration<double>(end_time - start_time).count() << std::endl;

    start_time = std::chrono::high_resolution_clock::now();
    runBicubic(base_path + "Hills_T5.bmp");
    end_time = std::chrono::high_resolution_clock::now();
    cout << "Time for test 5 bicubic: " << std::chrono::duration<double>(end_time - start_time).count() << std::endl;
    waitKey();
}
void test6_Gaussian() {
    auto start_time = std::chrono::high_resolution_clock::now();
    runBilinear(base_path + "Hills_T6.bmp");
    auto end_time = std::chrono::high_resolution_clock::now();
    cout << "Time for test 6 bilinear: " << std::chrono::duration<double>(end_time - start_time).count() << std::endl;

    start_time = std::chrono::high_resolution_clock::now();
    runBicubic(base_path + "Hills_T6.bmp");
    end_time = std::chrono::high_resolution_clock::now();
    cout << "Time for test 6 bilinear: " << std::chrono::duration<double>(end_time - start_time).count() << std::endl;

    waitKey();
}
void test7_Gaussian() {
    auto start_time = std::chrono::high_resolution_clock::now();
    runBilinear(base_path + "Hills_T7.bmp");
    auto end_time = std::chrono::high_resolution_clock::now();
    cout << "Time for test 7 bilinear: " << std::chrono::duration<double>(end_time - start_time).count() << std::endl;

    start_time = std::chrono::high_resolution_clock::now();
    runBicubic(base_path + "Hills_T7.bmp");
    end_time = std::chrono::high_resolution_clock::now();
    cout << "Time for test 7 bilinear: " << std::chrono::duration<double>(end_time - start_time).count() << std::endl;

    waitKey();
}
void test4_lines() {
    auto start_time = std::chrono::high_resolution_clock::now();
    runBilinear(base_path + "Hills_T4.bmp");
    auto end_time = std::chrono::high_resolution_clock::now();
    cout << "Time for test 4 bilinear: " << std::chrono::duration<double>(end_time - start_time).count() << std::endl;

    start_time = std::chrono::high_resolution_clock::now();
    runBicubic(base_path + "Hills_T4.bmp");
    end_time = std::chrono::high_resolution_clock::now();
    cout << "Time for test 4 bilinear: " << std::chrono::duration<double>(end_time - start_time).count() << std::endl;

    //runCriminisi(base_path + "Hills_T4_small.bmp", base_path + "Hills_T4_mask_small.png");
    waitKey();
}
void test9_ComparePerformances() {
    Mat bil1 = runBilinear(base_path + "Rubens_T1.bmp");
    Mat bic1 = runBicubic(base_path + "Rubens_T1.bmp");

    Mat bic2 = runBicubicV2(base_path + "Rubens_T1.bmp", base_path + "Rubens_T1_mask_V2.png");
    Mat bil2 = runBilinearV2(base_path + "Rubens_T1.bmp", base_path + "Rubens_T1_mask_V2.png");

    double similarityBilinear = computeImageSimilarity(bil1, bil2);
    double similarityBicubic = computeImageSimilarity(bic1, bic2);
    cout << "Similarity Bilinear: " << similarityBilinear << endl;
    cout << "Similarity Bicubic: " << similarityBicubic << endl;
}
void test10_ComparePerformances() {
    Mat bil1 = runBilinear(base_path + "Rubens_T3.bmp");
    Mat bic1 = runBicubic(base_path + "Rubens_T3.bmp");

    Mat bic2 = runBicubicV2(base_path + "Rubens_T3.bmp", base_path + "Rubens_T3_mask_V2.png");
    Mat bil2 = runBilinearV2(base_path + "Rubens_T3.bmp", base_path + "Rubens_T3_mask_V2.png");

    double similarityBilinear = computeImageSimilarity(bil1, bil2);
    double similarityBicubic = computeImageSimilarity(bic1, bic2);
    cout << "Similarity Bilinear: " << similarityBilinear << endl;
    cout << "Similarity Bicubic: " << similarityBicubic << endl;
}
void test11_ComparePerformances() {
    Mat bil1 = runBilinear(base_path + "Rubens_T2.bmp");
    Mat bic1 = runBicubic(base_path + "Rubens_T2.bmp");

    Mat bic2 = runBicubicV2(base_path + "Rubens_T2.bmp", base_path + "Rubens_T2_mask_V2.png");
    Mat bil2 = runBilinearV2(base_path + "Rubens_T2.bmp", base_path + "Rubens_T2_mask_V2.png");

    double similarityBilinear = computeImageSimilarity(bil1, bil2);
    double similarityBicubic = computeImageSimilarity(bic1, bic2);
    cout << "Similarity Bilinear: " << similarityBilinear << endl;
    cout << "Similarity Bicubic: " << similarityBicubic << endl;
}