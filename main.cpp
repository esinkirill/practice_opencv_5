#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

string getCardName(const Mat& image, const vector<Mat>& templates, const vector<Mat>& templatesDescriptors, const vector<string>& templateNames) {
    Ptr<FlannBasedMatcher> matcher = FlannBasedMatcher::create();
    Ptr<Feature2D> sift = SIFT::create();
    vector<KeyPoint> kp;
    Mat descriptors;
    sift->detectAndCompute(image, noArray(), kp, descriptors);

    auto minDistIndex = distance(templatesDescriptors.begin(), min_element(templatesDescriptors.begin(), templatesDescriptors.end(),
        [&](const Mat& a, const Mat& b) {
            vector<DMatch> matchesA, matchesB;
            matcher->match(descriptors, a, matchesA);
            matcher->match(descriptors, b, matchesB);

            double distA = 0.0, distB = 0.0;
            for (const auto& match : matchesA) {
                distA += match.distance;
            }
            for (const auto& match : matchesB) {
                distB += match.distance;
            }

            return distA < distB;
        }));

    return (minDistIndex != templatesDescriptors.size()) ? templateNames[minDistIndex] : "Unknown";
}

void rotateImage(RotatedRect& box, Mat& image, Mat& cropped) {
    Mat rotated;
    Size rectSize = box.size;
    if (box.angle < -45.) {
        swap(rectSize.width, rectSize.height);
        box.angle += 90.0;
    }
    Mat M = getRotationMatrix2D(box.center, box.angle, 1.0);
    warpAffine(image, rotated, M, image.size(), INTER_CUBIC);
    getRectSubPix(rotated, rectSize, box.center, cropped);
    if (cropped.size().width > cropped.size().height) {
        rotate(cropped, cropped, ROTATE_90_CLOCKWISE);
    }
}

int main(int argc, char** argv) {
    vector<string> templateNames = { "2 of Diamonds", "King of Clubs", "Queen of Clubs", "Ace of Diamonds" };
    vector<Mat> templates;
    vector<vector<KeyPoint>> templatesKeypoints;
    vector<Mat> templatesDescriptors;
    string templateFolder = "D:/card/";
    for (int i = 1; i <= 4; i++) {
        Mat templateImage = imread(templateFolder + to_string(i) + ".jpg");
        templates.push_back(templateImage);
        Ptr<Feature2D> sift = SIFT::create();
        vector<KeyPoint> kp;
        Mat descriptors;
        sift->detectAndCompute(templateImage, noArray(), kp, descriptors);
        templatesKeypoints.push_back(kp);
        templatesDescriptors.push_back(descriptors);
    }
    Mat image = imread("D:/test2.jpg");
    resize(image, image, Size(), 0.5, 0.5);
    Mat imageClone = image.clone();
    Mat frame = image.clone();
    Mat gaussFrame, edgesFrame;
    GaussianBlur(frame, gaussFrame, Size(11, 11), 0);
    Canny(gaussFrame, edgesFrame, 100, 150);
    vector<Vec4i> hierarchy;
    vector<vector<Point>> contours;
    findContours(edgesFrame, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    for (size_t i = 0; i < contours.size(); i++) {
        double epsilon = 0.02 * arcLength(contours[i], true);
        vector<Point> approx;
        approxPolyDP(contours[i], approx, epsilon, true);
        RotatedRect box = minAreaRect(approx);
        Mat img;
        rotateImage(box, frame, img);
        if (!img.empty()) {
            string name = getCardName(img, templates, templatesDescriptors, templateNames);
            if (approx.size() == 4 && isContourConvex(approx)) {
                Scalar color = Scalar(0, 255, 0);
                drawContours(frame, contours, (int)i, color, 1, LINE_8, hierarchy, 0);
                Moments M = moments(approx);
                Point center(M.m10 / M.m00, M.m01 / M.m00);
                putText(frame, name, center, FONT_HERSHEY_COMPLEX_SMALL, 0.65, color, 2);
            }
        }
    }
    imshow("image", frame);
    waitKey(0);
    destroyAllWindows();
    return 0;
}
