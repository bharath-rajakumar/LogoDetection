// Final Project - I
//People detection - Using HOG and SIFT/SURF
#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include <opencv2/nonfree/nonfree.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"

#define FRAME_WIDTH 640
#define FRAME_HEIGHT 480

using namespace std;
using namespace cv;

int main(int argc, const char * argv[]) {

	// load the UTD logo image here
	Mat UTDlogo = imread("UTDLogo4.jpg", CV_LOAD_IMAGE_GRAYSCALE);

	// detect key points using SURF
	int minHessian = 500;

	SiftFeatureDetector detector(minHessian);
	vector<KeyPoint> logo_KeyPoint;
	detector.detect(UTDlogo, logo_KeyPoint);

	// calculate descriptors
	SiftDescriptorExtractor extractor;
	Mat logo_Descriptor;

	extractor.compute(UTDlogo, logo_KeyPoint, logo_Descriptor);

	FlannBasedMatcher matcher;

	VideoCapture cap(0);
	cap.set(CV_CAP_PROP_FRAME_WIDTH, FRAME_WIDTH);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT);

	if (!cap.isOpened()) {
		return -1;
	}

	// Use HOG's people detector to get a bounding box around the person on the live video capture
	Mat img;
	namedWindow("opencv", CV_WINDOW_AUTOSIZE);
	HOGDescriptor hog;
	hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());

	vector<Point> centroidCollections;
	Mat lastScene;

	int counter = 0;

	// keep capturing the video until esc key is pressed
	while (true) {
		cap >> img;
		cap >> lastScene;

		Mat person_Descriptor, person_matches;
		vector<KeyPoint> person_KeyPoint;
		vector<vector<DMatch> > matches;
		vector<DMatch> good_matches;

		if (img.empty())
			continue;

		// use the HOG's built in function to get a bounding box around the detected person
		vector<Rect> found, found_filtered;
		hog.detectMultiScale(img, found, 0, Size(4, 4), Size(32, 32), 1.05, 2);
		size_t i, j;
		for (i = 0; i < found.size(); i++) {
			Rect r = found[i];
			for (j = 0; j < found.size(); j++) {
				if (j != i && (r & found[j]) == r)
					break;
			}
			if (j == found.size())
				found_filtered.push_back(r);
		}

		// store the detected person and store it in a Mat
		Mat person;

		// bounding box's top left and bottom right corner
		int tl_x;
		int tl_y;
		int br_x;
		int br_y;

		// Detected person are stored in found_filtered
		if (found_filtered.size() > 0) {
			for (i = 0; i < found_filtered.size(); i++) {
				Rect r = found_filtered[i];
				r.x = r.x + cvRound(r.width * 0.1);
				r.width = cvRound(r.width * 0.8);
				r.y = r.y + cvRound(r.height * 0.07);
				r.height = cvRound(r.height * 0.8);

				// modify the bounding box
				// if the detected person's bounding box is not drawn completely
				// make changes to the top left corner or the bottom right corner
				// co-ordinates

				tl_x = r.tl().x;
				tl_y = r.tl().y;
				br_x = r.br().x;
				br_y = r.br().y;

				if (tl_x < 1) {
					tl_x = 1;
				}

				if (tl_y < 1) {
					tl_y = 1;
				}

				if (br_x > FRAME_WIDTH) {
					br_x = FRAME_WIDTH;
				}

				if (br_y > FRAME_HEIGHT) {
					br_y = FRAME_HEIGHT;
				}

				cout << "Filter size " << found_filtered.size() << endl;
				cout << tl_x << "," << tl_y << " | " << br_x - tl_x << ","
						<< br_y - tl_y << endl;

				person = img(Rect(tl_x, tl_y, br_x - tl_x, br_y - tl_y));

				// find cetroid of the person, which will be used later for drawing the path
				Point rectCentroid;
				rectCentroid.x = ((br_x - tl_y) / 2);
				rectCentroid.y = ((br_y - tl_y) / 2);

				Mat personGrayScale;
				cvtColor(person, personGrayScale, CV_BGR2GRAY);

				detector.detect(personGrayScale, person_KeyPoint);
				extractor.compute(personGrayScale, person_KeyPoint, person_Descriptor);

				// if the person descriptor is empty, go to the next frame
				if (!(person_Descriptor.empty())) {
					matcher.knnMatch(logo_Descriptor, person_Descriptor,
							matches, 2);

					for (int i = 0;
							i
									< min(person_Descriptor.rows - 1,
											(int) matches.size()); i++) {
						if ((matches[i][0].distance
								< 0.7 * (matches[i][1].distance))
								&& ((int) matches[i].size() <= 2
										&& (int) matches[i].size() > 0)) {
							good_matches.push_back(matches[i][0]);
						}
					}

					cout << "Good Matches: " << good_matches.size() << endl;

					if (good_matches.size() > 3) {
						//draw a white rectangle around the person wearing the UTD Logo tShirt
						rectangle(img, r.tl(), r.br(), Scalar(255, 255, 255),
								3);
						centroidCollections.push_back(rectCentroid);
					} else {
						// draw a green rectangle around the person
						rectangle(img, r.tl(), r.br(), Scalar(0, 255, 0), 3);
					}

				} else {
					cout << "Person Descriptor is empty !!!" << endl;
					cout << "Skipping current frame" << endl;
				}
			}
		}

		imshow("opencv", img);

		counter = counter + 1;

		if (waitKey(10) >= 0)
			break;
	}

	// trace the centroids to draw the path traversed by the person wearing the UTD logo tShirt
	while (true) {
		for (int i = 0; i < centroidCollections.size() - 1; i++) {
			// draw i and i+1th point
			Point a, b;
			a.x = centroidCollections[i].x;
			a.y = centroidCollections[i].y;
			b.x = centroidCollections[i + 1].x;
			b.y = centroidCollections[i + 1].y;
			line(lastScene, a, b, Scalar(255, 255, 255), 3, 8, 0);
		}

		imshow("Traversed Path", lastScene);

		if (waitKey(10) >= 0)
			break;
	}

	return 0;
}
