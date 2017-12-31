#include <string>
#include <vector>
#include <limits>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/ml.hpp>
#include "../Common/Image.hpp"

using namespace std;
using namespace cv;
using namespace cv::ml;

#define MAX_CLUSTERS_PER_CLASS 20

void mark_nearest_cluster(Mat& vecwords, int cntr, Mat& centroids, Mat r){
	float dist = numeric_limits<float>::infinity();
	float dst = 0;
	int closest = 0;
	for(int i = 0; i < centroids.rows; i++){
		if((dst = norm(r-centroids.row(i))) < dist){
			dist = dst;
			closest = i;
		}
	}
	vecwords.at<float>(cntr, closest) += 1;
}

int main(int argv, char** argc){
	if(argv <= 1){
		cout << "How to use:" << endl;
		cout << string(argc[0]) << "  \"image_folder_path\"  " << endl;
		return 0;
	}

	//vector of image classes
	vector< vector<Mat> > images(argv-1);
	vector< vector<Mat> > test_images(argv-1);

	//Default folder location, to shorten input parameters
	string classes_base = "../../../Classes/";
	string image_base = "/original";
	string mask_base  = "/segment";

	//loading images
	for(int i = 1; i < argv; i++){
		cout << "Loading images of class: " << string(argc[i]) << endl;
		Image::loadFromFolder(images[i-1], classes_base + string(argc[i]) + image_base, classes_base + string(argc[i]) + mask_base);
	}
	
	Ptr<Feature2D> sft = xfeatures2d::SIFT::create();

	//allfeat equals to all images
	int allfeat = 0;
	for(int i = 0; i < images.size(); i++) allfeat += images[i].size();
	vector< vector<KeyPoint> > keypoints(allfeat);
	vector< Mat > descriptors(allfeat);

	//a vector of class labels
	Mat labels(allfeat, 1, CV_32S, Scalar(0));


	//counter
	int cntr = 0;

	//computing keypoints and associated descriptors
	for(int i = 0; i < images.size(); i++){
		for(int j = 0; j < images[i].size(); j++){
			sft->detectAndCompute(images[i][j], cv::noArray(), keypoints[cntr], descriptors[cntr]);
			labels.at<int>(cntr) = i;
			cntr++;
		}
	}

	//hierarchical clustering, due to unknown k
	//number of centroids = number of classes times max number of clusters per class
	Mat mcentroids(images.size() * MAX_CLUSTERS_PER_CLASS, descriptors[0].cols, descriptors[0].type());
	Mat mclust(0, descriptors[0].cols, descriptors[0].type());	
	for(int i = 0; i < allfeat; i++){
		vconcat(descriptors[i], mclust, mclust);
	}

	cout << "Descriptors: " << mclust.rows << " " << mclust.cols << endl;

	//getting clusters
	//num_clusters smaller or equal to number of rows in mcentroids
	cvflann::KMeansIndexParams params(8, 11, cvflann::FLANN_CENTERS_KMEANSPP);
	int num_clusters = flann::hierarchicalClustering<cvflann::L2<float>>(mclust, mcentroids, params);
	mcentroids = mcentroids(Rect(0, 0, descriptors[0].cols, num_clusters));

	cout << "Creating visual words class vectors" << endl;

	//creating vector(Mat) of visual words for each image in class
	Mat vecwords(allfeat, num_clusters, CV_32F, Scalar(0));
	
	cntr = 0;
	for(int i = 0; i < images.size(); i++){
		cout << "Class " << string(argc[i+1]) << endl;
		for(int j = 0; j < images[i].size(); j++){
			for(int k = 0; k < descriptors[cntr].rows; k++){
				mark_nearest_cluster(vecwords, cntr, mcentroids, descriptors[cntr].row(k));
			}
			cntr++;
		}
	}

	//saving 
	//TO-DO
	//support vector machine, learning phase
	//SVM auto train estimates best parameters using k-fold cross-validation
	Ptr<TrainData>  td = TrainData::create(vecwords, ROW_SAMPLE, labels);
	Ptr<SVM> svm = SVM::create();
    svm->trainAuto(td, 10);
  
  	//Confusion matrix
  	//columns - actual class
  	//rows - predicted class
    Mat confusion_matrix(images.size(), images.size(), CV_32S, Scalar(0));

    int allfeat2 = 0;
    string test_base = "../../../Test/";
    for(int i = 1; i < argv; i++){
		cout << "Loading images of class: " << string(argc[i]) << endl;
		Image::loadFromFolder(test_images[i-1], test_base + string(argc[i]));
		allfeat2 += test_images[i-1].size();
	}

	vector< vector<KeyPoint> > keypoints2(allfeat);
	vector< Mat > descriptors2(allfeat);

	Mat test_labels(allfeat2, 1, CV_32S, Scalar(0));
	cntr = 0;
	for(int i = 0; i < test_images.size(); i++){
		for(int j = 0; j < test_images[i].size(); j++){
			sft->detectAndCompute(test_images[i][j], cv::noArray(), keypoints2[cntr], descriptors2[cntr]);
			test_labels.at<int>(cntr) = i;
			cntr++;
		}
	}

	Mat vecwords2(allfeat2, num_clusters, CV_32F, Scalar(0));
	
	cntr = 0;
	for(int i = 0; i < test_images.size(); i++){
		cout << "Class " << string(argc[i+1]) << endl;
		for(int j = 0; j < test_images[i].size(); j++){
			for(int k = 0; k < descriptors2[cntr].rows; k++){
				mark_nearest_cluster(vecwords2, cntr, mcentroids, descriptors2[cntr].row(k));
			}
			cntr++;
		}
	}

	for(int i = 0; i < test_labels.rows; i++){
		int pred = svm->predict(vecwords2.row(i));
		//cout << pred << " " << labels.at<int>(i) << endl;
		confusion_matrix.at<int>(pred, test_labels.at<int>(i)) += 1;
	}
	cout << "confusion matrix" << endl;
	cout << confusion_matrix << endl;
}