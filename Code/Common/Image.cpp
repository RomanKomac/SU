#include "Image.hpp"

using namespace std;
using namespace cv;


bool matchType(string ending){
	transform(ending.begin(), ending.end(), ending.begin(), ::tolower);
	vector<string> arr = __supportedImageTypes;
	for(int i = 0; i < arr.size(); i++){
		if(arr[i] == ending)
			return true;
	}
	return false;
}

bool match(string fname, vector<string> paths) {	
	for(int i = 0; i < paths.size(); i++){
		if(paths[i] == fname)
			return true;
	}
	return false;
}


void Image::loadFromFolder(vector<Mat>& mats, string im_path, string seg_path){
	vector<string> fnames;
	loadFF(fnames, mats, im_path);
	if(seg_path != ""){
		applySegmentation(fnames, mats, seg_path);
	}
}

void Image::applySegmentation(vector<string>& fnames, vector<Mat>& mats, string seg_path){
	for(int i = 0; i < mats.size(); i++){
		Mat mask = imread( (seg_path + "/" + fnames[i]), cv::IMREAD_GRAYSCALE );
		double min, max;
		minMaxLoc(mask, &min, &max);
		mask.setTo(0, mask == max);
		mask *= 255;
		Mat mmm2;
		mats[i].copyTo(mmm2, mask);
		mmm2.copyTo(mats[i]);
	}
}

void Image::loadFF(vector<string>& fnames, vector<Mat>& mats, string im_path){
	mats = vector<Mat>();

	#if defined VERBOSE
	cout << "Searching for images" << endl;
	#endif


	struct dirent *entry;
	DIR *dp;

	dp = opendir(im_path.c_str());
	if (dp == NULL) {
		cerr << "Image path does not exist or is protected." << endl;
		return;
	}

	while ((entry = readdir(dp))){
		string dname(entry->d_name);
		if(matchType(dname.substr(dname.find_last_of(".") + 1))){
			#if defined VERBOSE
			cout << "matched: " << dname << endl;
			#endif
			mats.push_back(imread( (im_path+"/"+dname), cv::IMREAD_COLOR ));
			fnames.push_back(dname.substr(0, dname.find_last_of(".")) + ".png");
		}
	}


  	#if defined DEBUG || defined VERBOSE
  	cout << "Loaded " << mats.size() << " images" << endl;
  	#endif

	closedir(dp);
	return;
}