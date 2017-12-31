#include <string>
#include <vector>
#include <algorithm>
#include <stdio.h>
#include <dirent.h>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "const_defs.hpp"

class Image
{
	public:
		static void loadFromFolder(std::vector<cv::Mat>& mats, std::string im_path, std::string seg_path = "");
	private:
		static void loadFF(std::vector<std::string>& fnames, std::vector<cv::Mat>& mats, std::string im_path);
		static void applySegmentation(std::vector<std::string>& fnames, std::vector<cv::Mat>& mats, std::string seg_path);

};