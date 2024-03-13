#include <iostream>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>


int main(){
    // loading the image
    std::string filename = "original.jpg";
    cv::Mat image = cv::imread(filename, cv::IMREAD_COLOR);

    // converting to Lab and splitting
    cv::Mat image_Lab;
    cv::cvtColor(image, image_Lab, cv::COLOR_BGR2Lab);
    std::vector<cv::Mat> image_Lab_channels;
    cv::split(image_Lab, image_Lab_channels);
    cv::Mat L = image_Lab_channels[0];
    cv::Mat a = image_Lab_channels[1];
    cv::Mat b = image_Lab_channels[2];
    cv::imwrite("image_Lab.jpg", image_Lab);


    // getting color channels
    cv::Mat ab;
    cv::merge(std::vector<cv::Mat>{a, b}, ab);

    // flattening into samples
    cv::Mat samples(ab.rows * ab.cols, ab.channels(), CV_32F);
    for (int x = 0; x < ab.cols; ++x){
        for (int y = 0; y < ab.rows; ++y){
            for (int ch = 0; ch < ab.channels(); ++ch){
                samples.at<float>(x + ab.cols * y, ch) = ab.at<cv::Vec2b>(y, x)[ch];
            }
        }
    }

    int k = 9;    
    // TODO: насчитали 9 различных цветов, значит делим на 9 кластеров

    // using kmeans for samples
    cv::Mat labels;
    cv::Mat centers;
    cv::TermCriteria criteria = cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 0.01);
    cv::kmeans(samples, k, labels, criteria, 3, cv::KMEANS_PP_CENTERS, centers);

    
    // highlighting each unique color
    cv::Mat clustered_image(ab.size(), ab.type());
    for (int color_goal = 0; color_goal < k; ++color_goal) {
        cv::Mat clustered_image_cur;
        clustered_image.copyTo(clustered_image_cur);
        cv::Mat L_cur;
        L.copyTo(L_cur);

	    for (int y = 0; y < ab.rows; ++y){
		    for (int x = 0; x < ab.cols; ++x) {
			    int color = labels.at<int>(x + ab.cols * y);
			    
                for (int ch = 0; ch < ab.channels(); ++ch) {
					clustered_image_cur.at<cv::Vec2b>(y, x)[ch] = centers.at<float>(color, ch);
			    }

                if (color != color_goal){
                    int p = L_cur.at<uchar>(y, x); 
                    L_cur.at<uchar>(y, x) = (uchar)(p * 0.3);  
                }
		    }
	    }

        cv::merge(std::vector<cv::Mat>{L_cur, clustered_image_cur}, clustered_image_cur);
        cv::cvtColor(clustered_image_cur, clustered_image_cur, cv::COLOR_Lab2BGR);
	    cv::imwrite("clustered_image_" + std::to_string(color_goal) + ".jpg", clustered_image_cur);
    }

    return 0;
}