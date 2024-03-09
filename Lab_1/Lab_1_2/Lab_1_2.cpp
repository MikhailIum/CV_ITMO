#include <iostream>
#include <string>
#include <math.h>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <sciplot/sciplot.hpp>


void plotting(cv::Mat profile){
    // setting for a plot
    sciplot::Plot2D plot;
    plot.xlabel("col");
    plot.ylabel("intensity");
    plot.legend().hide();
    // plot.ytics().hide();
    plot.xtics().hide();

    
    // adding data
    sciplot::Vec x = sciplot::linspace(0, profile.cols, profile.cols);
    std::vector<int> y;
    for (int i = 0; i < profile.cols; ++i){
        y.push_back(profile.at<float>(i));
    }

    // drawing a plot
    plot.drawCurve(x, y);

    sciplot::Figure fig = {{plot}};
    sciplot::Canvas canvas = {{fig}};

    canvas.save("result.pdf");
}

int main(){

    // loading the image
    std::string image_path = "12273fb5b555e0b3e3f21b9b240e9551.png";
    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
    
    // checking if the image was properly loaded
    if (image.empty()){
        std::cout << "Couldn't read the image " << image_path << std::endl;
        std::exit(1);
    }

    // saving original image with a specified name
    cv::imwrite("original.jpg", image);

    // calculating profile
    cv::Mat profile = image.row(image.rows / 2);

    
    plotting(profile);

    return 0;
}