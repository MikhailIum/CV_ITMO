#include <iostream>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>


// Euclidean transformation (shift)
cv::Mat shift(cv::Mat image){
    cv::Mat T = (cv::Mat_<double> (2, 3) << 
                                1, 0, 120,
                                0, 1, 300);

    cv::warpAffine(image, image, T, cv::Size(image.cols, image.rows));
    return image;
}

// Euclidean transformation (reflection)
cv::Mat reflection(cv::Mat image, int ax){
    cv::flip(image, image, ax);
    return image;
}

// affine transformation (scaling)
cv::Mat scale(cv::Mat image, int horizontal, int vertical){
    cv::resize(image, image, cv::Size(image.cols * horizontal, image.rows * vertical));
    return image;
}

// Euclidean transformation (rotation)
cv::Mat rotation(cv::Mat image, double angle){
    cv::Mat T = cv::getRotationMatrix2D(cv::Point2f(float((image.cols - 1) / 2.0), float((image.rows - 1) / 2.0)), angle, 1);
    cv::warpAffine(image, image, T, cv::Size(image.cols, image.rows));
    return image;
}

// affine transformation (bevel)
cv::Mat bevel(cv::Mat image, double strength){
    cv::Mat T = (cv::Mat_<double> (2, 3) <<
                                1, strength, 0,
                                0, 1, 0);

    cv::warpAffine(image, image, T, cv::Size(image.cols, image.rows));
    return image;
}

// affine transformation (piecewise linear transformation)
cv::Mat piecewise(cv::Mat image, double stretch){
    cv::Mat T = (cv::Mat_<double> (2, 3) <<
                                stretch, 0, 0,
                                0, 1, 0);
    cv::Mat image_right = image(cv::Rect(int(image.cols / 2), 0, image.cols - int(image.cols / 2), image.rows));
    cv::warpAffine(image_right, image_right, T, cv::Size(image_right.cols, image_right.rows));
    cv::Mat image_left = image(cv::Rect(0, 0, image.cols - int(image.cols / 2), image.rows));
    cv::flip(image_left, image_left, 1);
    return image;
}

// nonlinear transformation (projection)
cv::Mat projection(cv::Mat image){
    cv::Mat T = (cv::Mat_<double> (3, 3) <<
                                1.1, 0.2, 0.00075 ,
                                0.35, 1.1, 0.0005 ,
                                0, 0, 1);
    
    cv::warpPerspective(image, image, T, cv::Size(image.cols, image.rows));
    return image;
}

// nonlinear transformation (polynomial)
cv::Mat polynomial(cv::Mat image){
    const double T[2][6] = {{0, 1, 0, 0.00001, 0.002, 0.002}, {0, 0, 1, 0, 0, 0}};
    if (image.depth() == CV_8U){
        image.convertTo(image, CV_32F, 1.0 / 255);
    }

    std::vector<cv::Mat> image_BGR;
    cv::split(image, image_BGR);
    for (int channel = 0; channel < image_BGR.size(); ++channel){
        image = cv::Mat::zeros(image_BGR[channel].rows, image_BGR[channel].cols, image_BGR[channel].type());
        for (int col = 0; col < image_BGR[channel].cols; ++col){
            for (int row = 0; row < image_BGR[channel].rows; ++row){
                int xnew = int(round(T[0][0] + col * T[0][1] + row * T[0][2] + col * col * T[0][3] + col * row * T[0][4] + row * row * T[0][5]));
                int ynew = int(round(T[1][0] + col * T[1][1] + row * T[1][2] + col * col * T[1][3] + col * row * T[1][4] + row * row * T[1][5]));

                if (xnew >= 0 && xnew < image_BGR[channel].cols && ynew >= 0 && ynew < image_BGR[channel].rows){
                    image.at<float>(ynew, xnew) = image_BGR[channel].at<float>(row, col);
                }
            }
        }

        image_BGR[channel] = image;
    }

    cv::merge(image_BGR, image);

    image.convertTo(image, CV_8U, 255);

    return image;
}


// nonlinear transformation (sinusoidal)
cv::Mat sinusoid(cv::Mat image){
    cv::Mat u = cv::Mat::zeros(image.cols, image.rows, CV_32F);
    cv::Mat v = cv::Mat::zeros(image.cols, image.rows, CV_32F);
    for (int x = 0; x < image.cols; ++x){
        for (int y = 0; y < image.rows; ++y){
            u.at<float>(y, x) = float(x + 20 * sin (2 * M_PI * y / 90));
            v.at<float>(y, x) = float(y);
        }
    }

    cv::remap(image, image, u, v, cv::INTER_LINEAR);
    return image;
}


int main(){
    // loading the image
    std::string filename = "original.png";
    cv::Mat image = cv::imread(filename, cv::IMREAD_COLOR);

    // checking if the image was properly loaded
    if (image.empty()){
        std::cout << "Couldn't read the image " << filename << std::endl;
        exit(0);
    }

    // saving shifted image
    cv::Mat working_image;
    image.copyTo(working_image);
    cv::imwrite("shifted.png", shift(working_image));
    

    // saving reflected image OX
    image.copyTo(working_image);
    cv::imwrite("reflected_OX.png", reflection(working_image, 0));

    // saving reflected image OY
    image.copyTo(working_image);
    cv::imwrite("reflected_OY.png", reflection(working_image, 1));

    // saving reflected image OX and OY
    image.copyTo(working_image);
    cv::imwrite("reflected_OX_OY.png", reflection(working_image, -1));

    // saving scaled image 
    image.copyTo(working_image);
    cv::imwrite("scaled.png", scale(working_image, 2, 3));
    // TODO: горизонтально растянуто в 2 раза, вертикально - в 3

    // saving rotated image
    image.copyTo(working_image);
    cv::imwrite("rotated.png", rotation(working_image, 60));
    // TODO: повернуто на 60 градусов против часовой стрелки

    // saving beveled image
    image.copyTo(working_image);
    cv::imwrite("beveled.png", bevel(working_image, 0.3));
    // скос 0.3

    // saving piecewise linear transformed image
    image.copyTo(working_image);
    cv::imwrite("piecewise_linear.png", piecewise(working_image, 2));
    // TODO: правая половина растянута в 2 раза по горизонтали, левая - отражена относительно вертикальной оси


    // saving projected image
    image.copyTo(working_image);
    cv::imwrite("projected.png", projection(working_image));


    // saving polynomial image
    image.copyTo(working_image);
    cv::imwrite("polynomial.png", polynomial(working_image));

    // saving sinusoidal image
    image.copyTo(working_image);
    cv::imwrite("sinusoidal.png", sinusoid(working_image));

    return 0;
}