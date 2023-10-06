#include "image_porn.h"
#include <iostream>
#include <string>
using namespace std;

int main()
{
    const string cmd = "ffmpeg -i ../input.mp4 -vf fps=1 ../images/out%d.jpg";
    system(cmd.c_str());
    const std::string model_repository = "../src/data/models/cls_image_porn_resnext50_v1.0.0";
    const std::string image_dir = "../images";
    Image_Porn image(model_repository,image_dir);
    image.test();
    return 0;
}