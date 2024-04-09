#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int main() 
{
    VideoCapture cap("video.mp4");
    cap.isOpened();
    int frame_width = cap.get(CAP_PROP_FRAME_WIDTH);
    int frame_height = cap.get(CAP_PROP_FRAME_HEIGHT);
    VideoWriter video("output_video.mp4", VideoWriter::fourcc('H', '2', '6', '4'), 10, Size(frame_width, frame_height));
    CascadeClassifier faceCascade, eyeCascade, smileCascade;
    faceCascade.load("haarcascade_frontalface_alt.xml");
    eyeCascade.load("haarcascade_eye_tree_eyeglasses.xml");
    smileCascade.load("haarcascade_smile.xml");
    Mat frame, grayFrame;
    while (cap.read(frame)) 
    {
        cvtColor(frame, grayFrame, COLOR_BGR2GRAY);
        equalizeHist(grayFrame, grayFrame);
        vector<Rect> faces, eyes, smiles;
        faceCascade.detectMultiScale(grayFrame, faces, 2, 2, 0 | CASCADE_SCALE_IMAGE, Size(50, 50));
        for (size_t i = 0; i < faces.size(); ++i) 
        {
            rectangle(frame, faces[i], Scalar(0, 255, 0), 1);
            Mat faceROI = grayFrame(faces[i]);
            eyeCascade.detectMultiScale(faceROI, eyes, 1.1, 5, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));
            smileCascade.detectMultiScale(faceROI, smiles, 2, 15, 0 | CASCADE_SCALE_IMAGE, Size(30, 30), Size(200, 200));
            for (size_t j = 0; j < eyes.size(); ++j) {
                Point eye_center(faces[i].x + eyes[j].x + eyes[j].width / 2, faces[i].y + eyes[j].y + eyes[j].height / 2);
                int radius = cvRound((eyes[j].width + eyes[j].height) * 0.25);
                circle(frame, eye_center, radius, Scalar(255, 0, 0), 1);
            }    
            for (size_t k = 0; k < smiles.size(); ++k) 
            {
                Point pt1(faces[i].x + smiles[k].x, faces[i].y + smiles[k].y);
                Point pt2(faces[i].x + smiles[k].x + smiles[k].width, faces[i].y + smiles[k].y + smiles[k].height);
                rectangle(frame, pt1, pt2, Scalar(0, 0, 255), 1);
            }
        }
        video.write(frame);
        imshow("Video", frame);
        if (waitKey(30) == 27) 
        {
            break;
        }
    }
    cap.release();
    destroyAllWindows();
    video.release();
    return 0;
}
