#include "DenseTrackStab.h"
#include "Initialize.h"
#include "Descriptors.h"
#include "OpticalFlow.h"
#include "CUDA_RANSAC_Homography.h"
#include <time.h>
#include <sys/time.h>
#include <unistd.h>
#include <pthread.h>

using namespace cv;
using namespace cv::gpu;

int show_track = 0; // set show_track = 1, if you want to visualize the trajectories
int calcSize(int octave, int layer)
{
    /* Wavelet size at first layer of first octave. */
    const int HAAR_SIZE0 = 9;

    /* Wavelet size increment between layers. This should be an even number,
     such that the wavelet sizes in an octave are either all even or all odd.
     This ensures that when looking for the neighbours of a sample, the layers

     above and below are aligned correctly. */
    const int HAAR_SIZE_INC = 6;

    return (HAAR_SIZE0 + HAAR_SIZE_INC * layer) << octave;
}
void swapMat(GpuMat& prev, GpuMat& cur)
{
    GpuMat tmp(prev);
    prev = cur;
    cur = tmp;
}
typedef struct multiResize_tdata
{
    int iScale;
    GpuMat* d_grey;
    GpuMat* d_grey_pyr;
} multiResize_tdata;

void *multiResize(void *ptr)
{
    setDevice(1);
    multiResize_tdata *tdata_ptr = (multiResize_tdata *)ptr;

    int iScale = tdata_ptr->iScale;
    GpuMat& d_grey = *(tdata_ptr->d_grey);
    GpuMat& d_grey_pyr = *(tdata_ptr->d_grey_pyr);

    if(iScale == 0) {
        d_grey.copyTo(d_grey_pyr);
    } else {
        resize(d_grey, d_grey_pyr, d_grey_pyr.size(), 0, 0, INTER_LINEAR);
    }
    return (void *)0;
}

typedef struct multiDoOptCalc_tdata
{
    FarnebackOpticalFlow *d_optCalc;
    GpuMat *d_prev_grey;
    GpuMat *d_grey;
    GpuMat *d_flow_x;
    GpuMat *d_flow_y;
} multiDoOptCalc_tdata;

void* multiDoOptCalc(void *ptr)
{
    multiDoOptCalc_tdata *tdata_ptr = (multiDoOptCalc_tdata *)ptr;
    FarnebackOpticalFlow d_optCalc = *(tdata_ptr->d_optCalc);
    d_optCalc(*tdata_ptr->d_prev_grey, *tdata_ptr->d_grey, *tdata_ptr->d_flow_x, *tdata_ptr->d_flow_y);
    return (void *)0;
}
int argc;
char **argv;
void *worker(void *args)
{
    struct timeval start, end;
    long secs_used,micros_used;

    Stream& streams = *(Stream *)args;
    gettimeofday(&start, NULL);
    setDevice(1);
    VideoCapture capture;
    char* video = argv[1];
    int flag = arg_parse(argc, argv);
    capture.open(video);

    if(!capture.isOpened()) {
        fprintf(stderr, "Could not initialize capturing..\n");
        return NULL;
    }

    TrackInfo trackInfo;
    DescInfo hogInfo, hofInfo, mbhInfo;

    InitTrackInfo(&trackInfo, track_length, init_gap);
    InitDescInfo(&hogInfo, 8, false, patch_size, nxy_cell, nt_cell);
    InitDescInfo(&hofInfo, 9, true, patch_size, nxy_cell, nt_cell);
    InitDescInfo(&mbhInfo, 8, false, patch_size, nxy_cell, nt_cell);

    SeqInfo seqInfo;
    InitSeqInfo(&seqInfo, video);

    std::vector<Frame> bb_list;
    if(bb_file) {
        LoadBoundBox(bb_file, bb_list);
        assert(bb_list.size() == seqInfo.length);
    }

    if(flag)
        seqInfo.length = end_frame - start_frame + 1;

//  fprintf(stderr, "video size, length: %d, width: %d, height: %d\n", seqInfo.length, seqInfo.width, seqInfo.height);

    if(show_track == 1)
        namedWindow("DenseTrackStab", 0);


    // std::vector<KeyPoint> d_prev_kpts_surf, kpts_surf;
    GpuMat d_prev_kpts_surf;
    GpuMat d_prev_desc_surf;
    Mat human_mask;
    GpuMat d_human_mask;

    GpuMat d_image, d_prev_grey, d_grey;

    std::vector<float> fscales(0);
    std::vector<Size> sizes(0);

    std::vector<GpuMat> d_prev_grey_pyr(0), d_grey_pyr(0), d_grey_warp_pyr(0);
    std::vector<GpuMat> d_flow_pyr_x(0), d_flow_pyr_y(0),
                        d_flow_warp_pyr_x(0), d_flow_warp_pyr_y(0);

    std::vector<std::list<Track> > xyScaleTracks;
    int init_counter = 0; // indicate when to detect new feature points

    SURF_GPU surf;
    surf.nOctaves = 2;
    int frame_num = 0;
    Stream work_stream[scale_num];
    int64 startTime, endTime;
    double frequency = cv::getTickFrequency();
    const int micro = 1000*1000;

    int64 timeSum = 0;
    int64 timeCount = 0;
    int64 TimeUpload = 0;
    int64 TimeSurf_Step_1 = 0;
    int64 TimeSurfDownloadKeyPoint = 0;
    int64 TimeComputeMatch = 0;
    int64 TimeResize_Step_1 = 0;
    int64 TimeOpticalFlow_Step_1 = 0;
    int64 TimeMatchFlow = 0;
    int64 TimeMergeFlow = 0;
    int64 TimeFindHomography = 0;
    int64 TimeWarpPerspective = 0;
    int64 TimeResize_Step_2 = 0;
    int64 TimeOpticalFlow_Step_2 = 0;
    int64 TimeDenseSample = 0;
    int64 TimeCheckTraj = 0;
    int64 TimeReadFrame = 0;
    int64 TimeEachRound = 0;

    BroxOpticalFlow d_flow(0.197, 50, 0.5, 10, 77, 10);
    while(true) {
        int64 eachRoundStartTime = cv::getTickCount();
        Mat frame;
        std::cout << frame_num << std::endl;
        startTime = cv::getTickCount();
		capture >> frame;
        endTime = cv::getTickCount();

        if(frame.empty())
            break;

        if(frame_num < start_frame || frame_num > end_frame) {
            frame_num++;
            continue;
        }

        if(frame_num == start_frame) {
            d_image.create(frame.size(), CV_8UC3);
            d_grey.create(frame.size(), CV_8UC1);
            d_prev_grey.create(frame.size(), CV_8UC1);

            InitPry(frame, fscales, sizes);

            BuildPry(sizes, CV_8UC1, d_prev_grey_pyr);
            BuildPry(sizes, CV_8UC1, d_grey_pyr);
            BuildPry(sizes, CV_32FC1, d_grey_warp_pyr);

            BuildPry(sizes, CV_32FC1, d_flow_pyr_x);
            BuildPry(sizes, CV_32FC1, d_flow_pyr_y);
            BuildPry(sizes, CV_32FC1, d_flow_warp_pyr_x);
            BuildPry(sizes, CV_32FC1, d_flow_warp_pyr_y);
            xyScaleTracks.resize(scale_num);

            d_image.upload(frame);
//            d_frame.copyTo(d_image);

            cvtColor(d_image, d_prev_grey, CV_BGR2GRAY, streams);
            for(int iScale = 0; iScale < scale_num; iScale++) {
                if(iScale == 0)
                    d_prev_grey.copyTo(d_prev_grey_pyr[0]);
                else
                    resize(d_prev_grey_pyr[iScale-1], d_prev_grey_pyr[iScale], d_prev_grey_pyr[iScale].size(), 0, 0, INTER_LINEAR, streams);

                // dense sampling feature points
                std::vector<Point2f> points(0);
                DenseSample(d_prev_grey_pyr[iScale], points, quality, min_distance);

                // save the feature points
                std::list<Track>& tracks = xyScaleTracks[iScale];
                for(unsigned i = 0; i < points.size(); i++)
                    tracks.push_back(Track(points[i], trackInfo, hogInfo, hofInfo, mbhInfo));
            }

            human_mask = Mat::ones(d_image.size(), CV_8UC1);
            if(bb_file)
                InitMaskWithBox(human_mask, bb_list[frame_num].BBs);
            d_human_mask.upload(human_mask);

            surf(d_prev_grey, d_human_mask, d_prev_kpts_surf, d_prev_desc_surf, streams);
            frame_num++;
            continue;
        }
        TimeReadFrame += (endTime - startTime);

        init_counter++;
        timeCount++;

        startTime = cv::getTickCount();
            d_image.upload(frame);
            cvtColor(d_image, d_grey, CV_BGR2GRAY);
        endTime = cv::getTickCount();
        TimeUpload += (endTime - startTime);

        if(bb_file) {
            InitMaskWithBox(human_mask, bb_list[frame_num].BBs);
            d_human_mask.upload(human_mask);
        }

        GpuMat d_kpts_surf, d_desc_surf;

        startTime = cv::getTickCount();
            surf(d_grey, d_human_mask, d_kpts_surf, d_desc_surf);
            printf("key:%d\n", d_kpts_surf.cols);
        endTime = cv::getTickCount();
        TimeSurf_Step_1 += (endTime - startTime);

        std::vector<KeyPoint> prev_kpts_surf, kpts_surf;

        startTime = cv::getTickCount();
            surf.downloadKeypoints(d_prev_kpts_surf, prev_kpts_surf);
            surf.downloadKeypoints(d_kpts_surf, kpts_surf); // all 100us
        endTime = cv::getTickCount();
        TimeSurfDownloadKeyPoint += (endTime - startTime);

        // std::cout << prev_kpts_surf.size() << " " << kpts_surf.size() << std::endl;

        std::vector<Point2f> prev_pts_surf, pts_surf;
        startTime = cv::getTickCount();
            ComputeMatch(prev_kpts_surf, kpts_surf, d_prev_desc_surf, d_desc_surf, prev_pts_surf, pts_surf, streams); // 7500us
        endTime = cv::getTickCount();
        TimeComputeMatch += (endTime - startTime);

        // 65us
        startTime = cv::getTickCount();
            for(int iScale = 0; iScale < scale_num; iScale++) {
                if(iScale == 0)
                    d_grey.copyTo(d_grey_pyr[0]);
                else {
                    resize(d_grey_pyr[iScale-1], d_grey_pyr[iScale], d_grey_pyr[iScale].size(), 0, 0, INTER_LINEAR, streams);
                }
            }
        endTime = cv::getTickCount();
        TimeResize_Step_1 += (endTime - startTime);


        // 10000us
        startTime = cv::getTickCount();
        for (unsigned int i = 0; i < d_prev_grey_pyr.size(); i++) {
            FarnebackOpticalFlow d_optCalc;
            d_optCalc.polyN     = 7;
            d_optCalc.polySigma = 1.5;
            d_optCalc.winSize   = 10;
            d_optCalc.numIters  = 2;
            d_optCalc.numLevels = 1;
            d_optCalc.fastPyramids = true;
            d_optCalc(d_prev_grey_pyr[i], d_grey_pyr[i], d_flow_pyr_x[i], d_flow_pyr_y[i], streams);
        }
        endTime = cv::getTickCount();
        TimeOpticalFlow_Step_1 += (endTime - startTime);

        std::vector<Point2f> prev_pts_flow, pts_flow;
        startTime = cv::getTickCount();
            MatchFromFlow(d_prev_grey, d_flow_pyr_x[0], d_flow_pyr_y[0], prev_pts_flow, pts_flow, d_human_mask, streams);
        endTime = cv::getTickCount();
        TimeMatchFlow += (endTime - startTime);

        std::vector<Point2f> prev_pts_all, pts_all;
        startTime = cv::getTickCount();
            MergeMatch(prev_pts_flow, pts_flow, prev_pts_surf, pts_surf, prev_pts_all, pts_all, streams);
        endTime = cv::getTickCount();
        TimeMergeFlow += (endTime - startTime);

	std::cout << pts_surf.size() << " " << pts_flow.size() << " " << pts_all.size() << std::endl;
        startTime = cv::getTickCount();
            Mat H = Mat::eye(3, 3, CV_64FC1);
            /*if(pts_all.size() > 50) {
                std::vector<char> match_mask;
                // Mat temp = findHomography(prev_pts_all, pts_all, RANSAC, 1, match_mask);
                const double CONFIDENCE = 0.99;
                const double INLIER_RATIO = 0.18; // Assuming lots of noise in the data!
                const double INLIER_THRESHOLD = 3.0; // pixel distance
                int K = (int)(log(1.0 - CONFIDENCE) / log(1.0 - pow(INLIER_RATIO, 4.0)));

                float best_H[9];
                std::cout << prev_pts_all.size() << " " << pts_all.size() << std::endl;
                CUDA_RANSAC_Homography(prev_pts_all, pts_all, INLIER_THRESHOLD, K, best_H, &match_mask);
                for (int c = 0; c < 3; c++) {
                    for (int r = 0; r < 3; r++)
                        H.ptr<float>(c)[r] = best_H[r * 3 + c];
                }
                if(countNonZero(Mat(match_mask)) > 25)
                    H = temp;
            }*/
	    if(pts_all.size() > 50) {
		std::vector<unsigned char> match_mask;
		Mat temp = findHomography(prev_pts_all, pts_all, RANSAC, 1, match_mask);
		if(countNonZero(Mat(match_mask)) > 25)
			H = temp;
                else
			printf("Find Failed\n");
	    }
	    printf("size: %d\n", pts_all.size());

            Mat H_inv = H.inv();
        endTime = cv::getTickCount();
        TimeFindHomography += (endTime - startTime);

        startTime = cv::getTickCount();
            GpuMat d_grey_warp; // = GpuMat::zeros(grey.size(), CV_8UC1);
            gpu::warpPerspective(d_prev_grey, d_grey_warp, H_inv, d_prev_grey.size(), streams);
        endTime = cv::getTickCount();
        TimeWarpPerspective += (endTime - startTime);

        startTime = cv::getTickCount();
            for(int iScale = 0; iScale < scale_num; iScale++) {
                if(iScale == 0)
                    d_grey_warp.copyTo(d_grey_warp_pyr[0]);
                else {
                    resize(d_grey_warp_pyr[iScale-1], d_grey_warp_pyr[iScale], d_grey_warp_pyr[iScale].size(), 0, 0, INTER_LINEAR, streams);
                }
            }
        endTime = cv::getTickCount();
        TimeResize_Step_2 += (endTime - startTime);

        /// TimeOpticalFlow_Step_2
        startTime = cv::getTickCount();
            for (unsigned int i = 0; i < d_prev_grey_pyr.size(); i++) {
                FarnebackOpticalFlow d_optCalc;
                d_optCalc.polyN     = 7;
                d_optCalc.polySigma = 1.5;
                d_optCalc.winSize   = 10;
                d_optCalc.numIters  = 2;
                d_optCalc.numLevels = 1;
                d_optCalc.fastPyramids = true;
                d_optCalc(d_prev_grey_pyr[i], d_grey_warp_pyr[i], d_flow_warp_pyr_x[i], d_flow_warp_pyr_y[i], streams); // , work_stream[i]);
            }
        endTime = cv::getTickCount();
        TimeOpticalFlow_Step_2 += (endTime - startTime);


        startTime = cv::getTickCount();
        for(int iScale = 0; iScale < scale_num; iScale++) {

            int width = d_grey_pyr[iScale].cols;
            int height = d_grey_pyr[iScale].rows;

            Mat flow_x(d_flow_pyr_x[iScale]), flow_y(d_flow_pyr_y[iScale]);
            Mat flow_warp_x(d_flow_warp_pyr_x[iScale]), flow_warp_y(d_flow_warp_pyr_y[iScale]);

            // track feature points in each scale separately
            // std::cout << "Checking validation of traj" << std::endl;
            std::list<Track>& tracks = xyScaleTracks[iScale];
            // std::cout << "Scale: " << iScale << " Track size: " <<  tracks.size() << std::endl;
            for (std::list<Track>::iterator iTrack = tracks.begin(); iTrack != tracks.end();) {
                int index = iTrack->index;
                Point2f prev_point = iTrack->point[index];
                int x = std::min<int>(std::max<int>(cvRound(prev_point.x), 0), width-1);
                int y = std::min<int>(std::max<int>(cvRound(prev_point.y), 0), height-1);

                Point2f point;
                point.x = prev_point.x + flow_x.ptr<float>(y)[x];
                point.y = prev_point.y + flow_y.ptr<float>(y)[x];

                if(point.x <= 0 || point.x >= width || point.y <= 0 || point.y >= height) {
                    iTrack = tracks.erase(iTrack);
                    // std::cout << "point overflow" << std::endl;
                    continue;
                }
                    // std::cout << "Checking validation of traj" << std::endl;

                iTrack->disp[index].x = flow_warp_x.ptr<float>(y)[x];
                iTrack->disp[index].y = flow_warp_y.ptr<float>(y)[x];

                iTrack->addPoint(point);

                // draw the trajectories at the first scale
                if(show_track == 1 && iScale == 0) {
                    Mat image;
                    d_image.download(image);
                    DrawTrack(iTrack->point, iTrack->index, fscales[iScale], image);
                }

                // if the trajectory achieves the maximal length
                if(iTrack->index >= trackInfo.length) {
                    std::vector<Point2f> trajectory(trackInfo.length+1);
                    for(int i = 0; i <= trackInfo.length; ++i)
                        trajectory[i] = iTrack->point[i]*fscales[iScale];

                    std::vector<Point2f> displacement(trackInfo.length);
                    for (int i = 0; i < trackInfo.length; ++i)
                        displacement[i] = iTrack->disp[i]*fscales[iScale];

                    float mean_x(0), mean_y(0), var_x(0), var_y(0), length(0);
                    if(IsValid(trajectory, mean_x, mean_y, var_x, var_y, length) && IsCameraMotion(displacement)) {

                    }

                    iTrack = tracks.erase(iTrack);
                    continue;
                }
                ++iTrack;
            }

            if(init_counter != trackInfo.gap)
                continue;

            // detect new feature points every gap frames
            std::vector<Point2f> points(0);
            for(std::list<Track>::iterator iTrack = tracks.begin(); iTrack != tracks.end(); iTrack++)
                points.push_back(iTrack->point[iTrack->index]);

            int64 s = cv::getTickCount();
                DenseSample(d_grey_pyr[iScale], points, quality, min_distance);
            int64 e = cv::getTickCount();
            TimeDenseSample += (e - s);
            // save the new feature points
            for(unsigned i = 0; i < points.size(); i++)
                tracks.push_back(Track(points[i], trackInfo, hogInfo, hofInfo, mbhInfo));
        }
        endTime = cv::getTickCount();
        TimeCheckTraj += (endTime - startTime);

        init_counter = 0;

        swapMat(d_prev_grey, d_grey);
        swapMat(d_prev_kpts_surf, d_kpts_surf);
        swapMat(d_prev_desc_surf, d_desc_surf);

        frame_num++;
        int64 eachRoundEndTime = cv::getTickCount();
        TimeEachRound += (eachRoundEndTime - eachRoundStartTime);
    }

    gettimeofday(&end, NULL);
    secs_used=(end.tv_sec - start.tv_sec); //avoid overflow by subtracting first
    micros_used= ((secs_used*1000000) + end.tv_usec) - (start.tv_usec);
    printf("micros_used: %ld\n",micros_used);
    printf("GREY SIZE: %d, %d\n", d_grey.rows, d_grey.cols);
    printf("TimeEachRound: %lf\n", 1.0 * TimeEachRound / timeCount / frequency*micro);
    printf("TimeReadFrame: %lf\n", 1.0 * TimeReadFrame / timeCount / frequency*micro);
    printf("TimeUpload: %lf\n", 1.0 * TimeUpload / timeCount / frequency*micro);
    printf("TimeSurf_Step_1: %lf\n", 1.0 * TimeSurf_Step_1 / timeCount / frequency*micro);
    printf("TimeSurfDownloadKeyPoint: %lf\n", 1.0 * TimeSurfDownloadKeyPoint / timeCount / frequency*micro);
    printf("TimeComputeMatch: %lf\n", 1.0 * TimeComputeMatch / timeCount / frequency*micro);
    printf("TimeResize_Step_1: %lf\n", 1.0 * TimeResize_Step_1 / timeCount / frequency*micro);
    printf("TimeOpticalFlow_Step_1: %lf\n", 1.0 * TimeOpticalFlow_Step_1 / timeCount / frequency*micro);
    printf("TimeMatchFlow: %lf\n", 1.0 * TimeMatchFlow / timeCount / frequency*micro);
    printf("TimeMergeFlow: %lf\n", 1.0 * TimeMergeFlow / timeCount / frequency*micro);
    printf("TimeFindHomography: %lf\n", 1.0 * TimeFindHomography / timeCount / frequency*micro);
    printf("TimeWarpPerspective: %lf\n", 1.0 * TimeWarpPerspective / timeCount / frequency*micro);
    printf("TimeResize_Step_2: %lf\n", 1.0 * TimeResize_Step_2 / timeCount / frequency*micro);
    printf("TimeOpticalFlow_Step_2: %lf\n", 1.0 * TimeOpticalFlow_Step_2 / timeCount / frequency*micro);
    printf("TimeDenseSample: %lf\n", 1.0 * TimeDenseSample / scale_num / timeCount / frequency*micro);
    printf("TimeCheckTraj: %lf\n", 1.0 * TimeCheckTraj / timeCount / frequency*micro);

    return 0;
}


int main(int argc_m, char **argv_m)
{
    argc = argc_m;
    argv = argv_m;
    const int STREAM_NUM = 15;
    pthread_t thread_list[STREAM_NUM];
    int id[15];
    setDevice(1);
    Stream streams[STREAM_NUM];

    for (int i = 0; i < STREAM_NUM; i++)
        id[i] = i;
    for (int i = 0; i < 2; i++)
        pthread_create(&thread_list[i], NULL, worker, &streams[i]);
    for (int i = 0; i < 2; i++)
        pthread_join(thread_list[i], NULL);

}
