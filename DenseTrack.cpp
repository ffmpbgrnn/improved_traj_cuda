#include "DenseTrackStab.h"
#include "Initialize.h"
#include "Descriptors.h"
#include "OpticalFlow.h"
#include "CUDA_RANSAC_Homography.h"
#include <time.h>
#include <sys/time.h>
#include <unistd.h>
using namespace cv;
using namespace cv::gpu;

int show_track = 0; // set show_track = 1, if you want to visualize the trajectories

void swapMat(GpuMat& prev, GpuMat& cur)
{
    GpuMat tmp(prev);
    prev = cur;
    cur = tmp;
}

int main(int argc, char** argv)
{
    struct timeval start, end;
    long secs_used, micros_used;

    gettimeofday(&start, NULL);

    gpu::setDevice(1);

    VideoCapture capture;
    char* video = argv[1];
    int flag = arg_parse(argc, argv);
    capture.open(video);

    if(!capture.isOpened()) {
        fprintf(stderr, "Could not initialize capturing..\n");
        return -1;
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

    Mat frame;
    capture >> frame;

    GpuMat d_image, d_prev_grey, d_grey;

    d_image.create(frame.size(), CV_8UC3);
    d_grey.create(frame.size(), CV_8UC1);
    d_prev_grey.create(frame.size(), CV_8UC1);

    std::vector<float> fscales(0);
    std::vector<Size> sizes(0);
    InitPry(frame, fscales, sizes);

    std::vector<GpuMat> d_prev_grey_pyr(0), d_grey_pyr(0), d_grey_warp_pyr(0);
    std::vector<GpuMat> d_flow_pyr_x(0), d_flow_pyr_y(0), 
                        d_flow_warp_pyr_x(0), d_flow_warp_pyr_y(0);

    BuildPry(sizes, CV_8UC1, d_prev_grey_pyr);
    BuildPry(sizes, CV_8UC1, d_grey_pyr);
    BuildPry(sizes, CV_32FC1, d_grey_warp_pyr);

    BuildPry(sizes, CV_32FC1, d_flow_pyr_x);
    BuildPry(sizes, CV_32FC1, d_flow_pyr_y);
    BuildPry(sizes, CV_32FC1, d_flow_warp_pyr_x);
    BuildPry(sizes, CV_32FC1, d_flow_warp_pyr_y);

    GpuMat trajMatrix[trackInfo.length];

    GpuMat trajLengthCounter(sizes[0], CV_32FC1);
    trajLengthCounter.setTo(Scalar::all(0));

    for (int i = 0; i < trackInfo.length; i++) {
        trajMatrix[i].create(sizes[i], CV_32FC2);
    }

    std::vector<std::list<Track> > xyScaleTracks;
    xyScaleTracks.resize(scale_num);


    d_image.upload(frame);
    cvtColor(d_image, d_prev_grey, CV_BGR2GRAY);

    for(int iScale = 0; iScale < scale_num; iScale++) {
        if(iScale == 0)
            d_prev_grey.copyTo(d_prev_grey_pyr[0]);
        else
            gpu::resize(d_prev_grey_pyr[iScale-1], d_prev_grey_pyr[iScale], d_prev_grey_pyr[iScale].size(), 0, 0, INTER_LINEAR);

        // dense sampling feature points
        std::vector<Point2f> points(0);
        DenseSample(d_prev_grey_pyr[iScale], points, quality, min_distance);

        // save the feature points
        std::list<Track>& tracks = xyScaleTracks[iScale];
        for(unsigned i = 0; i < points.size(); i++)
            tracks.push_back(Track(points[i], trackInfo, hogInfo, hofInfo, mbhInfo));
    }

    Mat human_mask = Mat::ones(d_image.size(), CV_8UC1);
    GpuMat d_human_mask;
    if(bb_file)
        InitMaskWithBox(human_mask, bb_list[0].BBs);
    d_human_mask.upload(human_mask);


    SURF_GPU surf;
    surf.nOctaves = 2;
    GpuMat d_prev_kpts_surf, d_prev_desc_surf;
    surf(d_prev_grey, d_human_mask, d_prev_kpts_surf, d_prev_desc_surf);

    int frame_num = 1;
    int init_counter = 0; // indicate when to detect new feature points
    
    while(true) {
        std::cout << frame_num << std::endl;
        // get a new frame
        capture >> frame;
        if(frame.empty())
            break;

        if (frame_num < start_frame) {
            frame_num++;
            continue;
        }
        if (frame_num > end_frame)
            break;

        init_counter++;

        d_image.upload(frame);
        cvtColor(d_image, d_grey, CV_BGR2GRAY);

        if(bb_file) {
            InitMaskWithBox(human_mask, bb_list[frame_num].BBs);
            d_human_mask.upload(human_mask);
        }

        #ifdef DEBUG
            std::cout << "surf..." << std::endl;
        #endif

        GpuMat d_kpts_surf, d_desc_surf;
        surf(d_grey, d_human_mask, d_kpts_surf, d_desc_surf);

        std::vector<KeyPoint> prev_kpts_surf, kpts_surf;
        surf.downloadKeypoints(d_prev_kpts_surf, prev_kpts_surf);
        surf.downloadKeypoints(d_kpts_surf, kpts_surf);

        #ifdef DEBUG
            std::cout << "Matching Surf..." << std::endl;
        #endif

        std::vector<Point2f> prev_pts_surf, pts_surf;
        ComputeMatch(prev_kpts_surf, kpts_surf, d_prev_desc_surf, d_desc_surf, prev_pts_surf, pts_surf);


        // compute optical flow for all scales once
        FarnebackOpticalFlow d_optCalc;
        d_optCalc.polyN     = 7;
        d_optCalc.polySigma = 1.5; 
        d_optCalc.winSize   = 10;
        d_optCalc.numIters  = 2;
        d_optCalc.fastPyramids = true;
        // GpuMat d_flowx, d_flowy;
        for(int iScale = 0; iScale < scale_num; iScale++) {
            if(iScale == 0)
                d_grey.copyTo(d_grey_pyr[0]);
            else
                resize(d_grey_pyr[iScale-1], d_grey_pyr[iScale], d_grey_pyr[iScale].size(), 0, 0, INTER_LINEAR);
        }
        #ifdef DEBUG
            std::cout << "Optical flow->two frames..." << std::endl;
        #endif

        for (unsigned int i = 0; i < d_prev_grey_pyr.size(); i++) {
            d_optCalc(d_prev_grey_pyr[i], d_grey_pyr[i], d_flow_pyr_x[i], d_flow_pyr_y[i]);
        }

        #ifdef DEBUG
            std::cout << "find Good feature point Optical flow..." << std::endl;
        #endif
        // Do goodFeatureToTrack here
        std::vector<Point2f> prev_pts_flow, pts_flow;
        MatchFromFlow(d_prev_grey, d_flow_pyr_x[0], d_flow_pyr_y[0], prev_pts_flow, pts_flow, d_human_mask);

        std::vector<Point2Df> prev_pts_all, pts_all;

        #ifdef DEBUG
            std::cout << "Merge SURF and Optical flow..." << std::endl;
        #endif
        MergeMatch(prev_pts_flow, pts_flow, prev_pts_surf, pts_surf, prev_pts_all, pts_all);

        #ifdef DEBUG
            std::cout << "Find Homography..." << std::endl;
        #endif

        Mat H = Mat::eye(3, 3, CV_64FC1);
        if(pts_all.size() > 50) {
            std::vector<char> match_mask;
            // Mat temp = findHomography(prev_pts_all, pts_all, RANSAC, 1, match_mask);
            // if(countNonZero(Mat(match_mask)) > 25)
            //     H = temp;
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
        }

        Mat H_inv = H.inv();
        // GpuMat d_H_inv(H_inv);
        GpuMat d_grey_warp; // = GpuMat::zeros(grey.size(), CV_8UC1);
        #ifdef DEBUG
            std::cout << "Warp..." << std::endl;
        #endif
        gpu::warpPerspective(d_prev_grey, d_grey_warp, H_inv, d_prev_grey.size());

/*************************************************************/
        
        d_grey_warp.copyTo(d_grey_warp_pyr[0]);
       /* for(int iScale = 0; iScale < scale_num; iScale++) {
            if(iScale == 0)
                d_grey_warp.copyTo(d_grey_warp_pyr[0]);
            else
                resize(d_grey_warp_pyr[iScale-1], d_grey_warp_pyr[iScale], d_grey_warp_pyr[iScale].size(), 0, 0, INTER_LINEAR);
        }*/

        #ifdef DEBUG
            std::cout << "Do Warp Optical flow..." << std::endl;
        #endif
        for (unsigned int i = 0; i < d_prev_grey_pyr.size(); i++) {
            d_optCalc(d_prev_grey_pyr[i], d_grey_warp_pyr[i], d_flow_warp_pyr_x[i], d_flow_warp_pyr_y[i]);
        }

        #ifdef DEBUG
            std::cout << "Finished Warp Optical flow..." << std::endl;
        #endif

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
                        // output the trajectory
                        // printf("%d\t%f\t%f\t%f\t%f\t%f\t%f\t", frame_num, mean_x, mean_y, var_x, var_y, length, fscales[iScale]);

                        // for spatio-temporal pyramid
                        // printf("%f\t", std::min<float>(std::max<float>(mean_x/float(seqInfo.width), 0), 0.999));
                        // printf("%f\t", std::min<float>(std::max<float>(mean_y/float(seqInfo.height), 0), 0.999));
                        // printf("%f\t", std::min<float>(std::max<float>((frame_num - trackInfo.length/2.0 - start_frame)/float(seqInfo.length), 0), 0.999));
                    
                        // output the trajectory
                        // for (int i = 0; i < trackInfo.length; ++i)
                        //     printf("%f\t%f\t", displacement[i].x, displacement[i].y);
        
                        // printf("\n");
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
            
            // std::cout << "DenseSampling new point..." << std::endl;
            DenseSample(d_grey_pyr[iScale], points, quality, min_distance);
            // save the new feature points
            for(unsigned i = 0; i < points.size(); i++)
                tracks.push_back(Track(points[i], trackInfo, hogInfo, hofInfo, mbhInfo));
        }
        #ifdef DEBUG
            std::cout << "End of Traj tracking..." << std::endl;
        #endif

        init_counter = 0;
        swapMat(d_prev_grey, d_grey);
        // d_grey.copyTo(d_prev_grey);

        // d_prev_kpts_surf = d_kpts_surf;
        swapMat(d_prev_kpts_surf, d_kpts_surf);

        // d_desc_surf.copyTo(d_prev_desc_surf);
        swapMat(d_prev_desc_surf, d_desc_surf);

        frame_num++;

        if( show_track == 1 ) {
            imshow( "DenseTrackStab", d_image);
            int c = cvWaitKey(3);
            if((char)c == 27) break;
        }
    }

    if( show_track == 1 )
        destroyWindow("DenseTrackStab");
    gettimeofday(&end, NULL);
    secs_used=(end.tv_sec - start.tv_sec); //avoid overflow by subtracting first
    micros_used= ((secs_used*1000000) + end.tv_usec) - (start.tv_usec);
    printf("micros_used: %ld\n",micros_used);

    return 0;
}
