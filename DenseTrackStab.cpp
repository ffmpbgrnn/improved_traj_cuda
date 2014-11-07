#include "DenseTrackStab.h"
#include "Initialize.h"
#include "Descriptors.h"
#include "OpticalFlow.h"

#include <time.h>

using namespace cv;

int show_track = 0; // set show_track = 1, if you want to visualize the trajectories

int main(int argc, char** argv)
{
	VideoCapture capture;
	char* video = argv[1];
	int flag = arg_parse(argc, argv);
	capture.open(video);

	if(!capture.isOpened()) {
		fprintf(stderr, "Could not initialize capturing..\n");
		return -1;
	}

	int frame_num = 0;
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

//	fprintf(stderr, "video size, length: %d, width: %d, height: %d\n", seqInfo.length, seqInfo.width, seqInfo.height);

	if(show_track == 1)
		namedWindow("DenseTrackStab", 0);


	// std::vector<KeyPoint> d_prev_kpts_surf, kpts_surf;
	GpuMat d_prev_kpts_surf, d_kpts_surf;
	GpuMat prev_desc_surf, desc_surf;
	Mat human_mask;

	GpuMat image, d_prev_grey, d_grey;

	std::vector<float> fscales(0);
	std::vector<Size> sizes(0);

	std::vector<GpuMat> d_prev_grey_pyr(0), d_grey_pyr(0), d_flow_pyr(0), d_flow_warp_pyr(0);
	std::vector<GpuMat> d_prev_poly_pyr(0), d_poly_pyr(0), d_poly_warp_pyr(0);

	std::vector<std::list<Track> > xyScaleTracks;
	int init_counter = 0; // indicate when to detect new feature points
	while(true) {
		Mat frame;
		int i, j, c;

		// get a new frame
		capture >> frame;
		if(frame.empty())
			break;

		if(frame_num < start_frame || frame_num > end_frame) {
			frame_num++;
			continue;
		}

    	GpuMat d_frame(frame);

		if(frame_num == start_frame) {
			image.create(frame.size(), CV_8UC3);
			d_grey.create(frame.size(), CV_8UC1);
			d_prev_grey.create(frame.size(), CV_8UC1);

			InitPry(frame, fscales, sizes);

			BuildPry(sizes, CV_8UC1, d_prev_grey_pyr);
			BuildPry(sizes, CV_8UC1, d_grey_pyr);
			BuildPry(sizes, CV_32FC2, d_flow_pyr);
			BuildPry(sizes, CV_32FC2, d_flow_warp_pyr);

			BuildPry(sizes, CV_32FC(5), d_prev_poly_pyr);
			BuildPry(sizes, CV_32FC(5), d_poly_pyr);
			BuildPry(sizes, CV_32FC(5), d_poly_warp_pyr);

			xyScaleTracks.resize(scale_num);

			d_frame.copyTo(image);
			cvtColor(image, d_prev_grey, CV_BGR2GRAY);

			for(int iScale = 0; iScale < scale_num; iScale++) {
				if(iScale == 0)
					d_prev_grey.copyTo(d_prev_grey_pyr[0]);
				else
					resize(d_prev_grey_pyr[iScale-1], d_prev_grey_pyr[iScale], d_prev_grey_pyr[iScale].size(), 0, 0, INTER_LINEAR);

				// dense sampling feature points
				std::vector<Point2f> points(0);
				DenseSample(d_prev_grey_pyr[iScale], points, quality, min_distance);

				// save the feature points
				std::list<Track>& tracks = xyScaleTracks[iScale];
				for(i = 0; i < points.size(); i++)
					tracks.push_back(Track(points[i], trackInfo, hogInfo, hofInfo, mbhInfo));
			}

			// compute polynomial expansion
			// my::FarnebackPolyExpPyr(prev_grey, d_prev_poly_pyr, fscales, 7, 1.5);

			human_mask = GpuMat::ones(d_frame.size(), CV_8UC1);
			if(bb_file)
				InitMaskWithBox(human_mask, bb_list[frame_num].BBs);

			/**
			 * is d_prev_kpts_surf in GPU or CPU?
			 * or use GpuMat?
			 */
    		surf(d_prev_grey, human_mask, d_prev_kpts_surf, prev_desc_surf);

			frame_num++;
			continue;
		}

		init_counter++;
		d_frame.copyTo(image);
		cvtColor(image, d_grey, CV_BGR2GRAY);

		if(bb_file)
			InitMaskWithBox(human_mask, bb_list[frame_num].BBs);


    	surf(d_grey, human_mask, d_kpts_surf, desc_surf);
    	std::vector<KeyPoint> prev_kpts_surf, kpts_surf;

    	surf.downloadKeypoints(d_prev_kpts_surf, prev_kpts_surf);
    	surf.downloadKeypoints(prev_kpts_surf, kpts_surf);

		std::vector<Point2f> prev_pts_surf, pts_surf;
		ComputeMatch(prev_kpts_surf, kpts_surf, prev_desc_surf, desc_surf, prev_pts_surf, pts_surf);

		// compute optical flow for all scales once
    	FarnebackOpticalFlow d_optCalc;
    	d_optCalc.polyN     = 7;
    	d_optCalc.polySigma = 1.5; 
    	d_optCalc.winSize   = 10;
    	d_optCalc.numIters  = 2;
    	// GpuMat d_flowx, d_flowy;
    	for (int i = 0; i < d_prev_grey_pyr.size(); i++) {
    		d_optCalc(d_prev_grey_pyr[i], d_grey_pyr[i], d_flow_pyr_x[i], d_flow_pyr_y[i]);
    	}

    	// Do goodFeatureToTrack here
		std::vector<Point2f> prev_pts_flow, pts_flow;
		MatchFromFlow(d_prev_grey, d_flow_pyr_x[0], d_flow_pyr_y[0], prev_pts_flow, pts_flow, human_mask);

		std::vector<Point2f> prev_pts_all, pts_all;

		MergeMatch(prev_pts_flow, pts_flow, prev_pts_surf, pts_surf, prev_pts_all, pts_all);

		Mat H = Mat::eye(3, 3, CV_64FC1);
		if(pts_all.size() > 50) {
			std::vector<unsigned char> match_mask;
			Mat temp = findHomography(prev_pts_all, pts_all, RANSAC, 1, match_mask);
			if(countNonZero(GpuMat(match_mask)) > 25)
				H = temp;
		}

		Mat H_inv = H.inv();
		GpuMat d_H_inv(H_inv);
		GpuMat d_grey_warp; // = GpuMat::zeros(grey.size(), CV_8UC1);
		// MyWarpPerspective(prev_grey, grey, grey_warp, H_inv); // warp the second frame
		gpu::warpPerspective(d_prev_grey, d_grey_warp, d_H_inv, d_prev_grey.size());


		// compute optical flow for all scales once
		// my::FarnebackPolyExpPyr(grey_warp, d_poly_warp_pyr, fscales, 7, 1.5);
		// my::calcOpticalFlowFarneback(d_prev_poly_pyr, d_poly_warp_pyr, d_flow_warp_pyr, 10, 2);
    	for (int i = 0; i < d_prev_grey_pyr.size(); i++) {
    		d_optCalc(d_prev_grey_pyr[i], d_poly_warp_pyr[i], d_flow_warp_pyr_x[i], d_flow_warp_pyr_y[i]);
    	}

		for(int iScale = 0; iScale < scale_num; iScale++) {
			if(iScale == 0)
				d_grey.copyTo(greflowy_pyr[0]);
			else
				resize(d_grey_pyr[iScale-1], d_grey_pyr[iScale], d_grey_pyr[iScale].size(), 0, 0, INTER_LINEAR);

			int width = d_grey_pyr[iScale].cols;
			int height = d_grey_pyr[iScale].rows;


			// track feature points in each scale separately
			std::list<Track>& tracks = xyScaleTracks[iScale];
			for (std::list<Track>::iterator iTrack = tracks.begin(); iTrack != tracks.end();) {
				int index = iTrack->index;
				Point2f prev_point = iTrack->point[index];
				int x = std::min<int>(std::max<int>(cvRound(prev_point.x), 0), width-1);
				int y = std::min<int>(std::max<int>(cvRound(prev_point.y), 0), height-1);

				Point2f point;
				point.x = prev_point.x + d_flow_pyr_x[iScale][y][x]; // .ptr<float>(y)[2*x];
				point.y = prev_point.y + d_flow_pyr_y[iScale][y][x]; //.ptr<float>(y)[2*x+1];
 
				if(point.x <= 0 || point.x >= width || point.y <= 0 || point.y >= height) {
					iTrack = tracks.erase(iTrack);
					continue;
				}

				iTrack->disp[index].x = d_flow_warp_pyr_x[iScale][y][x]; // .ptr<float>(y)[2*x];
				iTrack->disp[index].y = d_flow_warp_pyr_y[iScale][y][x]; // .ptr<float>(y)[2*x+1];

				
				iTrack->addPoint(point);

				// draw the trajectories at the first scale
				if(show_track == 1 && iScale == 0)
					DrawTrack(iTrack->point, iTrack->index, fscales[iScale], image);

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
						printf("%d\t%f\t%f\t%f\t%f\t%f\t%f\t", frame_num, mean_x, mean_y, var_x, var_y, length, fscales[iScale]);

						// for spatio-temporal pyramid
						printf("%f\t", std::min<float>(std::max<float>(mean_x/float(seqInfo.width), 0), 0.999));
						printf("%f\t", std::min<float>(std::max<float>(mean_y/float(seqInfo.height), 0), 0.999));
						printf("%f\t", std::min<float>(std::max<float>((frame_num - trackInfo.length/2.0 - start_frame)/float(seqInfo.length), 0), 0.999));
					
						// output the trajectory
						for (int i = 0; i < trackInfo.length; ++i)
							printf("%f\t%f\t", displacement[i].x, displacement[i].y);
		
						printf("\n");
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

			DenseSample(d_grey_pyr[iScale], points, quality, min_distance);
			// save the new feature points
			for(i = 0; i < points.size(); i++)
				tracks.push_back(Track(points[i], trackInfo, hogInfo, hofInfo, mbhInfo));
		}

		init_counter = 0;
		d_grey.copyTo(d_prev_grey);
		for(i = 0; i < scale_num; i++) {
			d_grey_pyr[i].copyTo(d_prev_grey_pyr[i]);
			d_poly_pyr[i].copyTo(d_prev_poly_pyr[i]);
		}

		d_prev_kpts_surf = d_kpts_surf;
		desc_surf.copyTo(prev_desc_surf);

		frame_num++;

		if( show_track == 1 ) {
			imshow( "DenseTrackStab", image);
			c = cvWaitKey(3);
			if((char)c == 27) break;
		}
	}

	if( show_track == 1 )
		destroyWindow("DenseTrackStab");
	return 0;
}
