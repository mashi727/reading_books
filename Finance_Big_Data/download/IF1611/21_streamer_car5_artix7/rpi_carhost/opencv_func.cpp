/*
 * Copyright (c) 2015-2016, Sumio Morioka
 * All rights reserved.
 *
 * This source code was originally written by Dr.Sumio Morioka for use in the 2015-2016 issues of 
 * "the Interface magazine", published by CQ publishing Co.Ltd in Japan (http://www.cqpub.co.jp).
 * The author has no responsibility on any results caused by using this code.
 *
 * - Author: Dr.Sumio Morioka (http://www002.upp.so-net.ne.jp/morioka, FB:Sumio Morioka)
 *
 *
 * IMPORTANT NOTICE:
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *   * Neither the name of the copyright holder nor the
 *     names of its contributors may be used to endorse or promote products
 *     derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <time.h>

#include "cv.h"
#include "highgui.h"
//using namespace cv;		// comment in if preferable

#include "opencv_func.h"
#include "websocket_flags.h"

//////////////////////////////////////////////////////////////////////////// >>
// ADD YOUR ORIGINAL CODE HERE
// any functions/global variables
//////////////////////////////////////////////////////////////////////////// <<

void opencv_func_init(void *)
{
	if (wantPreview != 0) {
		cv::namedWindow(WINNAME, CV_WINDOW_AUTOSIZE);
	}

//////////////////////////////////////////////////////////////////////////// >>
// ADD YOUR ORIGINAL CODE HERE
// preparation for image processing
//////////////////////////////////////////////////////////////////////////// <<
}


void opencv_func_run(IMG_PROCESSING_PARAM *param)
{
//////////////////////////////////////////////////////////////////////////// >>
// ADD YOUR ORIGINAL CODE HERE
// image processing main part
	{
		int src_x	= (*(param->srcImage)).cols;
		int src_y	= (*(param->srcImage)).rows;

		time_t		t;
		struct tm	*t_st;
		char		strbuf[256];

		(*(param->srcImage)).copyTo(*(param->dstImage));

		cv::Mat	rotmat;
		rotmat.create(cv::Size(3, 2), CV_32FC1);
		rotmat	= cv::getRotationMatrix2D(cv::Point(src_x / 2, src_y / 2), 180, 1);					// center, angle, scale
		cv::warpAffine(*(param->srcImage), *(param->dstImage), rotmat, cv::Size(src_x, src_y));		// rotate

//		time(&t);
//		t_st	= localtime(&t);
//
//		sprintf(strbuf, "%d/%d/%d %02d:%02d:%02d", 
//						t_st->tm_year + 1900, t_st->tm_mon + 1, t_st->tm_mday,
//						t_st->tm_hour, t_st->tm_min, t_st->tm_sec);

//		sprintf(strbuf, "R cnt %d",
//					receive_serial16(0x0B)
//				);
//
//		cv::putText(*(param->dstImage), strbuf, cv::Point(20,20), cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(0,0,0), 1, CV_AA);
//
//		sprintf(strbuf, "L cnt %d",
//					receive_serial16(0x0C)
//				);
//
//		cv::putText(*(param->dstImage), strbuf, cv::Point(20,40), cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(0,0,0), 1, CV_AA);

		int	v1, v2;
		i2c_read_word(FPGA_I2C_ADDR, 0x83, &v1);
		i2c_read_word(FPGA_I2C_ADDR, 0x81, &v2);
		sprintf(strbuf, "Dir val %d.%d tar %d.%d",
					v1 / 100, v1 % 100,
					v2 / 100, v2 % 100
				);

		cv::putText(*(param->dstImage), strbuf, cv::Point(20,20), cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(0,0,0), 1, CV_AA);
	}
//////////////////////////////////////////////////////////////////////////// <<
}


void opencv_func_stop(void *)
{
	if (wantPreview != 0) {
		cv::destroyWindow(WINNAME);
	}

//////////////////////////////////////////////////////////////////////////// >>
// ADD YOUR ORIGINAL CODE HERE
// release resources for program termination
//////////////////////////////////////////////////////////////////////////// <<
}

// end of file
