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
#ifndef _OPENCV_FUNC_H
#define _OPENCV_FUNC_H

//////////////////////////////////////////////////////////////////////////// >>
// CHANGE FOLLOWING DEFS IF NECESSARY

#define	DEFAULT_WIDTH		320		// camera frame size
#define	DEFAULT_HEIGHT		240		// camera frame size

#define	DEFAULT_FPS_INTVAL	1		// (unit: ms) flame wait interval
#define	DEFAULT_JPG_QUALITY	95

#define	THREAD_NUM			4
//#define	SPECIFY_CPUCORE

//#define	DISP_FPS

#define	WSOCKET_PORT		9001

#define	MBED_ADDR	0x3c			// NOTE: 7bit addr (correcponding 8bit addr is 0x78)

//#define CASCADE_PATH "/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml"

//////////////////////////////////////////////////////////////////////////// <<

#define WINNAME "camera"

#define	IMG_COLOR
//#define	IMG_GRAY

typedef struct {
	cv::Mat	*srcImage;
	cv::Mat	*dstImage;
	void	*params;				// other parameters
} IMG_PROCESSING_PARAM;

extern "C" {
void opencv_func_init(void *);
void opencv_func_run(IMG_PROCESSING_PARAM *);
void opencv_func_stop(void *);
}

extern int wantPreview;

#endif // _OPENCV_FUNC_H
// end of file
