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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <getopt.h>
#include <libwebsockets.h>
#include <linux/i2c-dev.h>
#include <math.h>
#include <memory.h>
#include <pthread.h>
#include <semaphore.h>
#include <signal.h>
#include <sysexits.h>
#include <syslog.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <termios.h>
#include <time.h>
#include <unistd.h>

//////////////////////////////////////////// >> mjpg-streamer (source r182)
//#include <linux/types.h>          /* for videodev2.h */
//#include <linux/videodev2.h>
#include "mjpg_streamer.h"
#include "utils.h"
#define INPUT_PLUGIN_NAME "raspicam + opencv input plugin "
#define MAX_ARGUMENTS 32

extern "C" {
int input_init(input_parameter *param, int);
int input_stop(int);
int input_run(int);
}
//////////////////////////////////////////// << mjpg-streamer (source r182)

#include "cv.h"
#include "highgui.h"

#include "bcm_host.h"
#include "interface/vcos/vcos.h"

#include "interface/mmal/mmal.h"
//#include "interface/mmal/mmal_logging.h"
#include "interface/mmal/mmal_buffer.h"
#include "interface/mmal/util/mmal_util.h"
#include "interface/mmal/util/mmal_util_params.h"
#include "interface/mmal/util/mmal_default_components.h"
#include "interface/mmal/util/mmal_connection.h"

#include "RaspiPreview.h"
#include "RaspiCLI.h"
//#include "RaspiCamControl.h"
#include "RaspiCamControl.hpp"

#define	DEV_I2C		"/dev/i2c-1"
#define	DEV_MEM		"/dev/mem"
#define	DEV_SERIAL	"/dev/ttyAMA0"
#define	SERIAL_RATE	B9600

#define MEM_PAGESIZ	4096
#define MEM_BLKSIZ	4096

#define PERIPHERAL_BASEADR	0x20000000
#define GPIO_BASEADR		(PERIPHERAL_BASEADR + 0x200000)
#define PWM_BASEADR			(PERIPHERAL_BASEADR + 0x20C000)
#define CLOCK_BASEADR		(PERIPHERAL_BASEADR + 0x101000)

#define	PWM_CTL		0
#define	PWM_RNG1	4
#define	PWM_DAT1	5
#define	PWMCLK_CNTL	40
#define	PWMCLK_DIV	41

volatile unsigned int *ioadr_gpio;
volatile unsigned int *ioadr_clk;
volatile unsigned int *ioadr_pwm;

static long			fps_intval;

struct timespec		tspec;
long				t0_in, t1_in, t0_out, t1_out;
int					tdiff;
long				tall	= 0;
int					fnum	= 0;

#include "opencv_func.h"		// DO NOT MOVE THIS LINE

// Standard port setting for the camera component
#define MMAL_CAMERA_PREVIEW_PORT	0
#define MMAL_CAMERA_VIDEO_PORT		1
#define MMAL_CAMERA_CAPTURE_PORT	2

// Video format information
#define VIDEO_FRAME_RATE_NUM	30
#define VIDEO_FRAME_RATE_DEN	1

/// Video render needs at least 2 buffers.
#define VIDEO_OUTPUT_BUFFERS_NUM	3

int mmal_status_to_int(MMAL_STATUS_T status);

// Structure containing all state information for the current run
typedef struct {
   int	width;
   int	height;
   int	framerate;

   RASPICAM_CAMERA_PARAMETERS	camera_parameters;	// Camera setup parameters
   MMAL_COMPONENT_T				*camera_component;	// Pointer to the camera component
   MMAL_POOL_T					*video_pool;		// Pointer to the pool of buffers used by encoder output port
} RASPIVID_STATE;

// Struct used to pass information in encoder port userdata to callback
typedef struct {
	VCOS_SEMAPHORE_T	complete_semaphore;	// semaphore which is posted when we reach end of frame (indicates end of capture or fault)
	RASPIVID_STATE		*pstate;           	// pointer to our state in case required in callback
} PORT_USERDATA;


typedef struct _thread_arg {
	int				thread_id;

	RASPIVID_STATE	*rasp_state;

	pthread_t		th;
	volatile char	th_inrdy;		// NOTE: use volatile
	volatile char	th_outrdy;		// NOTE: use volatile
	pthread_mutex_t	th_rdyflag;

	cv::Mat			th_in;
	cv::Mat			th_out;
	void			*params;		// other parameters
} THREAD_ARG;


THREAD_ARG	th_arg_G[THREAD_NUM];

static int	next_in_thread_G	= 0;
static int	next_out_thread_G	= 0;


////////////////////////////////////////////////////////
// mjpg-streamer (source r182)
////////////////////////////////////////////////////////

/* private functions and variables to this plugin */
static pthread_t   worker;
static globals     *pglobal;

static void *worker_thread(void *);
static void worker_cleanup(void *);
static void help(void);

static int plugin_number;
static int width		= DEFAULT_WIDTH;
static int height		= DEFAULT_HEIGHT;
static float fps		= (1000.0 / (float)DEFAULT_FPS_INTVAL);
static int quality		= DEFAULT_JPG_QUALITY;
int wantPreview	= 0;


////////////////////////////////////////////////////////
// websockets
////////////////////////////////////////////////////////

static int	wsocket_port	= WSOCKET_PORT;

static int wsocket_callback_http(struct libwebsocket_context *cntxt,
                         struct libwebsocket *wsi,
                         enum libwebsocket_callback_reasons reason,
                         void *user,
                         void *inp,
                         size_t len)
{
	return 0;
}


#include "websocket_func.h"		// DO NOT MOVE THIS LINE


static struct libwebsocket_protocols protocols[] = {
    /* first protocol must always be HTTP handler */
    {
        "http-only",			// name
        wsocket_callback_http,	// callback
        0              			// per_session_data_size
    },
    {
        "dumb-increment-protocol", 			// protocol name - very important!
        wsocket_callback_dumb_increment,   	// callback
        0                          			// we don't use any per session data
    },
    {
        NULL, NULL, 0   /* End of list */
    }
};

void *wsocket_thread_func(void *arg)
{
	const char	*interface = NULL;
	const char	*cert_path = NULL;
	const char	*key_path = NULL;
    int			opts = 0;
	struct lws_context_creation_info	info;
	struct libwebsocket_context			*context;

	memset(&info, 0, sizeof info);
	info.port		= wsocket_port;
	info.iface		= interface;
	info.protocols	= protocols;
	info.extensions	= libwebsocket_get_internal_extensions();
	info.gid		= -1;
	info.uid		= -1;
	info.options	= opts;
	info.ssl_cert_filepath			= NULL;
	info.ssl_private_key_filepath	= NULL;

	// https://github.com/awm086/libwebsocket/blob/master/README.coding
	//194 TCP Keepalive 
	//195 ------------- 
	//196 
	// 
	//197 It is possible for a connection which is not being used to send to die 
	//198 silently somewhere between the peer and the side not sending.  In this case 
	//199 by default TCP will just not report anything and you will never get any more 
	//200 incoming data or sign the link is dead until you try to send. 
	//201 
	// 
	//202 To deal with getting a notification of that situation, you can choose to 
	//203 enable TCP keepalives on all libwebsockets sockets, when you create the 
	//204 context. 
	//205 
	// 
	//206 To enable keepalive, set the ka_time member of the context creation parameter 
	//207 struct to a nonzero value (in seconds) at context creation time.  You should 
	//208 also fill ka_probes and ka_interval in that case. 
	//209 
	// 
	//210 With keepalive enabled, the TCP layer will send control packets that should 
	//211 stimulate a response from the peer without affecting link traffic.  If the 
	//212 response is not coming, the socket will announce an error at poll() forcing 
	//213 a close. 
	//214 
	// 
	//215 Note that BSDs don't support keepalive time / probes / inteveral per-socket 
	//216 like Linux does.  On those systems you can enable keepalive by a nonzero 
	//217 value in ka_time, but the systemwide kernel settings for the time / probes/ 
	//218 interval are used, regardless of what nonzero value is in ka_time. 

	// https://libwebsockets.org/libwebsockets-api-doc.html
	// ka_time 0 for no keepalive, otherwise apply this keepalive timeout to all 
	//		libwebsocket sockets, client or server 
	// ka_probes if ka_time was nonzero, after the timeout expires how many times to try 
	//		to get a response from the peer before giving up and killing the connection 
	// ka_interval if ka_time was nonzero, how long to wait before each ka_probes attempt 
	//		provided_client_ssl_ctx 
	info.ka_time		= 60;
	info.ka_probes		= 0;
	info.ka_interval	= 0;

	context = libwebsocket_create_context(&info);

	if (context == NULL) {
		fprintf(stderr, "libwebsocket init failed\n");
		return (void *)NULL;
	}
	else {
		fprintf(stderr, "starting websockets server\n");
	}

	while (1) {
        libwebsocket_service(context, 10);		// 2nd parameter; wait time

	}

//	libwebsocket_context_destroy(context);
//	return (void *)NULL;
}


////////////////////////////////////////////////////////
// I2C
////////////////////////////////////////////////////////

void i2c_write(char dev_adr, char reg_adr, char data)
{
	int		hd, st;
	char	wbuf[2];

	if ((hd = open("/dev/i2c-1", O_RDWR)) < 0) {
		fprintf(stderr, "cannot open /dev/i2c-1\n");
		return;
	}
	if ((st = ioctl(hd, I2C_SLAVE, dev_adr)) < 0) {
		fprintf(stderr, "cannot select device 0x%x\n", dev_adr);
		return;
	}

	wbuf[0]	= reg_adr;
	wbuf[1]	= data;
	write(hd, wbuf, 2);

	close(hd);
}


void i2c_read(char dev_adr, char reg_adr, char *data)
{
	int	hd, st;
	char	wbuf;

	if ((hd = open("/dev/i2c-1", O_RDWR)) < 0) {
		fprintf(stderr, "cannot open /dev/i2c-1\n");
		return;
	}
	if ((st = ioctl(hd, I2C_SLAVE, dev_adr)) < 0) {
		fprintf(stderr, "cannot select device 0x%x\n", dev_adr);
		return;
	}

	wbuf	= reg_adr;
	write(hd, &wbuf, 1);

//	st	= ioctl(hd, I2C_SLAVE, dev_adr + 1);	// read address

	read(hd, data, 1);

	close(hd);
}


void i2c_read_word(char dev_adr, char reg_adr, int *data)
{
	int		hd, st;
	char	rbuf[2];
	char	wbuf;

	if ((hd = open("/dev/i2c-1", O_RDWR)) < 0) {
		fprintf(stderr, "cannot open /dev/i2c-1\n");
		return;
	}
	if ((st = ioctl(hd, I2C_SLAVE, dev_adr)) < 0) {
		fprintf(stderr, "cannot select device 0x%x\n", dev_adr);
		return;
	}

	wbuf	= reg_adr;
	write(hd, &wbuf, 1);

//	st	= ioctl(hd, I2C_SLAVE, dev_adr + 1);	// read address

	read(hd, rbuf, 2);

	*data	= rbuf[0] * 256 + rbuf[1];

	close(hd);
}


////////////////////////////////////////////////////////////////////////////////
// thread func
////////////////////////////////////////////////////////////////////////////////
void *thread_func(void *arg)
{
	int						id;
	THREAD_ARG				*th_arg;
	IMG_PROCESSING_PARAM	img_param;

	th_arg	= (THREAD_ARG *)arg;
	id		= th_arg->thread_id;

	pthread_mutex_lock(&(th_arg->th_rdyflag));
	th_arg->th_outrdy	= 0;
	th_arg->th_inrdy	= 1;		// notify thread is generated
	pthread_mutex_unlock(&(th_arg->th_rdyflag));

	img_param.srcImage	= &(th_arg->th_in);
	img_param.dstImage	= &(th_arg->th_out);
	img_param.params	= th_arg->params;

	fprintf(stderr, "thread %d start\n", id);

	while (1) {
		//	inrdy	outrdy
		//	1		0			1.Waiting input
		//	0		0			2.Image processing
		//	0		1			3.Waiting output
		//	0		0			4.Done

		// wait input (status <1,0>)
		while (th_arg->th_inrdy == 1) {
			usleep(100);
			continue;
		}

		// process image (status <0, 0>)
		opencv_func_run(&img_param);

		pthread_mutex_lock(&(th_arg->th_rdyflag));
		th_arg->th_outrdy	= 1;
		pthread_mutex_unlock(&(th_arg->th_rdyflag));

		// wait output (status <0,1>)
		while (th_arg->th_outrdy == 1) {
			usleep(100);
			continue;
		}

		pthread_mutex_lock(&(th_arg->th_rdyflag));
		th_arg->th_inrdy	= 1;
		pthread_mutex_unlock(&(th_arg->th_rdyflag));
	}
}


////////////////////////////////////////////////////////////////////////////////
// callback of video frame capture
////////////////////////////////////////////////////////////////////////////////
static void video_buffer_callback(MMAL_PORT_T *port, MMAL_BUFFER_HEADER_T *buffer)
{
	MMAL_BUFFER_HEADER_T	*new_buffer;
	PORT_USERDATA			*pData;
	char					allocate_flag	= 0;

	pData	= (PORT_USERDATA *)(port->userdata);
	if (pData) {
		if (buffer->length) {
			// check interval time
			clock_gettime(CLOCK_MONOTONIC_RAW, &tspec);
			t1_in	= (tspec.tv_nsec / 1000000) + (tspec.tv_sec * 1000);

			if (t1_in - t0_in >= fps_intval) {		// finish waiting
				if ((th_arg_G[next_in_thread_G]).th_inrdy == 1) { // check thread input
					int	w, h;

					mmal_buffer_header_mem_lock(buffer);
					w	= pData->pstate->width;
					h	= pData->pstate->height;
#ifdef IMG_COLOR
					memcpy(((th_arg_G[next_in_thread_G]).th_in).data, buffer->data, w * h * 3);	// set image
#endif	// IMG_COLOR
#ifdef IMG_GRAY
					memcpy(((th_arg_G[next_in_thread_G]).th_in).data, buffer->data, w * h);	// set image
#endif	// IMG_GRAY

					mmal_buffer_header_mem_unlock(buffer);

					pthread_mutex_lock(&(th_arg_G[next_in_thread_G]).th_rdyflag);
					(th_arg_G[next_in_thread_G]).th_inrdy	= 0;	// change this flag after setting image
					pthread_mutex_unlock(&(th_arg_G[next_in_thread_G]).th_rdyflag);

					// release buffer back to the pool
					mmal_buffer_header_release(buffer);
					if (port->is_enabled) {
						MMAL_STATUS_T	status;
						new_buffer	= mmal_queue_get(pData->pstate->video_pool->queue);
						if (new_buffer)
							status = mmal_port_send_buffer(port, new_buffer);
					}
					allocate_flag	= 1;

//////////////////////////////////////////////////////////////////////////// >>
// ADD YOUR ORIGINAL CODE HERE
// copy other information to input thread (to a member of th_arg_G[next_in_thread_G])
//////////////////////////////////////////////////////////////////////////// <<

					next_in_thread_G++;
					if (next_in_thread_G >= THREAD_NUM) {
						next_in_thread_G	= 0;
					}

					clock_gettime(CLOCK_MONOTONIC_RAW, &tspec);
					t1_in	= (tspec.tv_nsec / 1000000) + (tspec.tv_sec * 1000);
					t0_in	= t1_in;
				}
				// else ... discard this frame
			}
			// else "discard this frame"
		}
	}

	if (allocate_flag == 0) {
		// release buffer back to the pool
		mmal_buffer_header_release(buffer);

		if (port->is_enabled) {
			MMAL_STATUS_T	status;
			new_buffer	= mmal_queue_get(pData->pstate->video_pool->queue);
			if (new_buffer)
				status = mmal_port_send_buffer(port, new_buffer);
		}
	}
}


////////////////////////////////////////////////////////
// mjpg-streamer (source r182)	plugin interface functions
////////////////////////////////////////////////////////

/******************************************************************************
Description.: parse input parameters
Input Value.: param contains the command line string and a pointer to globals
Return Value: 0 if everything is ok
******************************************************************************/
int input_init(input_parameter *param, int id)
{
  int i;
  plugin_number = id;

  param->argv[0] = (char *)(INPUT_PLUGIN_NAME);

  /* show all parameters for DBG purposes */
  for(i = 0; i < param->argc; i++) {
      DBG("argv[%d]=%s\n", i, param->argv[i]);
  }

  reset_getopt();
  while(1) {
    int option_index = 0, c=0;
    static struct option long_options[] = {
      {"h", no_argument, 0, 0},
      {"help", no_argument, 0, 0},
      {"x", required_argument, 0, 0},
      {"width", required_argument, 0, 0},
      {"y", required_argument, 0, 0},
      {"height", required_argument, 0, 0},
      {"fps", required_argument, 0, 0},
      {"framerate", required_argument, 0, 0},
      {"quality", required_argument, 0, 0},
      {"preview", no_argument, 0, 0},
      {"websocket_port", required_argument, 0, 0},
      {0, 0, 0, 0}
    };

    c = getopt_long_only(param->argc, param->argv, "", long_options, &option_index);

    /* no more options to parse */
    if (c == -1) break;

    /* unrecognized option */
    if (c == '?'){
      help();
      return 1;
    }

    switch(option_index) {
      /* h, help */
      case 0:
      case 1:
        DBG("case 0,1\n");
        help();
        return 1;
        break;
        /* width */
      case 2:
      case 3:
        DBG("case 2,3\n");
        width = atoi(optarg);
        break;
        /* height */
      case 4:
      case 5:
        DBG("case 4,5\n");
        height = atoi(optarg);
        break;
        /* fps */
      case 6:
      case 7:
        DBG("case 6, 7\n");
        fps = atof(optarg);
		if (fps != 0) {
			fps_intval = (long)(1000.0 / fps);	// unit: ms
		}
		else {
			fps			= (1000.0 / (float)DEFAULT_FPS_INTVAL);
			fps_intval	= DEFAULT_FPS_INTVAL;
		}
        break;
        /* quality */
        case 8:
        	quality = atoi(optarg);
        	break;
		case 9:
        	wantPreview = 1;
        	break;
        /* websocket */
		case 10:
			wsocket_port	= atoi(optarg);
			break;
      default:
        DBG("default case\n");
        help();
        return 1;
    }
  }

  pglobal = param->global;

  IPRINT("fps.............: %f (%d ms)\n", fps, fps_intval);
  IPRINT("resolution......: %i x %i\n", width, height);
  IPRINT("quality.........: %d\n", quality);
  IPRINT("websocket port..: %d\n", wsocket_port);

  return 0;
}

/******************************************************************************
Description.: stops the execution of the worker thread
Input Value.: -
Return Value: 0
******************************************************************************/
int input_stop(int id)
{
  	DBG("will cancel input thread\n");
  	pthread_cancel(worker);

	opencv_func_stop((void *)NULL);

  	return 0;
}

/******************************************************************************
Description.: starts the worker thread and allocates memory
Input Value.: -
Return Value: 0
******************************************************************************/
int input_run(int id)
{
  pglobal->in[id].buf = (unsigned char *)malloc(256*1024);
  if (pglobal->in[id].buf == NULL) {
    fprintf(stderr, "could not allocate memory\n");
    exit(EXIT_FAILURE);
  }

  if( pthread_create(&worker, 0, worker_thread, NULL) != 0) {
    free(pglobal->in[id].buf);
    fprintf(stderr, "could not start worker thread\n");
    exit(EXIT_FAILURE);
  }
  pthread_detach(worker);

  return 0;
}

/******************************************************************************
Description.: print help message
Input Value.: -
Return Value: -
******************************************************************************/
void help(void)
{
  fprintf(stderr, " ---------------------------------------------------------------\n" \
      " Help for input plugin..: "INPUT_PLUGIN_NAME"\n" \
      " ---------------------------------------------------------------\n" \
      " The following parameters can be passed to this plugin:\n\n" \
      " [-fps | --framerate]...: set video framerate, default 1 frame/sec \n"\
      " [-x | --width ]........: width of frame capture, default 640\n" \
      " [-y | --height]........: height of frame capture, default 480 \n"\
      " [-quality].............: set JPEG quality 0-100, default 95 \n"\
      " [-preview].............: Enable full screen preview\n"\
      " [-websocket_port]......: port number of websocket\n"\
      " ---------------------------------------------------------------\n");
}

/******************************************************************************
Description.: copy a picture from testpictures.h and signal this to all output
              plugins, afterwards switch to the next frame of the animation.
Input Value.: arg is not used
Return Value: NULL
******************************************************************************/
void *worker_thread( void *arg )
{
  	int i = 0;

  	/* set cleanup handler to cleanup allocated ressources */
  	pthread_cleanup_push(worker_cleanup, NULL);

	RASPIVID_STATE		state;
	MMAL_STATUS_T		status;
	MMAL_PORT_T			*camera_video_port = NULL;
	PORT_USERDATA		callback_data;

	MMAL_COMPONENT_T	*camera = 0;
	MMAL_ES_FORMAT_T	*format;
	MMAL_PORT_T			*video_port = NULL;
	MMAL_POOL_T			*pool;
	MMAL_PARAMETER_CAMERA_CONFIG_T	cam_config;

	cv::Mat				outImage;

	//////////////////////////////////////////////////////////
	// init variables
	//////////////////////////////////////////////////////////
	memset(&state, 0, sizeof(RASPIVID_STATE));

	state.width 		= width;					// DO NOT CHANGE (camera frame size)
	state.height 		= height;					// DO NOT CHANGE (camera frame size)
	state.framerate 	= VIDEO_FRAME_RATE_NUM;		// DO NOT CHANGE

	// Set up the camera_parameters to default
	raspicamcontrol_set_defaults(&(state.camera_parameters));

	opencv_func_init((void *)NULL);

#ifdef IMG_COLOR
	outImage.create(cv::Size(state.width, state.height), CV_8UC3);
#endif
#ifdef IMG_GRAY
	outImage.create(cv::Size(state.width, state.height), CV_8UC1);
#endif

	//////////////////////////////////////////////////////////
	// init camera
	//////////////////////////////////////////////////////////
	bcm_host_init();

	// Create the component
	status = mmal_component_create(MMAL_COMPONENT_DEFAULT_CAMERA, &camera);
	video_port = camera->output[MMAL_CAMERA_VIDEO_PORT];

	//  set up the camera configuration
	(cam_config.hdr).id				= MMAL_PARAMETER_CAMERA_CONFIG;
	(cam_config.hdr).size			= sizeof(cam_config);
	cam_config.max_stills_w			= state.width;
	cam_config.max_stills_h			= state.height;
	cam_config.stills_yuv422		= 0;
	cam_config.one_shot_stills 		= 0;
	cam_config.max_preview_video_w	= state.width;
	cam_config.max_preview_video_h	= state.height;
	cam_config.num_preview_video_frames	= 3;
	cam_config.stills_capture_circular_buffer_height	= 0;
	cam_config.fast_preview_resume	= 0;
	cam_config.use_stc_timestamp	= MMAL_PARAM_TIMESTAMP_MODE_RESET_STC;

	mmal_port_parameter_set(camera->control, &(cam_config.hdr));

	// Set the encode format on the video  port
	format	= video_port->format;
#ifdef IMG_COLOR
	format->encoding_variant		= MMAL_ENCODING_RGB24;
	format->encoding				= MMAL_ENCODING_RGB24;
#endif
#ifdef IMG_GRAY
	format->encoding_variant		= MMAL_ENCODING_I420;
	format->encoding				= MMAL_ENCODING_I420;
#endif
	format->es->video.width			= state.width;
	format->es->video.height		= state.height;
	format->es->video.crop.x		= 0;
	format->es->video.crop.y		= 0;
	format->es->video.crop.width	= state.width;
	format->es->video.crop.height	= state.height;
	format->es->video.frame_rate.num = state.framerate;
	format->es->video.frame_rate.den = VIDEO_FRAME_RATE_DEN;

	status = mmal_port_format_commit(video_port);
	status = mmal_port_enable(video_port, video_buffer_callback);

	// Ensure there are enough buffers to avoid dropping frames
	if (video_port->buffer_num < VIDEO_OUTPUT_BUFFERS_NUM) {
		video_port->buffer_num = VIDEO_OUTPUT_BUFFERS_NUM;
	}

	// create pool of message on video port
	video_port->buffer_size		= video_port->buffer_size_recommended;
	video_port->buffer_num		= video_port->buffer_num_recommended;
	pool				= mmal_port_pool_create(video_port, video_port->buffer_num, video_port->buffer_size);
	state.video_pool	= pool;

	// Enable component
	status	= mmal_component_enable(camera);
	raspicamcontrol_set_all_parameters(camera, &(state.camera_parameters));
	state.camera_component	= camera;

	camera_video_port		= state.camera_component->output[MMAL_CAMERA_VIDEO_PORT];
	callback_data.pstate	= &state;

	// assign data to use for callback
	camera_video_port->userdata = (struct MMAL_PORT_USERDATA_T *)&callback_data;

//////////////////////////////////////////////////////////////////////////// >>
// ADD YOUR ORIGINAL CODE HERE
// preparation of image processing, prior to starting capture
//////////////////////////////////////////////////////////////////////////// <<

	// init thread parameters
	for (i = 0; i < THREAD_NUM; i++) {
		(th_arg_G[i]).thread_id		= i;
		(th_arg_G[i]).rasp_state	= &(state);

		pthread_mutex_init(&((th_arg_G[i]).th_rdyflag), NULL);
		(th_arg_G[i]).th_inrdy	= 0;	// NOTE: set to 0 prior to generating thread
		(th_arg_G[i]).th_outrdy	= 0;

#ifdef IMG_COLOR
		((th_arg_G[i]).th_in).create(cv::Size(state.width, state.height),  CV_8UC3);
		((th_arg_G[i]).th_out).create(cv::Size(state.width, state.height),  CV_8UC3);
#endif
#ifdef IMG_GRAY
		((th_arg_G[i]).th_in).create(cv::Size(state.width, state.height),  CV_8UC1);
		((th_arg_G[i]).th_out).create(cv::Size(state.width, state.height),  CV_8UC1);
#endif

//////////////////////////////////////////////////////////////////////////// >>
// ADD YOUR ORIGINAL CODE HERE
// setting other thread parameters
//////////////////////////////////////////////////////////////////////////// <<
	}

	// create thread
	pthread_attr_t	attr;
	cpu_set_t		cpus;
    pthread_attr_init(&attr);

	for (i = 0; i < THREAD_NUM; i++) {
		CPU_ZERO(&cpus);
		CPU_SET(i % 4, &cpus);
		pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &cpus);

#ifdef SPECIFY_CPUCORE
		pthread_create(&((th_arg_G[i]).th), &attr, thread_func, (void *)(&(th_arg_G[i])));
#else
		pthread_create(&((th_arg_G[i]).th), NULL, thread_func, (void *)(&(th_arg_G[i])));
#endif
	}

	// confirm all threads are generated
	while (1) {
		int	break_flag;
		break_flag	= 1;
		for (i = 0; i < THREAD_NUM; i++) {
			if ((th_arg_G[i]).th_inrdy == 0) {
				break_flag	= 0;
				break;
			}
		}
		if (break_flag == 1)
			break;
	}

	fprintf(stderr, "all thread ready\n");

//{
//char r0, r1, r2;
//i2c_read(0x78, 0x00, &r0);
//i2c_read(0x78, 0x01, &r1);
//i2c_read(0x78, 0xFF, &r2);
//fprintf(stderr, "%02x %02x %02x\n", r0, r1, r2);
//}


	// websockets
	pthread_t		wsocket_th;

#ifdef SPECIFY_CPUCORE
	CPU_ZERO(&cpus);
	CPU_SET(0, &cpus);		// change cpu ID if necessary
	pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &cpus);
	pthread_create(&wsocket_th, &attr, wsocket_thread_func, (void *)NULL);
#else
	pthread_create(&wsocket_th, NULL, wsocket_thread_func, (void *)NULL);
#endif


	// i2c (fpga)
	pthread_mutex_init(&(i2c_mutex), NULL);


	//////////////////////////////////////////////////////////
	// start capture
	//////////////////////////////////////////////////////////
	clock_gettime(CLOCK_MONOTONIC_RAW, &tspec);
	t0_in	= t0_out	= (tspec.tv_nsec / 1000000) + (tspec.tv_sec * 1000);

    // start capture
	mmal_port_parameter_set_boolean(camera_video_port, MMAL_PARAMETER_CAPTURE, 1);

	// Send all the buffers to the video port
	int		num = mmal_queue_length(state.video_pool->queue);
	int		q;

	for (q = 0; q < num; q++) {
		MMAL_BUFFER_HEADER_T *buffer;
		buffer	= mmal_queue_get(state.video_pool->queue);
		mmal_port_send_buffer(camera_video_port, buffer);
	}

	while( !pglobal->stop ) {
		// output
		if ((th_arg_G[next_out_thread_G]).th_outrdy == 1) {
#ifdef IMG_COLOR
			memcpy(outImage.data, ((th_arg_G[next_out_thread_G]).th_out).data, state.width * state.height * 3);
#endif
#ifdef IMG_GRAY
			memcpy(outImage.data, ((th_arg_G[next_out_thread_G]).th_out).data, state.width * state.height);
#endif

//////////////////////////////////////////////////////////////////////////// >>
// ADD YOUR ORIGINAL CODE HERE
// copy other information from output thread (from a member of th_arg_G[next_out_thread_G])
//////////////////////////////////////////////////////////////////////////// <<

			pthread_mutex_lock(&((th_arg_G[next_out_thread_G]).th_rdyflag));
			(th_arg_G[next_out_thread_G]).th_outrdy	= 0;			// change this flag after getting image
			pthread_mutex_unlock(&((th_arg_G[next_out_thread_G]).th_rdyflag));

			next_out_thread_G++;
			if (next_out_thread_G >= THREAD_NUM) {
				next_out_thread_G	= 0;
			}

//////////////////////////////////////////////////////////////////////////// >>
// ADD YOUR ORIGINAL CODE HERE
// use image processing result (cv::Mat outImage)
//////////////////////////////////////////////////////////////////////////// <<

			//////////////////////////////////////
			// jpeg compression
			//////////////////////////////////////
			std::vector<uchar>	encbuf;
			std::vector<int>	enc_param = std::vector<int>(2);
			enc_param[0]	= CV_IMWRITE_JPEG_QUALITY;
			enc_param[1]	= quality;	// 0-100, default(95)
			imencode(".jpg", outImage, encbuf, enc_param);

			//////////////////////////////////////
			// copy jpeg to output buffer
			//////////////////////////////////////
			pthread_mutex_lock( &pglobal->in[plugin_number].db );
			pglobal->in[plugin_number].size = encbuf.size();
			for (int i = 0; i < encbuf.size(); i++) {
				*((pglobal->in[plugin_number].buf) + i)	= (char)(encbuf.at(i));
			}
			pthread_cond_broadcast(&(pglobal->in[plugin_number].db_update));    /* signal fresh_frame */
			pthread_mutex_unlock( &(pglobal->in[plugin_number].db) );

			//////////////////////////////////////
			// other task
			//////////////////////////////////////
			if (wantPreview != 0) {
				cv::imshow(WINNAME, outImage);
				cv::waitKey(1);
			}

#ifdef	DISP_FPS
			clock_gettime(CLOCK_MONOTONIC_RAW, &tspec);
			t1_out	= (tspec.tv_nsec / 1000000) + (tspec.tv_sec * 1000);
			tdiff	= (int)(t1_out - t0_out);
			tall	+= tdiff;
			fnum++;

			if (tdiff > 0 && tall > 0) {
				fprintf(stderr, "%d fps (%d ms)     \r", (1000 * fnum) / tall, tall / fnum);
			}
			t0_out	= t1_out;
#endif	// DISP_FPS
		}

		usleep(100);		// 0.1ms
	}

	//////////////////////////////////////////////////////////
	// end of process
	//////////////////////////////////////////////////////////
	vcos_semaphore_delete(&callback_data.complete_semaphore);

	mmal_status_to_int(status);

	// Disable all our ports that are not handled by connections
	if (state.camera_component) {
		mmal_component_disable(state.camera_component);
	}

	//	destroy_camera_component(&state);
	if (state.camera_component) {
		mmal_component_destroy(state.camera_component);
		state.camera_component = NULL;
	}

	opencv_func_stop((void *)NULL);
	outImage.release();

	IPRINT("leaving input thread, calling cleanup function now\n");
	pthread_cleanup_pop(1);

	return NULL;
}


/******************************************************************************
Description.: this functions cleans up allocated ressources
Input Value.: arg is unused
Return Value: -
******************************************************************************/
void worker_cleanup(void *arg)
{
  static unsigned char first_run=1;

  if ( !first_run ) {
    DBG("already cleaned up ressources\n");
    return;
  }

  first_run = 0;
  DBG("cleaning up ressources allocated by input thread\n");

  if (pglobal->in[plugin_number].buf != NULL) free(pglobal->in[plugin_number].buf);
}

// end of file
