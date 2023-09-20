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
#ifndef _WEBSOCKET_FUNC_H
#define _WEBSOCKET_FUNC_H

#include "websocket_flags.h"

//////////////////////////////////////////////////////////////////////////// >>
// ADD YOUR ORIGINAL CODE HERE
// BODY of global variables
pthread_mutex_t	i2c_mutex;
//////////////////////////////////////////////////////////////////////////// <<

static int wsocket_callback_dumb_increment(struct libwebsocket_context *cntxt,
                                   struct libwebsocket *wsi,
                                   enum libwebsocket_callback_reasons reason,
                                   void *user,
                                   void *inp,
                                   size_t len)	// (main callback func)
{
//////////////////////////////////////////////////////////////////////////// >>
// ADD YOUR ORIGINAL CODE HERE
	switch (reason) {
	case LWS_CALLBACK_ESTABLISHED:
		pthread_mutex_lock(&i2c_mutex);
		i2c_write(FPGA_I2C_ADDR, 0x80, 0x00);	// stop
		pthread_mutex_unlock(&i2c_mutex);

		fprintf(stderr, "websocket: connection established\n");
		break;

	case LWS_CALLBACK_RECEIVE:
		if (*((char *)inp) != '!') {	// NOT KEEP ALIVE MARK
			int	x, y;
			int	target_dir;

			switch (*((char *)inp)) {
			case 'c':
				pthread_mutex_lock(&i2c_mutex);
				i2c_write(FPGA_I2C_ADDR, 0x80, 0x00);	// stop
				pthread_mutex_unlock(&i2c_mutex);
				break;

			case 'u':
				pthread_mutex_lock(&i2c_mutex);
				i2c_write(FPGA_I2C_ADDR, 0x80, 0x00);	// stop

				i2c_read_word(FPGA_I2C_ADDR, 0x83, &target_dir);	// get current direction
				i2c_write(FPGA_I2C_ADDR, 0x81, (((unsigned int)target_dir) >> 8) & 0xFF);	// set same target
				i2c_write(FPGA_I2C_ADDR, 0x82, ((unsigned int)target_dir) & 0xFF);			// set same target

				i2c_write(FPGA_I2C_ADDR, 0x80,
					   (1 & 0x01)				// start
					+ ((0 << 1) & 0x06)			// mode
					+ ((255 << 3) & 0xF8)		// pwm
				);
				pthread_mutex_unlock(&i2c_mutex);
				break;

			case 'd':
				pthread_mutex_lock(&i2c_mutex);
				i2c_write(FPGA_I2C_ADDR, 0x80, 0x00);	// stop

				i2c_read_word(FPGA_I2C_ADDR, 0x83, &target_dir);	// get current direction
				i2c_write(FPGA_I2C_ADDR, 0x81, (((unsigned int)target_dir) >> 8) & 0xFF);	// set same target
				i2c_write(FPGA_I2C_ADDR, 0x82, ((unsigned int)target_dir) & 0xFF);			// set same target

				i2c_write(FPGA_I2C_ADDR, 0x80,
					   (1 & 0x01)				// start
					+ ((1 << 1) & 0x06)			// mode
					+ ((255 << 3) & 0xF8)		// pwm
				);
				pthread_mutex_unlock(&i2c_mutex);
				break;

			case 'l':
				pthread_mutex_lock(&i2c_mutex);
				i2c_write(FPGA_I2C_ADDR, 0x80, 0x00);	// stop

				i2c_read_word(FPGA_I2C_ADDR, 0x83, &target_dir);	// get current direction
				target_dir	-= 3000;					// 30 degree
				while (target_dir < 0) {
					target_dir	+= 36000;
				}
				i2c_write(FPGA_I2C_ADDR, 0x81, (((unsigned int)target_dir) >> 8) & 0xFF);	// set same target
				i2c_write(FPGA_I2C_ADDR, 0x82, ((unsigned int)target_dir) & 0xFF);			// set same target

				i2c_write(FPGA_I2C_ADDR, 0x80,
					   (1 & 0x01)				// start
					+ ((3 << 1) & 0x06)			// mode
					+ ((255 << 3) & 0xF8)		// pwm
				);
				pthread_mutex_unlock(&i2c_mutex);
				break;

			case 'r':
				pthread_mutex_lock(&i2c_mutex);
				i2c_write(FPGA_I2C_ADDR, 0x80, 0x00);	// stop

				i2c_read_word(FPGA_I2C_ADDR, 0x83, &target_dir);	// get current direction
				target_dir	+= 3000;					// 30 degree
				while (target_dir >= 36000) {
					target_dir	-= 36000;
				}
				i2c_write(FPGA_I2C_ADDR, 0x81, (((unsigned int)target_dir) >> 8) & 0xFF);	// set same target
				i2c_write(FPGA_I2C_ADDR, 0x82, ((unsigned int)target_dir) & 0xFF);			// set same target

				i2c_write(FPGA_I2C_ADDR, 0x80,
					   (1 & 0x01)				// start
					+ ((2 << 1) & 0x06)			// mode
					+ ((255 << 3) & 0xF8)		// pwm
				);
				pthread_mutex_unlock(&i2c_mutex);
				break;

//			case 'i':
//				sscanf((char *)inp, "i%d,%d", &x, &y);
//
//				if (x >= 33 && x <= 66 && y < 33) {			// up
//				}
//				else if (x >= 33 && x <= 66 && y > 66) {	// down
//				}
//				else if (y >= 33 && y <= 66 && x < 33) {	// left
//				}
//				else if (y >= 33 && y <= 66 && x > 66) {	// right
//				}
//				else if (x >= 33 && x <= 66 && y >= 33 && y <= 66) {	// center
//				}
//				break;

			default:
				break;
			}
		}

		// send response TO AVOID TIMEOUT
		{
			unsigned char	buf	= '!';
			libwebsocket_write(wsi, &buf, 1, LWS_WRITE_TEXT);
		}
		break;

	case LWS_CALLBACK_CLOSED:
		pthread_mutex_lock(&i2c_mutex);
		i2c_write(FPGA_I2C_ADDR, 0x80, 0x00);	// stop
		pthread_mutex_unlock(&i2c_mutex);

		fprintf(stderr, "websocket: connection closed\n");
		break;

	default:
		break;
    }

	return 0;
//////////////////////////////////////////////////////////////////////////// <<
}

#endif // _WEBSOCKET_FUNC_H

// end of file
