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
#ifndef _WEBSOCKET_FLAGS_H
#define _WEBSOCKET_FLAGS_H

//////////////////////////////////////////////////////////////////////////// >>
// ADD YOUR ORIGINAL CODE HERE
// EXTERN DECLARATION of global variables
extern pthread_mutex_t	i2c_mutex;

#define FPGA_I2C_ADDR	0x3C

extern "C" {
void i2c_write(char, char, char);
void i2c_read(char, char, char *);
void i2c_read_word(char, char, int *);
}
//////////////////////////////////////////////////////////////////////////// <<

#endif // _WEBSOCKET_FLAGS_H

// end of file
