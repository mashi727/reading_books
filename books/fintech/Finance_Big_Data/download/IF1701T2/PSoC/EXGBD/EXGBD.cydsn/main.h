/******************************************************************************
* File name:   main.h
* Version:     1.0.0
* Discription:
*   This file defines the commonly used macros for this project.
* History:
*   1.0.0  2016.07.28  Initial coding
*
* Copyright 2016 METOOL Inc. All Rights Reserved
******************************************************************************/

#if !defined(_MAIN_H)
#define _MAIN_H

/******************************************************************************
* Included headers
******************************************************************************/
#include <project.h>

/******************************************************************************
* Macros 
******************************************************************************/
#define BLE_ENABLED

#define CH1_INDEX                       (0)
#define CH2_INDEX                       (1)

// General
#define TRUE							(1)
#define FALSE							(0)
#define ZERO							(0)

#define LED_ON		    				(0)
#define LED_OFF	    					(1)

// Control command definision
#define CH1_CTRL_INDEX					(0)
#define CH2_CTRL_INDEX					(1)
#define SMPL_FREQ_INDEX 				(2)
#define LED_INDEX   					(3)

#define CH_CTRL_GAIN_MASK               (0x0F)
#define CH_CTRL_FILTER_MASK             (0xF0)
#define CH_CTRL_FILTER_SHIFT            (4)

#define GAIN_LOW						(0x01)
#define GAIN_HIGH						(0x00)

#define FILTER_LOW						(0x01)
#define FILTER_HIGH						(0x00)

#define SAMPLE_FREQ_COEFF				(10)
#define SAMPLE_FREQ_PARAM_MIN_VALUE		(1)         // 10 Hz
#define SAMPLE_FREQ_PARAM_MAX_VALUE		(100)       // 1000 Hz

#define CH_NUM                          (2)         // Number of input channel
#define SEND_DATA_LENGTH_PER_CH         (4)         // Wave data length per channel
#define SEND_DATA_LENGTH                (8)         // Wave data bytes
#define WAVE_FIFO_LENGTH                (32)        // Fifo buffer length

#define COMM_DATA_LENGTH                (4)         // Command data bytes

#define AD_SAMPLING_FREQ                (1000)      // A/D frequency
#define SAMPLING_TIMING_INIT_VAL        (5)         // 200 Hz (5=1000/200)
    
#define BIT_MASK_LED                    (0x01)
    
#define PI                              3.1415926535897932386   // Temp

/******************************************************************************
* Public functions
******************************************************************************/
void setChCtrl(uint8 ch, uint8 val);
void setSampleFreq(uint8 freq);
void setLED(uint8 flag);

#endif  /* #if !defined(_MAIN_H) */

/* [] END OF FILE */
