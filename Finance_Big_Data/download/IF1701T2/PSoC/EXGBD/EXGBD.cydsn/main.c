/******************************************************************************
* File name:   main.c
* Version:     1.0.0
* Discription:
*   This is the application for Sending biosignal data on EXG BD.
* History:
*   1.0.0  2016.07.28  Initial coding
*
* Copyright 2016 METOOL Inc. All Rights Reserved
******************************************************************************/

/******************************************************************************
* Included headers
******************************************************************************/
#include "main.h"
#include "BLEApplications.h"

/******************************************************************************
* Function Prototypes
******************************************************************************/
static void InitComponent(void);
static void InitWaveBuf(void);

/******************************************************************************
* Static variables 
******************************************************************************/
uint32 ad_counter;
uint32 sample_timing;
uint32 sample_counter;
int wave_buf_pointer;   // Circular buffer pointer
int16 wave_data[WAVE_FIFO_LENGTH][SEND_DATA_LENGTH];

/******************************************************************************
* Function Name: ADC_Interrupt_Handler
*******************************************************************************
* Summary:
* Acquiring the waveform data and call send BLE notification function each
* periodical timing.
*
* Parameters:
*  void
*
* Return:
*  int
*
******************************************************************************/
CY_ISR(AdcInterruptHandler)
{
    uint32 intr_status = ADC_SAR_Seq_1_SAR_INTR_REG;
    
    int16 ch1_value = ADC_SAR_Seq_1_GetResult16(CH1_INDEX);
    int16 ch2_value = ADC_SAR_Seq_1_GetResult16(CH2_INDEX);
    
    ad_counter++;
    if(ad_counter >= sample_timing)
    {
        ad_counter = 0;
        wave_data[wave_buf_pointer][sample_counter * CH_NUM] = ch1_value;
        wave_data[wave_buf_pointer][sample_counter * CH_NUM + 1] = ch2_value;
        
        sample_counter++;
        if(sample_counter >= SEND_DATA_LENGTH_PER_CH)
        {
            sample_counter = 0;
            SendBiosignalDataNotification(wave_data[wave_buf_pointer]);
            wave_buf_pointer = (wave_buf_pointer + 1) % WAVE_FIFO_LENGTH;
        }
    }
    ADC_SAR_Seq_1_SAR_INTR_REG = intr_status;
}


/******************************************************************************
* Function Name: main
*******************************************************************************
* Summary:
* System entrance point. This calls the initializing function and continuously
* process BLE events.
*
* Parameters:
*  void
*
* Return:
*  int
*
******************************************************************************/
int main(void)
{
    ad_counter = 0;
    sample_timing = SAMPLING_TIMING_INIT_VAL;
    sample_counter = 0;
    wave_buf_pointer = 0;
    InitWaveBuf();
    
    InitComponent();
    
    CyGlobalIntEnable;

    for(;;)
    {
        CyBle_ProcessEvents();
    }
}


/******************************************************************************
* Function Name: setChCtrl
*******************************************************************************
* Summary:
* Set gain and filter for each input channel. Called from BLEApplications.c.
*
* Parameters:
*  void
*
* Return:
*  void
*
******************************************************************************/
void setChCtrl(uint8 ch, uint8 val)
{
    uint8 gain = val & CH_CTRL_GAIN_MASK;
    uint8 filt = (val & CH_CTRL_FILTER_MASK) >> CH_CTRL_FILTER_SHIFT;
    
    if((gain==GAIN_HIGH) || (gain==GAIN_LOW))
    {
        if(ch==CH1_INDEX)
        {
            AMux_Ch1N_Select(gain);
        }
        else
        {
            AMux_Ch2N_Select(gain);
        }
    }
    
    if((filt==FILTER_HIGH) || (filt==FILTER_LOW))
    {
        if(ch==CH1_INDEX)
        {
            AMux_Ch1P_Select(filt);
        }
        else
        {
            AMux_Ch2P_Select(filt);
        }
    }
}

/******************************************************************************
* Function Name: setSampleFreq
*******************************************************************************
* Summary:
* Set sampling frequency. Called from BLEApplications.c.
*
* Parameters:
*  void
*
* Return:
*  void
*
******************************************************************************/
void setSampleFreq(uint8 freq)
{
    if((SAMPLE_FREQ_PARAM_MIN_VALUE <= freq) && (freq <= SAMPLE_FREQ_PARAM_MAX_VALUE))
    {
        sample_timing = (uint32)AD_SAMPLING_FREQ / (freq * SAMPLE_FREQ_COEFF);
    }
}


/******************************************************************************
* Function Name: setLED
*******************************************************************************
* Summary:
* Start LED ON/OFF. Called from BLEApplications.c.
*
* Parameters:
*  void
*
* Return:
*  void
*
******************************************************************************/
void setLED(uint8 flag)
{
    if(flag)
    {
        Pin_LED_Write(LED_ON);
    }
    else
    {
        Pin_LED_Write(LED_OFF);
    }
}


/******************************************************************************
* Function Name: InitComponent
*******************************************************************************
* Summary:
* Initialize components.
*
* Parameters:
*  void
*
* Return:
*  void
*
******************************************************************************/
void InitComponent(void)
{
    OPAMP_1_Start();
    OPAMP_2_Start();
    OPAMP_3_Start();
    OPAMP_4_Start();
    
    AMux_Ch1N_Start();
    AMux_Ch1N_Select(0);
    AMux_Ch2N_Start();
    AMux_Ch2N_Select(0);
    AMux_Ch1P_Start();
    AMux_Ch1P_Select(0);
    AMux_Ch2P_Start();
    AMux_Ch2P_Select(0);
    
    ADC_SAR_Seq_1_Start();
    ADC_SAR_Seq_1_IRQ_SetVector(AdcInterruptHandler);
    ADC_SAR_Seq_1_IRQ_Enable();
    ADC_SAR_Seq_1_StartConvert();

    Pin_LED_Write(LED_ON);
    
    CyBle_Start(CustomEventHandler);	
}

/******************************************************************************
* Function Name: InitWaveBuf
*******************************************************************************
* Summary:
* Initialize wave buufer.
*
* Parameters:
*  void
*
* Return:
*  void
*
******************************************************************************/
void InitWaveBuf(void)
{
    int i,j;
    for(i=0; i<WAVE_FIFO_LENGTH; i++)
    {
        for(j=0; j<SEND_DATA_LENGTH; j++)
        {
            wave_data[i][j] = 0;
        }
    }
}

/* [] END OF FILE */
