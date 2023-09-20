/******************************************************************************
* File name:   BLEApplications.c
* Version:     1.0.0
* Discription:
*   This is the BLE capability for Sending biosignal on EXG BD.
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
* Static variables 
******************************************************************************/

/* 'bdControlHandle' stores BD control data parameters */
CYBLE_GATT_HANDLE_VALUE_PAIR_T		bdControlHandle;	

/* This flag is set when the Central device writes to CCC (Client Characteristic
* Configuration) of the Biosignal data Characteristic to enable notifications */
uint8 sendBiosignalDataNotifications = FALSE;	

/* Array to send biosignal data.
* The 8 bytes of the array represents {Ch1_1, Ch2_1, ... , Ch1_4, Ch2_4 } */
int16 biosignalData[SEND_DATA_LENGTH];

/* Array to store the present BD control data.
* The 4 bytes of the array represents {Ch1gain/filter, Ch2gain/filter, sampring rate, LED} */
uint8 bdControlData[COMM_DATA_LENGTH];

/* This flag is used by application to know whether a Central device
* has been connected. This is updated in BLE event callback function */
uint8 deviceConnected = FALSE;


/******************************************************************************
* Function Name: CustomEventHandler
*******************************************************************************
* Summary:
* Call back event function to handle various events from BLE stack
*
* Parameters:
*  event:		event returned
*  eventParam:	link to value of the events returned
*
* Return:
*  void
*
******************************************************************************/
void CustomEventHandler(uint32 event, void * eventParam)
{
	CYBLE_GATTS_WRITE_REQ_PARAM_T *wrReqParam;
    
	uint8 BiosignalCCCDvalue[2];
	
	CYBLE_GATT_HANDLE_VALUE_PAIR_T BiosignalNotificationCCCDhandle;
   
    switch(event)
    {
        case CYBLE_EVT_STACK_ON:
        case CYBLE_EVT_GAP_DEVICE_DISCONNECTED:
			CyBle_GappStartAdvertisement(CYBLE_ADVERTISING_FAST);
			break;
			
		case CYBLE_EVT_GAPP_ADVERTISEMENT_START_STOP:
            if(CYBLE_STATE_DISCONNECTED == CyBle_GetState())
            {
                CyBle_GappStartAdvertisement(CYBLE_ADVERTISING_FAST);
            }
			break;
			
        case CYBLE_EVT_GATT_CONNECT_IND:
			deviceConnected = TRUE;
			break;
        
        case CYBLE_EVT_GATT_DISCONNECT_IND:
			deviceConnected = FALSE;
			sendBiosignalDataNotifications = FALSE;
			
			bdControlData[CH1_CTRL_INDEX] = ZERO;
            bdControlData[CH2_CTRL_INDEX] = ZERO;
            bdControlData[SMPL_FREQ_INDEX] = ZERO;
            bdControlData[LED_INDEX] = ZERO;
			break;
        
            
        case CYBLE_EVT_GATTS_WRITE_REQ: 							
            wrReqParam = (CYBLE_GATTS_WRITE_REQ_PARAM_T *) eventParam;
                         
            uint32 RecvAttrHandle = wrReqParam->handleValPair.attrHandle;
            uint32 BdCtrlHandle = cyBle_customs[BIOSIGNAL_SERVICE_INDEX].\
                                                customServiceInfo[BD_CONTROL_CHAR_INDEX].\
                                                customServiceCharHandle;
            uint32 BioSigHandle = cyBle_customs[BIOSIGNAL_SERVICE_INDEX].\
                                                customServiceInfo[BIOSIGNAL_DATA_CHAR_INDEX].\
                                                customServiceCharDescriptors[BIOSIGNAL_DATA_CCC_INDEX];

            /* if (attributeHandle == BD Control Characteristic Handle) */
            if(RecvAttrHandle == BdCtrlHandle)
            {
                bdControlData[CH1_CTRL_INDEX] = wrReqParam->handleValPair.value.val[CH1_CTRL_INDEX];
                bdControlData[CH2_CTRL_INDEX] = wrReqParam->handleValPair.value.val[CH2_CTRL_INDEX];
                bdControlData[SMPL_FREQ_INDEX] = wrReqParam->handleValPair.value.val[SMPL_FREQ_INDEX];
                bdControlData[LED_INDEX] = wrReqParam->handleValPair.value.val[LED_INDEX];
                
                /* Update BD settings */
                UpdateBdSettings();
            }
            
            /* if (attributeHandle == Biosignal Data Characteristics Handle) */
            if(RecvAttrHandle == BioSigHandle)
            {
                sendBiosignalDataNotifications = wrReqParam->handleValPair.value.val[CCC_DATA_INDEX];
				
                /* When the Client Characteristic Configuration descriptor (CCCD) is
                * written by the Central device for enabling/disabling notifications, 
                * then the same descriptor value has to be explicitly updated in 
                * application so that it reflects the correct value when the 
                * descriptor is read. 
                */
                /* Write the present Biosignal data notification status to the local variable */
        		BiosignalCCCDvalue[0] = sendBiosignalDataNotifications;
        		BiosignalCCCDvalue[1] = 0x00;
        		
        		/* Update CCCD handle with notification status data*/
                BiosignalNotificationCCCDhandle.attrHandle = BIOSIGNAL_DATA_CCC_HANDLE;
        		BiosignalNotificationCCCDhandle.value.val = BiosignalCCCDvalue;
        		BiosignalNotificationCCCDhandle.value.len = sizeof(BiosignalCCCDvalue);
        		
        		/* Report data to BLE component for sending data when read by Central device */
        		CyBle_GattsWriteAttributeValue(&BiosignalNotificationCCCDhandle, ZERO, &cyBle_connHandle, \
                                                                            CYBLE_GATT_DB_LOCALLY_INITIATED);
           }
			
			/* Send the response to the write request received. */
			CyBle_GattsWriteRsp(cyBle_connHandle);
			
			break;

        default:

       	 	break;
    }
}


/******************************************************************************
* Function Name: SendBiosignalDataNotification
*******************************************************************************
* Summary:
* Send Biosignal data as BLE Notifications. This function updates
* the notification handle with data and triggers the BLE component to send 
* notification
*
* Parameters:
*  BiosignalData:	Biosignal value
*
* Return:
*  void
*
******************************************************************************/
void SendBiosignalDataNotification(int16 *bioData)
{
	/* 'CapSensenotificationHandle' stores Biosignal notification data parameters */
    CYBLE_GATT_HANDLE_VALUE_PAIR_T      BiosignalNotificationHandle;
	//CYBLE_GATTS_HANDLE_VALUE_NTF_T		BiosignalNotificationHandle;	
	
	/* Update notification handle with Biosignal data*/
	BiosignalNotificationHandle.attrHandle = CYBLE_BIOSIGNAL_SERVICE_BIOSIGNAL_DATA_CHARACTERISTIC_CHAR_HANDLE;				
	BiosignalNotificationHandle.value.val = (uint8*)bioData;
	BiosignalNotificationHandle.value.len = sizeof(biosignalData);
	BiosignalNotificationHandle.value.actualLen = sizeof(biosignalData);
    
	/* Send notifications. */
	CyBle_GattsNotification(cyBle_connHandle, &BiosignalNotificationHandle);
}


/******************************************************************************
* Function Name: UpdateBdSettings
*******************************************************************************
* Summary:
* Receive the settings data and apply parameters. Also, update the read
* characteristic handle so that the next read from the BLE central device
* gives present settings.
*
* Parameters:
*  void
*
* Return:
*  void
*
******************************************************************************/
void UpdateBdSettings(void)
{
	uint8 ch1_ctrl = bdControlData[CH1_CTRL_INDEX];
	uint8 ch2_ctrl = bdControlData[CH2_CTRL_INDEX];
	uint8 smpl_freq = bdControlData[SMPL_FREQ_INDEX];
	uint8 led = bdControlData[LED_INDEX];
	
    setChCtrl(CH1_INDEX, ch1_ctrl);
    setChCtrl(CH2_INDEX, ch2_ctrl);
    
    setSampleFreq(smpl_freq);
    
    setLED(led & BIT_MASK_LED);
	
	bdControlHandle.attrHandle = BD_CONTROL_CHAR_HANDLE;
	bdControlHandle.value.val = bdControlData;
	bdControlHandle.value.len = sizeof(bdControlData);
	bdControlHandle.value.actualLen = sizeof(bdControlData);
	
	/* Send updated BD control handle as attribute for read by central device */
	CyBle_GattsWriteAttributeValue(&bdControlHandle, FALSE, &cyBle_connHandle, FALSE);  
}

/* [] END OF FILE */
