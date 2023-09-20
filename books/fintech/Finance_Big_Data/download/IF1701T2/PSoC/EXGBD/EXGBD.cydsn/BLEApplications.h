/******************************************************************************
* File name:   BLEApplications.h
* Version:     1.0.0
* Discription:
*   This file defines the commonly used macros for BLE functionality.
* History:
*   1.0.0  2016.07.28  Initial coding
*
* Copyright 2016 METOOL Inc. All Rights Reserved
******************************************************************************/

#if !defined(_BLE_APPLICATIONS_H)
#define _BLE_APPLICATIONS_H

/******************************************************************************
* Included headers
******************************************************************************/
#include <project.h>


/******************************************************************************
* Macros 
******************************************************************************/
#define BIOSIGNAL_SERVICE_INDEX         (CYBLE_BIOSIGNAL_SERVICE_SERVICE_INDEX)

#define BIOSIGNAL_DATA_CHAR_INDEX       (CYBLE_BIOSIGNAL_SERVICE_BIOSIGNAL_DATA_CHARACTERISTIC_CHAR_INDEX)
#define BD_CONTROL_CHAR_INDEX           (CYBLE_BIOSIGNAL_SERVICE_BD_CONTROL_CHARACTERISTIC_CHAR_INDEX)

#define BIOSIGNAL_DATA_CHAR_HANDLE		(CYBLE_BIOSIGNAL_SERVICE_BIOSIGNAL_DATA_CHARACTERISTIC_CHAR_HANDLE)
#define BD_CONTROL_CHAR_HANDLE		    (CYBLE_BIOSIGNAL_SERVICE_BD_CONTROL_CHARACTERISTIC_CHAR_HANDLE)

#define BIOSIGNAL_DATA_CCC_INDEX	    (CYBLE_BIOSIGNAL_SERVICE_BIOSIGNAL_DATA_CHARACTERISTIC_CLIENT_CHARACTERISTIC_CONFIGURATION_DESC_INDEX)
#define BIOSIGNAL_DATA_CCC_HANDLE	    (CYBLE_BIOSIGNAL_SERVICE_BIOSIGNAL_DATA_CHARACTERISTIC_CLIENT_CHARACTERISTIC_CONFIGURATION_DESC_HANDLE)

#define CCC_DATA_INDEX					(0u)

#define BLE_STATE_ADVERTISING			(0x01)
#define BLE_STATE_CONNECTED				(0x02)
#define BLE_STATE_DISCONNECTED			(0x00)

#define MTU_XCHANGE_DATA_LEN			(0x0020)
    
    
/******************************************************************************
* Extern variables
******************************************************************************/
extern uint8 deviceConnected;
extern uint8 sendBiosignalDataNotifications;


/******************************************************************************
* Public functions
******************************************************************************/
void CustomEventHandler(uint32 event, void *eventParam);
void SendBiosignalDataNotification(int16 *bioData);
void UpdateBdSettings(void);
void switchGain(uint8 ch, uint8 val);

#endif  /* #if !defined(_BLE_APPLICATIONS_H) */

/* [] END OF FILE */
