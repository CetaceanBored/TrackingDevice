#include <stdint.h>

#ifndef __SERIAL_H_
#define __SERIAL_H_

extern uint8_t Serial_RxFlag, Serial_TxFlag;
extern char Serial_Packet[100];
void Serial_UASRT2_init(void);
void Serial_SendByte(uint8_t Byte);
void Serial_SendString(char *String);

#endif
