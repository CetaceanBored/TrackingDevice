#include "stm32f10x.h"
#include <stdint.h>

#ifndef __SERIAL_H_
#define __SERIAL_H_

extern uint8_t Serial_Flag[2];
extern char Serial_Packet[2][100];

void Serial_UASRT1_init(void);
void Serial_UASRT2_init(void);
void Serial_SendByte(USART_TypeDef *USARTx, uint8_t Byte);
void Serial_SendString(USART_TypeDef *USARTx, char *String);
uint8_t Serial_GetFlag(uint8_t x);

#endif
