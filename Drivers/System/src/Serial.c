#include "stm32f10x.h"
#include "stm32f10x_gpio.h"
#include "stm32f10x_rcc.h"
#include "stm32f10x_usart.h"
#include <stdint.h>
#include <string.h>
#include "OLED.h"

#define USART2_BurdRate 9600

uint8_t Serial_RxFlag, Serial_TxFlag;
char Serial_Packet[100];

void Serial_UASRT2_init(void)      //USART2------PA2 = TX  PA3 = RX
{
    RCC_APB1PeriphClockCmd(RCC_APB1Periph_USART2, ENABLE);
    RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOA, ENABLE);

    GPIO_InitTypeDef GPIO_InitStruct;
    GPIO_InitStruct.GPIO_Mode = GPIO_Mode_AF_PP;
    GPIO_InitStruct.GPIO_Pin = GPIO_Pin_2;
    GPIO_InitStruct.GPIO_Speed = GPIO_Speed_50MHz;
    GPIO_Init(GPIOA, &GPIO_InitStruct);
	GPIO_InitStruct.GPIO_Mode = GPIO_Mode_IPU;
	GPIO_InitStruct.GPIO_Pin = GPIO_Pin_3;
	GPIO_Init(GPIOA, &GPIO_InitStruct);

    USART_InitTypeDef USART_InitStruct;
	USART_InitStruct.USART_BaudRate = USART2_BurdRate;
	USART_InitStruct.USART_HardwareFlowControl = USART_HardwareFlowControl_None;
	USART_InitStruct.USART_Mode = USART_Mode_Rx | USART_Mode_Tx;
	USART_InitStruct.USART_Parity = USART_Parity_No;
	USART_InitStruct.USART_StopBits = USART_StopBits_1;
	USART_InitStruct.USART_WordLength = USART_WordLength_8b;
	USART_Init(USART2, &USART_InitStruct);
    
	NVIC_PriorityGroupConfig(NVIC_PriorityGroup_2);
    USART_ITConfig(USART2, USART_IT_RXNE, ENABLE);
    NVIC_InitTypeDef NVIC_InitStruct;
	NVIC_InitStruct.NVIC_IRQChannel = USART2_IRQn;
	NVIC_InitStruct.NVIC_IRQChannelCmd = ENABLE;
	NVIC_InitStruct.NVIC_IRQChannelPreemptionPriority = 1;
	NVIC_InitStruct.NVIC_IRQChannelSubPriority = 1;
	NVIC_Init(&NVIC_InitStruct);

	USART_Cmd(USART2, ENABLE);
}

void Serial_SendByte(uint8_t Byte)
{
    USART_SendData(USART2, Byte);
    while(USART_GetFlagStatus(USART2, USART_FLAG_TXE) == RESET);
}

void Serial_SendString(char *String)
{
	Serial_TxFlag = 1;
	for (uint8_t i = 0; String[i] != '\0'; i ++) {
		Serial_SendByte(String[i]);
	}
	Serial_TxFlag = 0;
}

void USART2_IRQHandler(void)
{
	static uint8_t RxState = 0, pRxPacket = 0;
	if (USART_GetITStatus(USART2, USART_IT_RXNE) == SET) {
		uint8_t RxData = USART_ReceiveData(USART2);
		if (RxState == 0) {
			if (Serial_RxFlag == 0 && RxData == '@') {
				RxState = 1;
				pRxPacket = 0;
				memset(Serial_Packet, 0, sizeof(Serial_Packet));
			}
		}
		else {
			if (RxData == '#') {
				RxState = 0;
				Serial_RxFlag = 1;
				Serial_Packet[pRxPacket] = '\0';
			}
			else {
				Serial_Packet[pRxPacket++] = RxData;
			}
		}
	}
	USART_ClearITPendingBit(USART2, USART_IT_RXNE);
}
