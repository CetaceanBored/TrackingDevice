#include "stm32f10x.h"
#include "stm32f10x_gpio.h"
#include "stm32f10x_rcc.h"
#include "stm32f10x_usart.h"
#include <stdint.h>

#define USART1_BurdRate 115200
#define USART2_BurdRate 115200

uint8_t Serial_RxFlag[2], Serial_TxFlag;
char Serial_Packet[2][100];

void Serial_UASRT1_init(void)      //USART1------PA9 = TX  PA10 = RX
{
    RCC_APB2PeriphClockCmd(RCC_APB2Periph_USART1, ENABLE);
    RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOA, ENABLE);

    GPIO_InitTypeDef GPIO_InitStruct;
    GPIO_InitStruct.GPIO_Mode = GPIO_Mode_AF_PP;
    GPIO_InitStruct.GPIO_Pin = GPIO_Pin_9;
    GPIO_InitStruct.GPIO_Speed = GPIO_Speed_50MHz;
    GPIO_Init(GPIOA, &GPIO_InitStruct);
	GPIO_InitStruct.GPIO_Mode = GPIO_Mode_IPU;
	GPIO_InitStruct.GPIO_Pin = GPIO_Pin_10;
	GPIO_Init(GPIOA, &GPIO_InitStruct);

    USART_InitTypeDef USART_InitStruct;
	USART_InitStruct.USART_BaudRate = USART1_BurdRate;
	USART_InitStruct.USART_HardwareFlowControl = USART_HardwareFlowControl_None;
	USART_InitStruct.USART_Mode = USART_Mode_Rx | USART_Mode_Tx;
	USART_InitStruct.USART_Parity = USART_Parity_No;
	USART_InitStruct.USART_StopBits = USART_StopBits_1;
	USART_InitStruct.USART_WordLength = USART_WordLength_8b;
	USART_Init(USART1, &USART_InitStruct);
	
	NVIC_PriorityGroupConfig(NVIC_PriorityGroup_2);
	USART_ITConfig(USART1, USART_IT_RXNE, ENABLE);
	NVIC_InitTypeDef NVIC_InitStruct;
	NVIC_InitStruct.NVIC_IRQChannel = USART1_IRQn;
	NVIC_InitStruct.NVIC_IRQChannelCmd = ENABLE;
	NVIC_InitStruct.NVIC_IRQChannelPreemptionPriority = 1;
	NVIC_InitStruct.NVIC_IRQChannelSubPriority = 1;
	NVIC_Init(&NVIC_InitStruct);

	USART_Cmd(USART1, ENABLE);
}

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

void Serial_SendByte(USART_TypeDef *USARTx, uint8_t Byte)
{
    USART_SendData(USARTx, Byte);
    while(USART_GetFlagStatus(USARTx, USART_FLAG_TXE) == RESET);
}

void Serial_SendString(USART_TypeDef *USARTx, char *String)
{
	Serial_TxFlag = 1;
	for (uint8_t i = 0; String[i] != '\0'; i ++) {
		Serial_SendByte(USARTx, String[i]);
	}
	Serial_TxFlag = 0;
}


void USART1_IRQHandler(void)
{
	static uint8_t RxState = 0, pRxPacket = 0;
	if (USART_GetITStatus(USART1, USART_IT_RXNE) == SET) {
		uint8_t RxData = USART_ReceiveData(USART1);
		if(RxState == 0) {
			if (Serial_RxFlag[0] == 0) {
				RxState = 1;
				pRxPacket = 0;
				Serial_Packet[0][pRxPacket++] = RxData;
			}
		}
		else if (RxState == 1) {
			if(RxData == '\r') {
				RxState = 2;
			}
			else {
				Serial_Packet[0][pRxPacket++] = RxData;
			}
		}
		else if(RxState == 2) {
			if (RxData == '\n') {
				RxState = 0;
				Serial_RxFlag[0] = 1;
				Serial_Packet[0][pRxPacket] = '\0';
			}
		}
	}
	USART_ClearITPendingBit(USART1, USART_IT_RXNE);
}


void USART2_IRQHandler(void)
{
	static uint8_t RxState = 0, pRxPacket = 0;
	if (USART_GetITStatus(USART2, USART_IT_RXNE) == SET) {
		uint8_t RxData = USART_ReceiveData(USART2);
		if(RxState == 0) {
			if (Serial_RxFlag[1] == 0) {
				RxState = 1;
				pRxPacket = 0;
				Serial_Packet[1][pRxPacket++] = RxData;
			}
		}
		else {
			if (RxData == '@') {
				RxState = 0;
				Serial_RxFlag[1] = 1;
				Serial_Packet[1][pRxPacket] = '\0';
			}
			else {
				Serial_Packet[1][pRxPacket++] = RxData;
			}
		}
	}
	USART_ClearITPendingBit(USART2, USART_IT_RXNE);
}

uint8_t Serial_GetFlag(uint8_t x)
{
	if (Serial_RxFlag[x] == 1) {
		Serial_RxFlag[x] = 0;
		return 1;
	}
	return 0;
}

