#include "stm32f10x.h"
#include "stm32f10x_gpio.h"
#include "stm32f10x_rcc.h"
#include <stdint.h>

#include "OLED.h"

void Key_Init(void)
{
    RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOA, ENABLE);
    RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOE, ENABLE);

    GPIO_InitTypeDef GPIO_InitStructure;
    GPIO_InitStructure.GPIO_Mode = GPIO_Mode_IPU;
    GPIO_InitStructure.GPIO_Pin = GPIO_Pin_2 | GPIO_Pin_3 | GPIO_Pin_4;
    GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
    GPIO_Init(GPIOE, &GPIO_InitStructure);
    
    GPIO_InitStructure.GPIO_Mode = GPIO_Mode_IPD;
    GPIO_InitStructure.GPIO_Pin = GPIO_Pin_0;
    GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
    GPIO_Init(GPIOA, &GPIO_InitStructure);
}

uint8_t Key(void)
{
    static uint8_t last, cnt;
    last = cnt;
    
    if (GPIO_ReadInputDataBit(GPIOE, GPIO_Pin_4) == RESET)   cnt = 1;
    else if (GPIO_ReadInputDataBit(GPIOE, GPIO_Pin_3) == RESET) cnt = 2;
    else if (GPIO_ReadInputDataBit(GPIOE, GPIO_Pin_2) == RESET) cnt = 3;
    else if (GPIO_ReadInputDataBit(GPIOA, GPIO_Pin_0) == SET) cnt = 4;
    else cnt = 0;

    if(last && !cnt)
    {
        uint8_t KeyValue = last;
        last = 0;
        return KeyValue;
    }
    else return 0;
}