#include "stm32f10x.h" 

#ifndef __OLED_H_
#define __OLED_H_

void OLED_Init(void);
void OLED_SendData(uint8_t Data);
void OLED_SetCursor(uint8_t Y, uint8_t X);
void OLED_ShowChar(int16_t X, int16_t Y, char Char, uint8_t Font);
void OLED_ShowString(int16_t Y, int16_t X, char *String, uint8_t Font);
void OLED_ShowNum(uint8_t Y, uint8_t X, uint32_t Num, uint8_t Length, uint8_t Font);

void OLED_SCL(uint8_t x);
void OLED_SDA(uint8_t x);
void OLED_I2C_Init(void);

#endif
