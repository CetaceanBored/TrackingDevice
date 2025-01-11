#include "stm32f10x.h"
#include "OLED_Font.h"

#define OLED_GPIO_PORT 			GPIOA
#define OLED_SCL_GPIO_PIN		GPIO_Pin_1
#define OLED_SDA_GPIO_PIN		GPIO_Pin_4
#define OLED_GPIO_CLK_ENABLE()	RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOA, ENABLE)

void OLED_SCL(uint8_t x)
{	
	GPIO_WriteBit(OLED_GPIO_PORT, OLED_SCL_GPIO_PIN, (BitAction)(x));
}

void OLED_SDA(uint8_t x)
{	
	GPIO_WriteBit(OLED_GPIO_PORT, OLED_SDA_GPIO_PIN, (BitAction)(x));
}

void OLED_I2C_Init(void)
{
    OLED_GPIO_CLK_ENABLE();
	
	GPIO_InitTypeDef GPIO_InitStructure;
	
 	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_Out_OD;
	GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
	
	GPIO_InitStructure.GPIO_Pin = OLED_SCL_GPIO_PIN;
 	GPIO_Init(OLED_GPIO_PORT, &GPIO_InitStructure);
	
	GPIO_InitStructure.GPIO_Pin = OLED_SDA_GPIO_PIN;
 	GPIO_Init(OLED_GPIO_PORT, &GPIO_InitStructure);

	OLED_SCL(1);
	OLED_SDA(1);
}

void OLED_I2C_Start(void)
{
	OLED_SDA(1);
	OLED_SCL(1);
	OLED_SDA(0);
	OLED_SCL(0);
}

void OLED_I2C_Stop(void)
{
	OLED_SDA(0);
	OLED_SCL(1);
	OLED_SDA(1);
}

void OLED_I2C_SendByte(uint8_t Byte)
{
	uint8_t i;
	for (i = 0; i < 8; i++)
	{
		OLED_SDA(Byte & (0x80 >> i));
		OLED_SCL(1);
		OLED_SCL(0);
	}
	OLED_SCL(1);
	OLED_SCL(0);
}

void OLED_SendCommand(uint8_t Command)
{
	OLED_I2C_Start();
	OLED_I2C_SendByte(0x78);
	OLED_I2C_SendByte(0x00);
	OLED_I2C_SendByte(Command); 
	OLED_I2C_Stop();
}

void OLED_SendData(uint8_t Data)
{
	OLED_I2C_Start();
	OLED_I2C_SendByte(0x78);
	OLED_I2C_SendByte(0x40);
	OLED_I2C_SendByte(Data);
	OLED_I2C_Stop();
}

void OLED_SetCursor(uint8_t Y, uint8_t X)
{
	OLED_SendCommand(0xB0 | Y);
	OLED_SendCommand(0x10 | ((X & 0xF0) >> 4));
	OLED_SendCommand(0x00 | (X & 0x0F));
}

void OLED_Clear(void)
{  
	uint8_t i, j;
	for (j = 0; j < 8; j++)
	{
		OLED_SetCursor(j, 0);
		for(i = 0; i < 128; i++)
		{
			//Buff[j][i] = 0x00;
			OLED_SendData(0x00);
		}
	}
}

void OLED_ShowImage(int16_t Y, int16_t X, uint8_t Width, uint8_t Height, const uint8_t *Image)
{
	for (uint8_t i = 0; i < Height; i++)
	{
		OLED_SetCursor(Y + i, X);
		for (uint8_t j = 0; j < Width; j++)
		{
			OLED_SendData(Image[Width * i + j]);
		}
	}
}

void OLED_ShowChar(int16_t Y, int16_t X, char Char, uint8_t Font)
{
	if(Font == 6)
	{
		OLED_SetCursor(Y, X);
		for (uint8_t i = 0; i < 6; i++)
		{
			OLED_SendData(OLED_F6x8[Char - ' '][i]);
		}
	}
	else if(Font == 8)
	{
		OLED_SetCursor(Y, X);
		for (uint8_t i = 0; i < 8; i++)
		{
			OLED_SendData(OLED_F8x16[Char - ' '][i]);
		}
		OLED_SetCursor(Y + 1, X);
		for (uint8_t i = 0; i < 8; i++)
		{
			OLED_SendData(OLED_F8x16[Char - ' '][i + 8]);
		}
	}
}

uint32_t OLED_Pow(uint32_t X, uint32_t Y)	//return X^Y
{
	uint32_t res = 1;
	while (Y--)	res *= X;
	return res;
}

void OLED_ShowString(int16_t Y, int16_t X, char *String, uint8_t Font)
{
	for (uint8_t i = 0; String[i] != '\0'; i++)
	{
		OLED_ShowChar(Y, X + i * Font, String[i], Font);
	}
}

void OLED_ShowNum(uint8_t Y, uint8_t X, uint32_t Num, uint8_t Length, uint8_t Font)
{
	for (uint8_t i = 0; i < Length; i++)							
	{
		OLED_ShowChar(Y, X + i * Font, Num / OLED_Pow(10, Length - i - 1) % 10 + '0', Font);
	}
}

void OLED_Init(void)
{	
	uint32_t i, j;
	for (i = 0; i < 1000; i++)
		for (j = 0; j < 1000; j++);
	
	OLED_I2C_Init();
	
	OLED_SendCommand(0xAE);
	
	OLED_SendCommand(0xD5);
	OLED_SendCommand(0x80);
	
	OLED_SendCommand(0xA8);
	OLED_SendCommand(0x3F);
	
	OLED_SendCommand(0xD3);
	OLED_SendCommand(0x00);
	
	OLED_SendCommand(0x40);
	
	OLED_SendCommand(0xA1);
	
	OLED_SendCommand(0xC8);

	OLED_SendCommand(0xDA);
	OLED_SendCommand(0x12);
	
	OLED_SendCommand(0x81);
	OLED_SendCommand(0xCF);

	OLED_SendCommand(0xD9);
	OLED_SendCommand(0xF1);

	OLED_SendCommand(0xDB);
	OLED_SendCommand(0x30);

	OLED_SendCommand(0xA4);

	OLED_SendCommand(0xA6);

	OLED_SendCommand(0x8D);
	OLED_SendCommand(0x14);

	OLED_SendCommand(0xAF);
		
	OLED_Clear();
}
