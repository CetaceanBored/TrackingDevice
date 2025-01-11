#include "stm32f10x.h"
#include "Delay.h"
#include "PWM.h"
#include "Serial.h"
#include "OLED.h"
#include <stdint.h>


int main(void)
{
	Delay_init();
	OLED_Init();
	Serial_UASRT2_init();
	//PWM_Init();
	while (1)
	{
		OLED_ShowString(6, 2, "Testing...", 6);
		/*
		for (uint8_t i = 10; i <= 170; i+=5)
		{
			OLED_ShowNum(0, 0, i, 3, 8);
			PWM_Set(1, (i / 180.0) * 1000 + 250);
			Delay_ms(200);
		}
		for (uint8_t i = 170; i >= 10; i-=5)
		{
			OLED_ShowNum(0, 0, i, 3, 8);
			PWM_Set(1, (i / 180.0) * 1000 + 250);
			Delay_ms(200);
		}
		*/
		if (Serial_RxFlag) {
			
			OLED_ShowString(1, 0, Serial_Packet, 8);
			Serial_RxFlag = 0;
		}
		

		
	}
}
