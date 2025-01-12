#include "stm32f10x.h"
#include "Delay.h"
#include "PWM.h"
#include "Serial.h"
#include "OLED.h"
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

int main(void)
{
	Delay_init();
	OLED_Init();
	Serial_UASRT2_init();
	PWM_Init();
	while (1)
	{
		OLED_ShowString(6, 2, "Testing...", 6);
		if (Serial_RxFlag) {
			//OLED_ShowString(1, 0, Serial_Packet, 6);
			char str1[10], str2[10];
			uint16_t val1, val2;
			strncpy(str1, Serial_Packet, 4);
			strncpy(str2, Serial_Packet + 4, 4);
			str1[4] = '\0', str2[4] = '\0';
			val1 = atoi(str1), val2 = atoi(str2);
			OLED_ShowNum(0, 0, val1, 4, 6);
			OLED_ShowNum(0, 30, val2, 4, 6);
			PWM_Set(1, val1);
			PWM_Set(2, val2);
			Serial_RxFlag = 0;
		}	
	}
}
