#include "stm32f10x.h"
#include "Delay.h"
#include "PWM.h"
#include "Serial.h"
#include "OLED.h"
#include <stdint.h>

int main(void)
{	
	Delay_init();
	PWM_Init();
	while (1)
	{
		for (uint8_t i = 10; i <= 170; i+=5)
		{
			PWM_Set(1, (i / 180.0) * 1000 + 250);
			Delay_ms(200);
		}
		for (uint8_t i = 170; i >= 10; i-=5)
		{
			PWM_Set(1, (i / 180.0) * 1000 + 250);
			Delay_ms(200);
		}
	}
}
