#include "stm32f10x.h"

void Delay_init(void)
{
	SysTick_CLKSourceConfig(SysTick_CLKSource_HCLK_Div8);
}

void Delay_us(uint32_t us)
{
	SysTick->LOAD = us * 72 / 8;
	SysTick->VAL = 0x00;
	SysTick->CTRL |= SysTick_CTRL_ENABLE_Msk;
	while(!(SysTick->CTRL & (1 << 16)));
	SysTick->CTRL &= ~SysTick_CTRL_ENABLE_Msk;
}

void Delay_ms(uint32_t ms)
{
	while(ms--)	Delay_us(1000);
}

void Delay_s(uint32_t s)
{
	while(s--)	Delay_ms(1000);
}
