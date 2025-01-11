#include "stm32f10x.h"

#ifndef __DELAY_H_
#define __DELAY_H_

void Delay_init(void);
void Delay_us(uint32_t us);
void Delay_ms(uint32_t ms);
void Delay_s(uint32_t s);

#endif
