#include <stdint.h>
#ifndef __PWM_H_
#define __PWM_H_

void PWM_Init(void);
void PWM_Set(uint8_t channel, uint16_t compare);

#endif
