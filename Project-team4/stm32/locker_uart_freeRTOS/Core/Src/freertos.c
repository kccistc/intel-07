/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * File Name          : freertos.c
  * Description        : Code for freertos applications
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2025 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */

/* Includes ------------------------------------------------------------------*/
#include "FreeRTOS.h"
#include "task.h"
#include "main.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */

/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
/* USER CODE BEGIN Variables */
//osMessageQueueId_t uartQueueHandle;		// 변수 선언(전역 핸들)

/* USER CODE END Variables */

/* Private function prototypes -----------------------------------------------*/
/* USER CODE BEGIN FunctionPrototypes */

//void StartUartTask(void *argument);		// 프로토타입 선언


/* USER CODE END FunctionPrototypes */

/* Private application code --------------------------------------------------*/
/* USER CODE BEGIN Application */

//void MX_FREERTOS_Init(void) {
//  uartQueueHandle = osMessageQueueNew(16, RX_BUF_SIZE, NULL);	// 큐 생성
//  osThreadNew(StartUartTask, NULL, NULL);	// 태스크 생성
//}
//
//void StartUartTask(void *argument)
//{
//    char recvBuf[RX_BUF_SIZE];
//
//    for(;;)
//    {
//    	// 큐에서 메시지 꺼내서 processMessage(메시지처리) 함수 호출
//    	// 큐에 데이터가 없으면 block 상태가 됨 (실행을 멈추고 대기함)
//        if (osMessageQueueGet(uartQueueHandle, recvBuf, NULL, osWaitForever) == osOK)
//        {
//            processMessage(recvBuf);  // main.c에 이미 구현된 함수 호출
//        }
//    }
//}

/* USER CODE END Application */

