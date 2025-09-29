/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
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
#include "main.h"
#include "cmsis_os.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include <stdio.h>
#include <string.h>
#include "mfrc522.h"

/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
#ifdef __GNUC__
/* With GCC, small printf (option LD Linker->Libraries->Small printf
   set to 'Yes') calls __io_putchar() */
#define PUTCHAR_PROTOTYPE int __io_putchar(int ch)
#else
#define PUTCHAR_PROTOTYPE int fputc(int ch, FILE *f)
#endif /* __GNUC__ */

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
SPI_HandleTypeDef hspi2;

TIM_HandleTypeDef htim2;
TIM_HandleTypeDef htim3;

UART_HandleTypeDef huart2;
UART_HandleTypeDef huart6;
DMA_HandleTypeDef hdma_usart6_rx;

/* Definitions for UartTask */
osThreadId_t UartTaskHandle;
const osThreadAttr_t UartTask_attributes = {
  .name = "UartTask",
  .stack_size = 256 * 4,
  .priority = (osPriority_t) osPriorityHigh,
};
/* Definitions for RfidTask */
osThreadId_t RfidTaskHandle;
const osThreadAttr_t RfidTask_attributes = {
  .name = "RfidTask",
  .stack_size = 128 * 4,
  .priority = (osPriority_t) osPriorityNormal,
};
/* Definitions for VibTask */
osThreadId_t VibTaskHandle;
const osThreadAttr_t VibTask_attributes = {
  .name = "VibTask",
  .stack_size = 128 * 4,
  .priority = (osPriority_t) osPriorityLow,
};
/* Definitions for PostureTask */
osThreadId_t PostureTaskHandle;
const osThreadAttr_t PostureTask_attributes = {
  .name = "PostureTask",
  .stack_size = 128 * 4,
  .priority = (osPriority_t) osPriorityLow,
};
/* Definitions for uartQueue */
osMessageQueueId_t uartQueueHandle;
const osMessageQueueAttr_t uartQueue_attributes = {
  .name = "uartQueue"
};
/* Definitions for uartMutex */
osMutexId_t uartMutexHandle;
const osMutexAttr_t uartMutex_attributes = {
  .name = "uartMutex"
};
/* Definitions for vibEvent */
osEventFlagsId_t vibEventHandle;
const osEventFlagsAttr_t vibEvent_attributes = {
  .name = "vibEvent"
};
/* Definitions for posEvent */
osEventFlagsId_t posEventHandle;
const osEventFlagsAttr_t posEvent_attributes = {
  .name = "posEvent"
};
/* USER CODE BEGIN PV */
#define RX_BUF_SIZE 256
#define TX_BUF_SIZE 256
#define ARR_CNT 4

#define VIB_ON (0x01U)

#define POSTURE_BAD (0x01U)
#define POSTURE_OK (0x02U)

uint8_t rxBuf[RX_BUF_SIZE];

volatile uint8_t rx_ch;
volatile uint8_t rx_index;
uint8_t uart6_flag = 0;
uint8_t uart6_rx_len = 0;
volatile uint8_t vib_flag = 0;	// 진동 기능 flag
volatile uint8_t vib_repeat = 0;	// 패턴 반복 횟수
static volatile uint8_t old = 0;
char msg_temp[RX_BUF_SIZE];
char* msg = msg_temp;

// RFID
uint8_t cardID[5];


//uint8_t status;
//uint8_t str[16];
uint8_t sNum[5];

char uid_str[16];
char uid_msg[TX_BUF_SIZE];

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_DMA_Init(void);
static void MX_USART2_UART_Init(void);
static void MX_USART6_UART_Init(void);
static void MX_TIM3_Init(void);
static void MX_TIM2_Init(void);
static void MX_SPI2_Init(void);
void StartUartTask(void *argument);
void StartRfidTask(void *argument);
void StartVibTask(void *argument);
void StartPostureTask(void *argument);

/* USER CODE BEGIN PFP */
void processMessage(char *msg);
void servo_unlock(void);
void servo_lock(void);
void send_status(const char *status);
void Motor_SetDutyPercent(uint8_t percent);
int Vib_PulsePattern(void);
void servo_setAngle(uint8_t angle);

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

/* 들어오는 메시지 처리 */
void processMessage(char *msg)
{
#if 1
	  /*=== 문자열 파싱 ===*/
	  if (msg == NULL || strlen(msg) == 0)
	  {
		printf("Error: Empty or NULL message received\r\n");
		return;
	  }

	  char buf[RX_BUF_SIZE];
	  strncpy(buf, msg, RX_BUF_SIZE - 1); // 안전한 복사
	  buf[RX_BUF_SIZE - 1] = '\0';        // 널 종료 보장

	  printf("Original Received: %s\r\n", buf);

	  char *pArray[ARR_CNT] = {0};
	  char *saveptr = NULL;
	  char *pToken = strtok_r(buf, ":", &saveptr);
	  int i = 0;
	  while (pToken && i < ARR_CNT)
	  {
		pArray[i++] = pToken;
		pToken = strtok_r(NULL, ":", &saveptr);
	  }

	  // 토큰 수가 충분한지 확인
	  if (i < 4)
	  {
		printf("Error: Invalid message format, too few tokens (%d)\r\n", i);
		return;
	  }

	  if (!strcmp(pArray[2], "DRAWER"))
	  {
		if (!strcmp(pArray[3], "UNLOCK"))
		{
		  printf(">>>>>debug : unlock\r\n");
		  servo_unlock();                    // 서랍 잠금해제
		  send_status("unlocked success\n"); // 상태 회신 : 잠금해제 성공
		}
		else if (!strcmp(pArray[3], "LOCK"))
		{
		  printf(">>>>>debug : lock\r\n");
		  servo_lock();                    // 서랍 잠금
		  send_status("locked success\n"); // 상태 회신 : 잠금 성공
		}
	  }
	  if (!strcmp(pArray[2], "SLP"))
	  {
		if (!strcmp(pArray[3], "ON"))
		{
		  printf(">>>>>debug : sleeping \r\n");
		  osEventFlagsSet(vibEventHandle, VIB_ON); // vibEventHandle에 VIB_ON 이벤트 발생
		}
		else if (!strcmp(pArray[3], "OFF"))
		{
		  printf(">>>>>debug : No sleeping \r\n");
		}
	  }
	  if (!strcmp(pArray[2], "POSTURE"))
	  {
		if (!strcmp(pArray[3], "BAD"))
		{
		  printf(">>>>>debug : bad posture \r\n");
		  printf("Running monitor tilt...\r\n");
		  osEventFlagsSet(posEventHandle, POSTURE_BAD); // posEventHandle에 POSTURE_BAD 이벤트 발생
		}
		else if (!strcmp(pArray[3], "OK"))
		{
		  printf(">>>>>debug : good posture \r\n");
		  printf("Running monitor reset...\r\n");
		  osEventFlagsSet(posEventHandle, POSTURE_OK); // posEventHandle에 POSTURE_OK 이벤트 발생
		}
	  }
#endif
#if 0
	if(strcmp(msg, "unlock") == 0)
	{
		servo_unlock();		// 서랍 잠금해제
		send_status("unlocked success\n");	// 상태 회신 : 잠금해제 성공
	}
	else if(strcmp(msg, "lock") == 0)
	{
		servo_lock();		// 서랍 잠금
		send_status("locked success\n");	// 상태 회신 : 잠금 성공
	}
	else if(strcmp(msg, "sleeping") == 0)
	{
	    printf("Running Vibration...\r\n");
	    osEventFlagsSet(vibEventHandle, VIB_ON);	// vibEventHandle에 VIB_ON 이벤트 발생
	}

	else if(strcmp(msg, "2") == 0)		// 좋은 자세일 때
	{
		printf("Running monitor reset...\r\n");
		osEventFlagsSet(posEventHandle, POSTURE_OK);	// posEventHandle에 POSTURE_OK 이벤트 발생
	}

	else if(strcmp(msg, "1") == 0)		// 나쁜 자세일 때
	{
		printf("Running monitor tilt...\r\n");
		osEventFlagsSet(posEventHandle, POSTURE_BAD);	// posEventHandle에 POSTURE_BAD 이벤트 발생
	}
	else
	{
		printf(">>> unknown \r\n");
	}
#endif
}

/* 메시지 전송 */
void send_status(const char *status)
{
//	osMutexAcquire(uartMutexHandle, osWaitForever);
	HAL_UART_Transmit(&huart6, (uint8_t*)status, strlen(status), HAL_MAX_DELAY);
//	osMutexRelease(uartMutexHandle);
}

/*================== 잠금서랍 기능 ==================*/
void servo_unlock(void)
{
	// PWM 제어 코드 (0 도)
	__HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_1, 500);	// 0도
	__HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_2, 1500);	// 90도
	printf("0도\r\n");

}

void servo_lock(void)
{
	// PWM 제어 코드 ( 90 도)
	__HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_1, 1500);	// 90도
	__HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_2, 500);	// 0도
	printf("90도\r\n");
}

/*================== 졸음 진동 기능 ==================*/
// 모터 듀티 설정 (percent 0~100)
void Motor_SetDutyPercent(uint8_t percent) {
    if (percent > 100) percent = 100;
    uint32_t arr = __HAL_TIM_GET_AUTORELOAD(&htim2); // 499
    uint32_t ccr = (arr + 1) * percent / 100;
    __HAL_TIM_SET_COMPARE(&htim2, TIM_CHANNEL_1, ccr);
    __HAL_TIM_SET_COMPARE(&htim2, TIM_CHANNEL_2, ccr);
}

// 진동 패턴 구현
#if 0
int Vib_PulsePattern(void) {
	static uint32_t last_tick = 0;
	static int state = 0;
	static int pulse_count = 0;

	uint32_t now = HAL_GetTick();

	switch(state)
	{
		case 0:		// 모터 on
			Motor_SetDutyPercent(70);	// 70%로 모터 동작
			last_tick = now;	// 현재 시간을 저장
			pulse_count = 0;	// 강약 반복 횟수
			state = 1;
			break;
		case 1:		// on 500ms -> off
			if(now - last_tick >= 500)	// 500ms 넘어가면
			{
				Motor_SetDutyPercent(0);	// 0%로 모터 동작
				last_tick = now;
				state = 2;
			}
			break;
		case 2:		// off 300 ms -> on
		{
			if(now - last_tick >= 300)	// 300ms 넘어가면
			{
				pulse_count++;
				if(pulse_count < 3)
				{
					Motor_SetDutyPercent(50);	// 50%로 모터 동작
					last_tick = now;
					state = 1;
				}
				else		// 강약 반복 3번 넘어가면
				{
					state = 0;
					return 1;
				}
			}
			break;
		}
	}
	return 0;

//    for (int i=0;i<3;i++) {
//        Motor_SetDutyPercent(50); HAL_Delay(500);
//        Motor_SetDutyPercent(0);  HAL_Delay(100);
//    }
}
#endif

/*================== 모니터 조절 기능 ==================*/
void servo_setAngle(uint8_t angle)
{
    /* angle: 0~180 범위 */
    if (angle > 180) angle = 180;

    /*
     * PWM 주기 설정값
     * PSC = 84-1, ARR = 20000-1 → 1 tick = 1 us, 주기 = 20 ms (50Hz)
     * Servo: 0도 ≈ 1ms 펄스, 180도 ≈ 2ms 펄스
     * 즉, CCR = 1000 ~ 2000 (us 단위)
     */
    uint16_t pulse = 1000 + ((angle * 1000) / 180);

    __HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_3, pulse);
}


/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{

  /* USER CODE BEGIN 1 */
//	uint8_t status;
//	uint8_t str[16];

  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_DMA_Init();
  MX_USART2_UART_Init();
  MX_USART6_UART_Init();
  MX_TIM3_Init();
  MX_TIM2_Init();
  MX_SPI2_Init();
  /* USER CODE BEGIN 2 */
  printf("main start()!!\r\n");

  if(HAL_TIM_PWM_Start(&htim3, TIM_CHANNEL_1) != HAL_OK)
	  Error_Handler();
  if(HAL_TIM_PWM_Start(&htim3, TIM_CHANNEL_2) != HAL_OK)
	  Error_Handler();
  if(HAL_TIM_PWM_Start(&htim3, TIM_CHANNEL_3) != HAL_OK)
	  Error_Handler();
  if(HAL_TIM_PWM_Start(&htim2, TIM_CHANNEL_1) != HAL_OK)
	  Error_Handler();
  if(HAL_TIM_PWM_Start(&htim2, TIM_CHANNEL_2) != HAL_OK)
	  Error_Handler();

  /*=== 인터럽트 방식 ===*/
//  HAL_UART_Receive_IT(&huart6, (uint8_t*)&rx_ch, 1);

  /*=== DMA 방식 ===*/
//  __HAL_UART_ENABLE_IT(&huart6, UART_IT_IDLE);	// IDLE 인터럽트 활성화
//  HAL_UART_Receive_DMA(&huart6, (uint8_t*)rxBuf, RX_BUF_SIZE);
// 	-> 옛날 방식

  HAL_UARTEx_ReceiveToIdle_DMA(&huart6, rxBuf, RX_BUF_SIZE);

  /*=== RFID ===*/
	MFRC522_Init();                               // RC522 초기화

  /* USER CODE END 2 */

  /* Init scheduler */
  osKernelInitialize();
  /* Create the mutex(es) */
  /* creation of uartMutex */
  uartMutexHandle = osMutexNew(&uartMutex_attributes);

  /* USER CODE BEGIN RTOS_MUTEX */
  /* add mutexes, ... */
  /* USER CODE END RTOS_MUTEX */

  /* USER CODE BEGIN RTOS_SEMAPHORES */
  /* add semaphores, ... */
  /* USER CODE END RTOS_SEMAPHORES */

  /* USER CODE BEGIN RTOS_TIMERS */
  /* start timers, add new ones, ... */
  /* USER CODE END RTOS_TIMERS */

  /* Create the queue(s) */
  /* creation of uartQueue */
  uartQueueHandle = osMessageQueueNew (16, 256, &uartQueue_attributes);

  /* USER CODE BEGIN RTOS_QUEUES */
  /* add queues, ... */
  /* USER CODE END RTOS_QUEUES */

  /* Create the thread(s) */
  /* creation of UartTask */
  UartTaskHandle = osThreadNew(StartUartTask, NULL, &UartTask_attributes);

  /* creation of RfidTask */
  RfidTaskHandle = osThreadNew(StartRfidTask, NULL, &RfidTask_attributes);

  /* creation of VibTask */
  VibTaskHandle = osThreadNew(StartVibTask, NULL, &VibTask_attributes);

  /* creation of PostureTask */
  PostureTaskHandle = osThreadNew(StartPostureTask, NULL, &PostureTask_attributes);

  /* USER CODE BEGIN RTOS_THREADS */
  /* add threads, ... */
  /* USER CODE END RTOS_THREADS */

  /* creation of vibEvent */
  vibEventHandle = osEventFlagsNew(&vibEvent_attributes);

  /* creation of posEvent */
  posEventHandle = osEventFlagsNew(&posEvent_attributes);

  /* USER CODE BEGIN RTOS_EVENTS */
  /* add events, ... */
  /* USER CODE END RTOS_EVENTS */

  /* Start scheduler */
  osKernelStart();

  /* We should never get here as control is now taken by the scheduler */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
  }
  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Configure the main internal regulator output voltage
  */
  __HAL_RCC_PWR_CLK_ENABLE();
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSI;
  RCC_OscInitStruct.PLL.PLLM = 16;
  RCC_OscInitStruct.PLL.PLLN = 336;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV4;
  RCC_OscInitStruct.PLL.PLLQ = 4;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV2;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_2) != HAL_OK)
  {
    Error_Handler();
  }
}

/**
  * @brief SPI2 Initialization Function
  * @param None
  * @retval None
  */
static void MX_SPI2_Init(void)
{

  /* USER CODE BEGIN SPI2_Init 0 */

  /* USER CODE END SPI2_Init 0 */

  /* USER CODE BEGIN SPI2_Init 1 */

  /* USER CODE END SPI2_Init 1 */
  /* SPI2 parameter configuration*/
  hspi2.Instance = SPI2;
  hspi2.Init.Mode = SPI_MODE_MASTER;
  hspi2.Init.Direction = SPI_DIRECTION_2LINES;
  hspi2.Init.DataSize = SPI_DATASIZE_8BIT;
  hspi2.Init.CLKPolarity = SPI_POLARITY_LOW;
  hspi2.Init.CLKPhase = SPI_PHASE_1EDGE;
  hspi2.Init.NSS = SPI_NSS_SOFT;
  hspi2.Init.BaudRatePrescaler = SPI_BAUDRATEPRESCALER_8;
  hspi2.Init.FirstBit = SPI_FIRSTBIT_MSB;
  hspi2.Init.TIMode = SPI_TIMODE_DISABLE;
  hspi2.Init.CRCCalculation = SPI_CRCCALCULATION_DISABLE;
  hspi2.Init.CRCPolynomial = 10;
  if (HAL_SPI_Init(&hspi2) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN SPI2_Init 2 */

  /* USER CODE END SPI2_Init 2 */

}

/**
  * @brief TIM2 Initialization Function
  * @param None
  * @retval None
  */
static void MX_TIM2_Init(void)
{

  /* USER CODE BEGIN TIM2_Init 0 */

  /* USER CODE END TIM2_Init 0 */

  TIM_MasterConfigTypeDef sMasterConfig = {0};
  TIM_OC_InitTypeDef sConfigOC = {0};

  /* USER CODE BEGIN TIM2_Init 1 */

  /* USER CODE END TIM2_Init 1 */
  htim2.Instance = TIM2;
  htim2.Init.Prescaler = 84-1;
  htim2.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim2.Init.Period = 1000-1;
  htim2.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim2.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  if (HAL_TIM_PWM_Init(&htim2) != HAL_OK)
  {
    Error_Handler();
  }
  sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
  if (HAL_TIMEx_MasterConfigSynchronization(&htim2, &sMasterConfig) != HAL_OK)
  {
    Error_Handler();
  }
  sConfigOC.OCMode = TIM_OCMODE_PWM1;
  sConfigOC.Pulse = 0;
  sConfigOC.OCPolarity = TIM_OCPOLARITY_HIGH;
  sConfigOC.OCFastMode = TIM_OCFAST_DISABLE;
  if (HAL_TIM_PWM_ConfigChannel(&htim2, &sConfigOC, TIM_CHANNEL_1) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_TIM_PWM_ConfigChannel(&htim2, &sConfigOC, TIM_CHANNEL_2) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM2_Init 2 */

  /* USER CODE END TIM2_Init 2 */
  HAL_TIM_MspPostInit(&htim2);

}

/**
  * @brief TIM3 Initialization Function
  * @param None
  * @retval None
  */
static void MX_TIM3_Init(void)
{

  /* USER CODE BEGIN TIM3_Init 0 */

  /* USER CODE END TIM3_Init 0 */

  TIM_MasterConfigTypeDef sMasterConfig = {0};
  TIM_OC_InitTypeDef sConfigOC = {0};

  /* USER CODE BEGIN TIM3_Init 1 */

  /* USER CODE END TIM3_Init 1 */
  htim3.Instance = TIM3;
  htim3.Init.Prescaler = 84-1;
  htim3.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim3.Init.Period = 20000-1;
  htim3.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim3.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  if (HAL_TIM_PWM_Init(&htim3) != HAL_OK)
  {
    Error_Handler();
  }
  sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
  if (HAL_TIMEx_MasterConfigSynchronization(&htim3, &sMasterConfig) != HAL_OK)
  {
    Error_Handler();
  }
  sConfigOC.OCMode = TIM_OCMODE_PWM1;
  sConfigOC.Pulse = 0;
  sConfigOC.OCPolarity = TIM_OCPOLARITY_HIGH;
  sConfigOC.OCFastMode = TIM_OCFAST_DISABLE;
  if (HAL_TIM_PWM_ConfigChannel(&htim3, &sConfigOC, TIM_CHANNEL_1) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_TIM_PWM_ConfigChannel(&htim3, &sConfigOC, TIM_CHANNEL_2) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_TIM_PWM_ConfigChannel(&htim3, &sConfigOC, TIM_CHANNEL_3) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM3_Init 2 */

  /* USER CODE END TIM3_Init 2 */
  HAL_TIM_MspPostInit(&htim3);

}

/**
  * @brief USART2 Initialization Function
  * @param None
  * @retval None
  */
static void MX_USART2_UART_Init(void)
{

  /* USER CODE BEGIN USART2_Init 0 */

  /* USER CODE END USART2_Init 0 */

  /* USER CODE BEGIN USART2_Init 1 */

  /* USER CODE END USART2_Init 1 */
  huart2.Instance = USART2;
  huart2.Init.BaudRate = 115200;
  huart2.Init.WordLength = UART_WORDLENGTH_8B;
  huart2.Init.StopBits = UART_STOPBITS_1;
  huart2.Init.Parity = UART_PARITY_NONE;
  huart2.Init.Mode = UART_MODE_TX_RX;
  huart2.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart2.Init.OverSampling = UART_OVERSAMPLING_16;
  if (HAL_UART_Init(&huart2) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN USART2_Init 2 */

  /* USER CODE END USART2_Init 2 */

}

/**
  * @brief USART6 Initialization Function
  * @param None
  * @retval None
  */
static void MX_USART6_UART_Init(void)
{

  /* USER CODE BEGIN USART6_Init 0 */

  /* USER CODE END USART6_Init 0 */

  /* USER CODE BEGIN USART6_Init 1 */

  /* USER CODE END USART6_Init 1 */
  huart6.Instance = USART6;
  huart6.Init.BaudRate = 115200;
  huart6.Init.WordLength = UART_WORDLENGTH_8B;
  huart6.Init.StopBits = UART_STOPBITS_1;
  huart6.Init.Parity = UART_PARITY_NONE;
  huart6.Init.Mode = UART_MODE_TX_RX;
  huart6.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart6.Init.OverSampling = UART_OVERSAMPLING_16;
  if (HAL_UART_Init(&huart6) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN USART6_Init 2 */

  /* USER CODE END USART6_Init 2 */

}

/**
  * Enable DMA controller clock
  */
static void MX_DMA_Init(void)
{

  /* DMA controller clock enable */
  __HAL_RCC_DMA2_CLK_ENABLE();

  /* DMA interrupt init */
  /* DMA2_Stream1_IRQn interrupt configuration */
  HAL_NVIC_SetPriority(DMA2_Stream1_IRQn, 5, 0);
  HAL_NVIC_EnableIRQ(DMA2_Stream1_IRQn);

}

/**
  * @brief GPIO Initialization Function
  * @param None
  * @retval None
  */
static void MX_GPIO_Init(void)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};
  /* USER CODE BEGIN MX_GPIO_Init_1 */

  /* USER CODE END MX_GPIO_Init_1 */

  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOC_CLK_ENABLE();
  __HAL_RCC_GPIOH_CLK_ENABLE();
  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_GPIOB_CLK_ENABLE();

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(GPIOA, RC522_CS_Pin|LD2_Pin, GPIO_PIN_RESET);

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(RC522_RST_GPIO_Port, RC522_RST_Pin, GPIO_PIN_RESET);

  /*Configure GPIO pin : B1_Pin */
  GPIO_InitStruct.Pin = B1_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_IT_FALLING;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(B1_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pins : RC522_CS_Pin LD2_Pin */
  GPIO_InitStruct.Pin = RC522_CS_Pin|LD2_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

  /*Configure GPIO pin : RC522_RST_Pin */
  GPIO_InitStruct.Pin = RC522_RST_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(RC522_RST_GPIO_Port, &GPIO_InitStruct);

  /* USER CODE BEGIN MX_GPIO_Init_2 */

  /* USER CODE END MX_GPIO_Init_2 */
}

/* USER CODE BEGIN 4 */
osMutexId_t uart2PrintMutex;

PUTCHAR_PROTOTYPE
{
  uint8_t c = (uint8_t)ch;
  if (osKernelGetState() == osKernelRunning && uart2PrintMutex)
  {
    if (osMutexAcquire(uart2PrintMutex, 50) == osOK)
    {
      HAL_UART_Transmit(&huart2, &c, 1, 50);
      osMutexRelease(uart2PrintMutex);
    }
    else
    {
      HAL_UART_Transmit(&huart2, &c, 1, 10);
    }
  }
  else
  {
    HAL_UART_Transmit(&huart2, &c, 1, 0xFFFF);
  }
  return ch;
}

/* === UART 메시지 인터럽트로 처리 === */
//void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart)
//{
//	if(huart->Instance == USART6)	// USART6에서 인터럽트 발생했을 때 처리
//	{
//		if(rx_ch == '\n')
//		 {
//			uart6_flag = 1;
//		}
//		else
//		{
//			if(rx_index < RX_BUF_SIZE -1)
//			{
//				rxBuf[rx_index++] = rx_ch;
//			}
//		}
//
//		HAL_UART_Receive_IT(&huart6, (uint8_t*)&rx_ch, 1);	// 다시 수신 요청
//	}
//}

/* === UART DMA로 처리 (옛날 방식) === */
// void USART6_IRQHandler(void)		// stm32f4xx_it.c안에 구현
// {
//		if (__HAL_UART_GET_FLAG(&huart6, UART_FLAG_IDLE)) {
//			__HAL_UART_CLEAR_IDLEFLAG(&huart6);
//			  uart6_flag = 1;  // main 루프에서 처리
//		}
//		HAL_UART_IRQHandler(&huart6);
// }

/* === UART DMA로 처리 === */
void HAL_UARTEx_RxEventCallback(UART_HandleTypeDef *huart, uint16_t Size)
{
	if(huart->Instance == USART6)
	{
// Non-freeRTOS
//		  memcpy(msg_temp, &rxBuf, Size-1);
//	      processMessage(msg_temp);					// msg_temp에 따라 기능 처리
//	      memset(msg_temp, 0, sizeof(msg_temp));
//
//	      HAL_UARTEx_ReceiveToIdle_DMA(&huart6, rxBuf, RX_BUF_SIZE);

		printf("[DBG] uart receive test \r\n");
		memcpy(msg_temp, &rxBuf, Size-1);	// '\n' 제거
		msg_temp[Size - 1] = '\0';			// 널문자 추가
		osMessageQueuePut(uartQueueHandle, msg_temp, 0, 0);		// FreeRTOS 큐로 메시지 전달
		//memset(msg_temp, 0, sizeof(msg_temp));
		HAL_UARTEx_ReceiveToIdle_DMA(&huart6, rxBuf, RX_BUF_SIZE);	// IDLE인터럽트 다시 활성화
	}
}

/* USER CODE END 4 */

/* USER CODE BEGIN Header_StartUartTask */
/**
* @brief Function implementing the UartTask thread.
* @param argument: Not used
* @retval None
*/
/* USER CODE END Header_StartUartTask */
void StartUartTask(void *argument)
{
  /* USER CODE BEGIN 5 */
	char recvBuf[RX_BUF_SIZE];
  /* Infinite loop */
  for(;;)
  {
	  printf("[DBG] UartTask waiting...\r\n");
	  	// 큐에 데이터가 없으면 block 상태가 됨 (Task가 실행을 멈추고 대기함)
	      if (osMessageQueueGet(uartQueueHandle, recvBuf, NULL, osWaitForever) == osOK)
	      {
	    	  printf("[DBG] UartTask: msg received: %s\r\n", recvBuf);
	          processMessage(recvBuf);  // 큐에서 메시지 꺼내서 processMessage(메시지처리) 함수 호출
	      }
    osDelay(100);
  }
  /* USER CODE END 5 */
}

/* USER CODE BEGIN Header_StartRfidTask */
/**
* @brief Function implementing the RfidTask thread.
* @param argument: Not used
* @retval None
*/
/* USER CODE END Header_StartRfidTask */
void StartRfidTask(void *argument)
{
  /* USER CODE BEGIN StartRfidTask */
	uint8_t status;
	uint8_t str[16];
	static uint8_t card_present = 0;



  /* Infinite loop */
  for(;;)
  {
	  //========= RFID 체크 ===============
//	  	  printf("[DBG] RfidTask running...\r\n");
	  		status = MFRC522_Request(PICC_REQIDL, str);		// 주기적으로 카드 감지 요청

	  		if (status == MI_OK && card_present == 0)   // 카드가 감지됨
	  		{
	  			printf("[DBG] RfidTask: card detected!\r\n");

	  		    status = MFRC522_Anticoll(str);			// 카드 UID 읽기: str[0] ~ [4]에 저장
	  		    if (status == MI_OK)   // UID 읽기 성공
	  		    {
	  		    	card_present = 1;	// 카드 감지된 상태

	  		        // UID를 문자열로 변환
	  	            sprintf(uid_str, "%02X%02X%02X%02X", str[0], str[1], str[2], str[3]);
	  	            printf("Card UID: %s\r\n", uid_str);

	  	            // Pi로 전송할 메시지 생성 및 전송
	  	            sprintf(uid_msg, "STM32:none:RFID:%s\n", uid_str);
	  	            printf("uid_msg : %s\r\n", uid_msg);
	  	            send_status(uid_msg);

	  	            osDelay(3000);	// task 3초 대기(block) 후 다시 카드 감지
	  	            card_present = 0;	// 카드 인식되지 않은 상태
	  		    }
	  		}
//	  		printf("[DBG] RfidTask end...\r\n");
    osDelay(100);
  }
  /* USER CODE END StartRfidTask */
}

/* USER CODE BEGIN Header_StartVibTask */
/**
* @brief Function implementing the VibTask thread.
* @param argument: Not used
* @retval None
*/
/* USER CODE END Header_StartVibTask */
void StartVibTask(void *argument)
{
  /* USER CODE BEGIN StartVibTask */
  /* Infinite loop */
  for(;;)
  {
	  // VIB_ON 이벤트가 올 때까지 Task가 무한 대기
	  printf("[DBG] VibTask waiting...\r\n");
	  osEventFlagsWait(vibEventHandle, VIB_ON, osFlagsWaitAny, osWaitForever);

	  printf("[DBG] VibTask triggered...\r\n");
	  // 이벤트가 들어오면 아래 코드 실행
	  send_status("vibration started\n");	// 상태 회신 : 진동 시작
	  for(int r = 0; r < 5; r++)		// r = 진동 패턴 반복 횟수
	  {
		  for(int i = 0; i < 3; i++)
		  {
			  Motor_SetDutyPercent(50);
//			  vTaskDelay(pdMS_TO_TICKS(500));
			  osDelay(500);
			  Motor_SetDutyPercent(0);
//			  vTaskDelay(pdMS_TO_TICKS(100));
			  osDelay(100);
			  printf(">> Debug : vibration \r\n");
		  }
		  printf(">> vib_repeat : %d \r\n", r);
	  }
	  send_status("vibration finished\n");	// 상태 회신 : 진동 성공

    osDelay(100);
  }
  /* USER CODE END StartVibTask */
}

/* USER CODE BEGIN Header_StartPostureTask */
/**
* @brief Function implementing the PostureTask thread.
* @param argument: Not used
* @retval None
*/
/* USER CODE END Header_StartPostureTask */
void StartPostureTask(void *argument)
{
  /* USER CODE BEGIN StartPostureTask */
	uint32_t posture_flag;
  /* Infinite loop */
  for(;;)
  {
//	  printf("[DBG] PostureTask waiting...\r\n");
	  posture_flag = osEventFlagsWait(posEventHandle, POSTURE_BAD | POSTURE_OK, osFlagsWaitAny, osWaitForever);
	  printf("[DBG] PostureTask triggered: flag=0x%lx\r\n", posture_flag);

	  if(posture_flag & POSTURE_BAD)
	  {
		  __HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_3, 2200);	// 모니터 각도 조절 서보모터 제어
		  send_status("monitor tilted up\n");
	  }
	  if(posture_flag & POSTURE_OK)
	  {
		  __HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_3, 2500);	// 모니터 각도 조절 원상복구
		  send_status("monitor reset\n");
	  }

	  osDelay(100);
  }
  /* USER CODE END StartPostureTask */
}

/**
  * @brief  Period elapsed callback in non blocking mode
  * @note   This function is called  when TIM10 interrupt took place, inside
  * HAL_TIM_IRQHandler(). It makes a direct call to HAL_IncTick() to increment
  * a global variable "uwTick" used as application time base.
  * @param  htim : TIM handle
  * @retval None
  */
void HAL_TIM_PeriodElapsedCallback(TIM_HandleTypeDef *htim)
{
  /* USER CODE BEGIN Callback 0 */

  /* USER CODE END Callback 0 */
  if (htim->Instance == TIM10)
  {
    HAL_IncTick();
  }
  /* USER CODE BEGIN Callback 1 */

  /* USER CODE END Callback 1 */
}

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}
#ifdef USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
