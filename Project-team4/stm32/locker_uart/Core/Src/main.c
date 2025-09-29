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
DMA_HandleTypeDef hdma_usart6_tx;

/* USER CODE BEGIN PV */
#define RX_BUF_SIZE 256
#define TX_BUF_SIZE 256
#define ARR_CNT 4
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
char uid_str[16];
static volatile int authentication_flag = 0;
static volatile int auth_start_time = -1; // 인증 시작 시점의 tim3Sec 저장
static uint8_t card_present = 0;

uint8_t status;
uint8_t str[16];
uint8_t sNum[5];

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
void processMessage(char* msg)
{
	/*=== 문자열 파싱 ===*/
	int i = 0;
	char * pToken;
	char * pArray[ARR_CNT]={0};

	printf("Original Received : %s\r\n", msg);

	pToken = strtok(msg,":");
	while(pToken != NULL)
	{
	    pArray[i] = pToken;
	    if(++i >= ARR_CNT)
	      break;
	    pToken = strtok(NULL,":");
	}
	if(!strcmp(pArray[2],"DRAWER"))
	{
	 	if(!strcmp(pArray[3],"UNLOCK"))
	  	{
	  		printf(">>>>>debug : unlock\r\n");
			servo_unlock();		// 서랍 잠금해제
			send_status("unlocked success\n");	// 상태 회신 : 잠금해제 성공
	  	}
		else if(!strcmp(pArray[3],"LOCK"))
		{
			printf(">>>>>debug : lock\r\n");
			servo_lock();		// 서랍 잠금
			send_status("locked success\n");	// 상태 회신 : 잠금 성공
		}
	}
	if(!strcmp(pArray[2],"SLP"))
	{
	 	if(!strcmp(pArray[3],"ON"))
	  	{
	  		printf(">>>>>debug : sleeping \r\n");
//		    for(int i = 0; i < 5; i ++)
//		    {
//			    Vib_PulsePattern(); // "두두둑" 패턴 실행
//			    HAL_Delay(800);
//		    }
//		    send_status("vibration success\n");	// 상태 회신 : 진동 성공
		    vib_repeat = 5;		// 5번 반복
		    vib_flag = 1;		// 진동 시작
		    send_status("vibration started\n");	// 상태 회신 : 진동 성공

	  	}
		else if(!strcmp(pArray[3],"OFF"))
		{
			printf(">>>>>debug : No sleeping \r\n");
		}
	}
	if(!strcmp(pArray[2],"POSTURE"))
	{
	 	if(!strcmp(pArray[3],"BAD"))
	  	{
	  		printf(">>>>>debug : bad posture \r\n");
			__HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_3, 2200);
			send_status("monitor tilted up\n");	// 상태 회신 : 모니터 강제 조절 성공
	  	}
		else if(!strcmp(pArray[3],"OK"))
		{
			printf(">>>>>debug : good posture \r\n");
			__HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_3, 2500);
			send_status("monitor reset\n");	// 상태 회신 : 모니터 원상복구
		}
	}

#if 1
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
//	    for(int i = 0; i < 5; i ++)
//	    {
//		    Vib_PulsePattern(); // "두두둑" 패턴 실행
//		    HAL_Delay(800);
//	    }
	    vib_repeat = 5;		// 5번 반복
	    vib_flag = 1;		// 진동 시작
	    send_status("vibration started\n");	// 상태 회신 : 진동 성공
	}

	else if(strcmp(msg, "2") == 0)		// 좋은 자세일 때
	{
		printf("Running monitor reset...\r\n");
		__HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_3, 2500);
		send_status("monitor reset\n");	// 상태 회신 : 모니터 원상복구
	}

	else if(strcmp(msg, "1") == 0)		// 나쁜 자세일 때
	{
		printf("Running monitor tilt...\r\n");
		__HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_3, 2200);
		send_status("monitor tilted up\n");	// 상태 회신 : 모니터 강제 조절 성공
	}
	else if(strcmp(msg, "test") == 0)
	{
        // Pi로 전송
        HAL_UART_Transmit(&huart6, (uint8_t*)uid_str, strlen(uid_str), HAL_MAX_DELAY);
        HAL_UART_Transmit(&huart6, (uint8_t*)"\n", 1, HAL_MAX_DELAY);
	}
#endif

}

/* 메시지 전송 */
void send_status(const char *status)
{
	HAL_UART_Transmit(&huart6, (uint8_t*)status, strlen(status), HAL_MAX_DELAY);
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
	uint8_t status;
	uint8_t str[16];

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

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
//========= RFID 체크 ===============
		status = MFRC522_Request(PICC_REQIDL, str);		// 카드 타입

		if (status == MI_OK && card_present == 0)   // 카드가 감지됨
		{
		    status = MFRC522_Anticoll(str);			// 카드 UID : str[0] ~ [4]에 저장
		    if (status == MI_OK)   // UID 읽기 성공
		    {
		    	card_present = 1;	// 카드 감지된 상태

		        // UID를 문자열로 변환
	            sprintf(uid_str, "%02X%02X%02X%02X", str[0], str[1], str[2], str[3]);
	            printf("Card UID: %s\r\n", uid_str);

	            sprintf(uid_msg, "STM32:none:RFID:%s\n", uid_str);
	            printf("uid_msg : %s\r\n", uid_msg);

	            // Pi로 전송
	            send_status(uid_msg);

	            HAL_Delay(5000);
	            card_present = 0;	// 카드 인식되지 않은 상태
		    }
		}

//========= pi에서 메시지 들어온 경우 처리 ================
// 	-> 옛날 방식
#if 0
	  if(uart6_flag)
	  {
		  uart6_flag = 0;

		//=== DMA 처리 ===
		  uart6_rx_len = RX_BUF_SIZE - __HAL_DMA_GET_COUNTER(huart6.hdmarx);	// 수신된 길이 계산
		  if(uart6_rx_len == 0)	continue;							// 메시지 없으면 무시
	      if(rxBuf[uart6_rx_len - 1] == '\n') uart6_rx_len--;		// 메시지 끝 개행(\n) 제거

	      // 화면 출력
	      if(uart6_rx_len > old)	 // RX_BUF_SIZE보다 작은 메시지가 들어올 때
	      {
	    	  printf(">>Received: %.*s\r\n", uart6_rx_len-old, &rxBuf[old]);
	    	  memcpy(msg_temp, &rxBuf[old], uart6_rx_len-old);		// msg_temp에 rxBuf를 copy
	      }
	      else	// RX_BUF_SIZE보다 큰 메시지가 들어올 때
	      {
		      printf("Received: %.*s", RX_BUF_SIZE-old, &rxBuf[old]);
		      memcpy(msg_temp, &rxBuf[old], RX_BUF_SIZE-old);
		      if(uart6_rx_len) {
		    	  printf("%.*s", uart6_rx_len, &rxBuf[0]);
		    	  memcpy(msg_temp, &rxBuf[old], uart6_rx_len);
		      }
		      printf("\r\n");
	      }

//	      memcpy(msg_temp, &rxBuf, size-2);
	      processMessage(msg_temp);		// msg_temp에 따라 기능 처리
	      old = uart6_rx_len+1;		// old : 이전 메시지의 마지막 부분

	      memset(msg_temp, 0, sizeof(msg_temp));
	  }

#endif
	  //=== 진동 반복 처리 ===
	 if(vib_flag)
	 {
		 if(Vib_PulsePattern())
		 {
			 vib_repeat--;
			 if(vib_repeat <= 0) { vib_flag = 0; }
		 }
	 }

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
  HAL_NVIC_SetPriority(DMA2_Stream1_IRQn, 0, 0);
  HAL_NVIC_EnableIRQ(DMA2_Stream1_IRQn);
  /* DMA2_Stream6_IRQn interrupt configuration */
  HAL_NVIC_SetPriority(DMA2_Stream6_IRQn, 0, 0);
  HAL_NVIC_EnableIRQ(DMA2_Stream6_IRQn);

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
/**
  * @brief  Retargets the C library printf function to the USART.
  * @param  None
  * @retval None
  */
PUTCHAR_PROTOTYPE
{
  /* Place your implementation of fputc here */
  /* e.g. write a character to the USART6 and Loop until the end of transmission */
  HAL_UART_Transmit(&huart2, (uint8_t *)&ch, 1, 0xFFFF);

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

/* === UART DMA로 처리 === */
// void USART6_IRQHandler(void)		// stm32f4xx_it.c안에 구현
// {
//		if (__HAL_UART_GET_FLAG(&huart6, UART_FLAG_IDLE)) {
//			__HAL_UART_CLEAR_IDLEFLAG(&huart6);
//			  uart6_flag = 1;  // main 루프에서 처리
//		}
//		HAL_UART_IRQHandler(&huart6);
// }
// -> 옛날 방식


void HAL_UARTEx_RxEventCallback(UART_HandleTypeDef *huart, uint16_t Size)
{
	if(huart->Instance == USART6)
	{
		  memcpy(msg_temp, &rxBuf, Size-1);
	      processMessage(msg_temp);					// msg_temp에 따라 기능 처리
	      memset(msg_temp, 0, sizeof(msg_temp));

	      HAL_UARTEx_ReceiveToIdle_DMA(&huart6, rxBuf, RX_BUF_SIZE);
	}
}

/* USER CODE END 4 */

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
