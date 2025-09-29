#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/fs.h>
#include <linux/init.h>
#include <linux/io.h>
#include <linux/uaccess.h>
#include <linux/cdev.h>
#include <linux/device.h>
#include <linux/interrupt.h>
#include <linux/wait.h>
#include <linux/poll.h>
#include <linux/of.h>
#include <linux/of_address.h>
#include <linux/of_irq.h>
#include <linux/spinlock.h>

#define DEVICE_NAME "uart3_raw"

// BCM2711 PL011 UART3 (Raspberry Pi 4)
#define UART3_BASE_PHYS 0xFE201600
#define UART3_REG_SIZE 0x90

/* 레지스터 오프셋 */
#define UART_DR 0x00 // Data
#define UART_FR 0x18 // Flag
#define UART_IBRD 0x24
#define UART_FBRD 0x28
#define UART_LCRH 0x2C
#define UART_CR 0x30
#define UART_IFLS 0x34
#define UART_IMSC 0x38
#define UART_RIS 0x3C
#define UART_MIS 0x40
#define UART_ICR 0x44

// FR 비트
#define UART_FR_TXFF (1 << 5) // TX FIFO Full
#define UART_FR_RXFE (1 << 4) // RX FIFO Empty

// IMSC/MIS/ICR 비트
#define RXIM (1 << 4)  // RX interrupt
#define TXIM (1 << 5)  // TX interrupt
#define RTIM (1 << 6)  // RX timeout
#define OEIM (1 << 10) // Overrun error

#define RXIC (1 << 4)
#define TXIC (1 << 5)
#define RTIC (1 << 6)
#define OEIC (1 << 10)

// 링버퍼
#define RB_SIZE 2048
struct ringbuf
{
    char buf[RB_SIZE];
    unsigned head;
    unsigned tail;
};

static inline bool rb_empty(struct ringbuf *r) { return r->head == r->tail; }
static inline bool rb_full(struct ringbuf *r) { return ((r->head + 1) % RB_SIZE) == r->tail; }
static inline void rb_put(struct ringbuf *r, char c)
{
    r->buf[r->head] = c;
    r->head = (r->head + 1) % RB_SIZE;
}
static inline char rb_get(struct ringbuf *r)
{
    char c = r->buf[r->tail];
    r->tail = (r->tail + 1) % RB_SIZE;
    return c;
}

// MMIO 베이스
static void __iomem *uart3_base;

// 문자 디바이스 자료구조
static struct cdev uart3_cdev;
static dev_t devno;
static struct class *uart3_class;
static int irq = -1; // 모듈 파라미터로 강제 지정 가능
module_param(irq, int, 0444);
MODULE_PARM_DESC(irq, "Override UART3 IRQ number (if -1, try from DT)");

// 동기화/대기
static spinlock_t uart_lock; // ISR <-> read/write 보호
static wait_queue_head_t rx_wq;
static wait_queue_head_t tx_wq;
static struct ringbuf rx_rb, tx_rb;

// cookie for shared IRQ (must match in free_irq)
static struct
{
    int cookie;
} irq_cookie;

// 선택: /dev 권한 0666
static char *uart3_devnode(const struct device *dev, umode_t *mode)
{
    if (mode)
        *mode = 0666;
    return NULL;
}

// DT에서 UART3 IRQ 구하기: /soc/serial@fe201600 노드 기준
static int get_uart3_irq_from_dt(void)
{
    struct device_node *np;
    int ret = -ENODEV;

    np = of_find_node_by_path("/soc/serial@fe201600");
    if (!np)
    {
        // 일부 트리에서는 버스 주소(0x7e201600)로 적히는 경우도 있어 보정
        np = of_find_node_by_path("/soc/serial@7e201600");
    }
    if (!np)
        return -ENODEV;

    ret = irq_of_parse_and_map(np, 0);
    of_node_put(np);
    if (ret <= 0)
        return -ENODEV;
    return ret;
}

static irqreturn_t uart3_irq_handler(int _irq, void *dev_id)
{
    u32 mis = readl(uart3_base + UART_MIS);
    if (!mis)
        return IRQ_NONE;

    // 잠금: ISR에서도 사용 → 스핀락
    unsigned long flags;
    spin_lock_irqsave(&uart_lock, flags);

    // RX: RXIM/RTIM/OEIM → FIFO 비울 만큼 싹 읽어 링버퍼에 push
    if (mis & (RXIM | RTIM | OEIM))
    {
        while (!(readl(uart3_base + UART_FR) & UART_FR_RXFE))
        {
            char c = readb(uart3_base + UART_DR); // 하위 8비트만 데이터
            if (!rb_full(&rx_rb))
                rb_put(&rx_rb, c);
            // 가득찼다면 버리거나 통계를 잡을 수 있음
        }
        // 인터럽트 클리어
        writel(RXIC | RTIC | OEIC, uart3_base + UART_ICR);
        // 블로킹 read 깨우기
        wake_up_interruptible(&rx_wq);
    }

    // TX: TXIM → FIFO 여유만큼 tx_rb에서 채워 넣기
    if (mis & TXIM)
    {
        while (!(readl(uart3_base + UART_FR) & UART_FR_TXFF))
        {
            if (rb_empty(&tx_rb))
            {
                // 더 보낼 게 없으면 TX 인터럽트 끔(폭주 방지)
                u32 im = readl(uart3_base + UART_IMSC);
                im &= ~TXIM;
                writel(im, uart3_base + UART_IMSC);
                break;
            }
            writeb(rb_get(&tx_rb), uart3_base + UART_DR);
        }
        writel(TXIC, uart3_base + UART_ICR);
        // write가 잠깐 막혔다가(버퍼 풀) 풀렸을 수 있으니 깨움
        wake_up_interruptible(&tx_wq);
    }

    spin_unlock_irqrestore(&uart_lock, flags);
    return IRQ_HANDLED;
}

static int uart3_open(struct inode *inode, struct file *file)
{
    unsigned long flags; // <-- FIX: 선언 추가

    // 0) 일단 비활성 + 인터럽트 마스크/클리어
    writel(0x0, uart3_base + UART_CR);    // UART Disable
    writel(0x0, uart3_base + UART_IMSC);  // 모든 UART IRQ 마스크
    writel(0x7FF, uart3_base + UART_ICR); // pending IRQ 클리어

    // 1) HW RX FIFO 드레인
    while (!(readl(uart3_base + UART_FR) & UART_FR_RXFE))
        (void)readl(uart3_base + UART_DR);

    // 2) SW 링버퍼 리셋 (ISR/동시 접근 대비)
    spin_lock_irqsave(&uart_lock, flags);
    rx_rb.head = rx_rb.tail = 0;
    tx_rb.head = tx_rb.tail = 0;
    spin_unlock_irqrestore(&uart_lock, flags);

    // 3) 보레이트/라인 설정
    writel(26, uart3_base + UART_IBRD); // 115200 @ 48MHz (예시)
    writel(3, uart3_base + UART_FBRD);
    writel((1 << 4) | (3 << 5), uart3_base + UART_LCRH); // FIFO + 8N1
    writel(0, uart3_base + UART_IFLS);                   // 기본 임계

    // 4) 수신 관련 인터럽트 언마스크 (TXIM은 write()에서 켬)
    writel(RXIM | RTIM | OEIM, uart3_base + UART_IMSC);

    // 5) UART Enable
    writel((1 << 0) | (1 << 8) | (1 << 9), uart3_base + UART_CR); // UARTEN | TXE | RXE

    return 0;
}

static int uart3_release(struct inode *inode, struct file *file)
{
    // 인터럽트 모두 끄기
    writel(0, uart3_base + UART_IMSC);
    return 0;
}

static ssize_t uart3_read(struct file *file, char __user *ubuf, size_t count, loff_t *ppos)
{
    size_t done = 0;
    int ret;

    if (!count)
        return 0;

    // 블로킹: 비어 있으면 대기(논블록이면 -EAGAIN)
    while (done == 0)
    {
        unsigned long flags;
        // 먼저 현재 있는 만큼 뽑아감
        spin_lock_irqsave(&uart_lock, flags);
        while (done < count && !rb_empty(&rx_rb))
        {
            char c = rb_get(&rx_rb);
            spin_unlock_irqrestore(&uart_lock, flags);
            if (put_user(c, (char __user *)ubuf + done)) // 1바이트씩
                return done ? (ssize_t)done : -EFAULT;
            done++;
            spin_lock_irqsave(&uart_lock, flags);
        }
        spin_unlock_irqrestore(&uart_lock, flags);

        if (done)
            break;
        if (file->f_flags & O_NONBLOCK)
            return -EAGAIN;

        ret = wait_event_interruptible(rx_wq, !rb_empty(&rx_rb));
        if (ret)
            return ret; // -ERESTARTSYS 등
    }
    return done;
}

static ssize_t uart3_write(struct file *file, const char __user *ubuf, size_t count, loff_t *ppos)
{
    size_t sent = 0;
    int ret;

    if (!count)
        return 0;

    while (sent < count)
    {
        char c;

        if (get_user(c, ubuf + sent))
            return sent ? (ssize_t)sent : -EFAULT;

        // 우선 HW FIFO 여유면 바로 씀(지연 최소화)
        if (!(readl(uart3_base + UART_FR) & UART_FR_TXFF))
        {
            writeb(c, uart3_base + UART_DR);
            sent++;
            continue;
        }

        // 아니면 소프트 TX 링버퍼에 push
        unsigned long flags;
        spin_lock_irqsave(&uart_lock, flags);
        if (!rb_full(&tx_rb))
        {
            rb_put(&tx_rb, c);
            sent++;
            // TX 인터럽트 ON (송신 시작 트리거)
            {
                u32 im = readl(uart3_base + UART_IMSC);
                if (!(im & TXIM))
                    writel(im | TXIM, uart3_base + UART_IMSC);
            }
            spin_unlock_irqrestore(&uart_lock, flags);
        }
        else
        {
            spin_unlock_irqrestore(&uart_lock, flags);
            // 꽉 찼으면: 논블록이면 즉시 반환, 블록이면 여유를 기다림
            if (file->f_flags & O_NONBLOCK)
                return sent ? (ssize_t)sent : -EAGAIN;

            ret = wait_event_interruptible(tx_wq, !rb_full(&tx_rb));
            if (ret)
                return sent ? (ssize_t)sent : ret;
        }
    }
    return sent;
}

// poll/select 지원
static __poll_t uart3_poll(struct file *file, poll_table *wait)
{
    __poll_t mask = 0;
    unsigned long flags;

    poll_wait(file, &rx_wq, wait);
    poll_wait(file, &tx_wq, wait);

    spin_lock_irqsave(&uart_lock, flags);
    if (!rb_empty(&rx_rb))
        mask |= POLLIN | POLLRDNORM;
    if (!rb_full(&tx_rb))
        mask |= POLLOUT | POLLWRNORM;
    spin_unlock_irqrestore(&uart_lock, flags);

    return mask;
}

static const struct file_operations fops = {
    .owner = THIS_MODULE,
    .open = uart3_open,
    .release = uart3_release,
    .read = uart3_read,
    .write = uart3_write,
    .poll = uart3_poll,
};

static int __init uart3_init(void)
{
    int ret;

    // dev_t 할당 (동적 메이저)
    ret = alloc_chrdev_region(&devno, 0, 1, DEVICE_NAME);
    if (ret)
        return ret;

    // cdev 등록
    cdev_init(&uart3_cdev, &fops);
    uart3_cdev.owner = THIS_MODULE;
    ret = cdev_add(&uart3_cdev, devno, 1);
    if (ret)
        goto err_cdev;

    // class/device 생성 → /dev/uart3_raw 자동 생성
    uart3_class = class_create(DEVICE_NAME);
    if (IS_ERR(uart3_class))
    {
        ret = PTR_ERR(uart3_class);
        goto err_class;
    }
    uart3_class->devnode = uart3_devnode;

    if (IS_ERR(device_create(uart3_class, NULL, devno, NULL, DEVICE_NAME)))
    {
        ret = -EINVAL;
        goto err_dev;
    }

    // MMIO 매핑
    uart3_base = ioremap(UART3_BASE_PHYS, UART3_REG_SIZE);
    if (!uart3_base)
    {
        ret = -ENOMEM;
        goto err_map;
    }

    // 동기화 객체 초기화
    spin_lock_init(&uart_lock);
    init_waitqueue_head(&rx_wq);
    init_waitqueue_head(&tx_wq);
    rx_rb.head = rx_rb.tail = 0;
    tx_rb.head = tx_rb.tail = 0;

    // IRQ 번호 결정 (모듈 파라미터 우선, 아니면 DT에서 조회)
    if (irq < 0)
    {
        irq = get_uart3_irq_from_dt();
        if (irq < 0)
        {
            pr_err("Failed to obtain UART3 IRQ from DT\n");
            ret = -ENODEV;
            goto err_irq;
        }
    }

    // IRQ 등록
    ret = request_irq(irq, uart3_irq_handler, IRQF_SHARED, DEVICE_NAME, &irq_cookie);
    if (ret)
    {
        pr_err("request_irq(%d) failed: %d\n", irq, ret);
        goto err_irq;
    }

    pr_info("UART3 IRQ driver loaded: /dev/%s (major=%d) irq=%d\n",
            DEVICE_NAME, MAJOR(devno), irq);
    return 0;

err_irq:
    iounmap(uart3_base);
err_map:
    device_destroy(uart3_class, devno);
err_dev:
    class_destroy(uart3_class);
err_class:
    cdev_del(&uart3_cdev);
err_cdev:
    unregister_chrdev_region(devno, 1);
    return ret;
}

static void __exit uart3_exit(void)
{
    // 사용자 접근 경로부터 차단
    writel(0, uart3_base + UART_IMSC); // 모든 UART IRQ 끄기
    if (irq >= 0)
        free_irq(irq, &irq_cookie);

    iounmap(uart3_base);
    device_destroy(uart3_class, devno);
    class_destroy(uart3_class);
    cdev_del(&uart3_cdev);
    unregister_chrdev_region(devno, 1);
    pr_info("UART3 IRQ driver unloaded\n");
}

module_init(uart3_init);
module_exit(uart3_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("KimMS");
MODULE_DESCRIPTION("BCM2711 PL011 UART3 interrupt-driven driver with ring buffers");
