"""
专门用于Actiview脑电采集的Trigger测试
发送的trigger信号会被记录在.bdf文件中
"""

import serial
import serial.tools.list_ports
import time
from datetime import datetime
import numpy as np

class ActiviewTrigger:
    def __init__(self, port=None, baudrate=57600):
        """
        初始化Actiview Trigger
        
        Parameters:
        port: COM端口名称，如果为None则自动检测
        baudrate: 波特率，默认57600
        """
        self.ser = None
        self.port = port or self._find_trigger_port()
        self.baudrate = baudrate
        self.is_connected = False
        
    def _find_trigger_port(self):
        """自动查找trigger端口"""
        ports = list(serial.tools.list_ports.comports())
        
        print("可用COM端口:")
        for i, port in enumerate(ports):
            print(f"  {i+1}. {port.device}: {port.description}")
        
        # 查找USB Serial Port或类似设备
        for port in ports:
            if any(keyword in port.description.upper() for keyword in 
                   ['USB SERIAL', 'USB-SERIAL', 'SERIAL PORT', 'CH340', 'CP210', 'FTDI']):
                print(f"自动选择端口: {port.device} - {port.description}")
                return port.device
        
        # 如果没找到特定设备，让用户选择
        if ports:
            print(f"未找到标准USB串口设备，使用第一个可用端口: {ports[0].device}")
            return ports[0].device
            
        return None
    
    def connect(self):
        """连接到trigger端口"""
        if not self.port:
            print("错误: 未找到可用的COM端口")
            return False
            
        try:
            self.ser = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=0.1,  # 短超时时间
                write_timeout=0.1
            )
            
            # 清空缓冲区
            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()
            
            self.is_connected = True
            print(f"成功连接到trigger端口: {self.port}")
            print(f"串口配置: 波特率={self.baudrate}, 数据位=8, 停止位=1, 无校验")
            
            return True
            
        except serial.SerialException as e:
            print(f"串口连接失败: {e}")
            print("请检查:")
            print("1. COM端口是否被其他程序占用")
            print("2. trigger设备是否正确连接")
            print("3. 是否有足够的权限访问串口")
            return False
    
    def send_trigger(self, trigger_value, duration_ms=10):
        """
        发送trigger信号到Actiview
        
        Parameters:
        trigger_value: trigger值 (1-255)
        duration_ms: trigger持续时间(毫秒)，默认10ms
        
        Returns:
        bool: 发送是否成功
        """
        if not self.is_connected or not self.ser:
            print("错误: 未连接到trigger端口")
            return False
            
        if not (1 <= trigger_value <= 255):
            print(f"错误: trigger值必须在1-255之间，当前值: {trigger_value}")
            return False
            
        try:
            # 记录发送时间
            send_time = datetime.now()
            
            # 发送trigger值
            trigger_byte = bytes([trigger_value])
            self.ser.write(trigger_byte)
            self.ser.flush()
            
            # 维持trigger持续时间
            if duration_ms > 0:
                time.sleep(duration_ms / 1000.0)
                
                # 发送0来结束trigger
                self.ser.write(bytes([0]))
                self.ser.flush()
            
            print(f"[{send_time.strftime('%H:%M:%S.%f')[:-3]}] "
                  f"Trigger发送成功: {trigger_value} (0x{trigger_value:02X}), "
                  f"持续时间: {duration_ms}ms")
            
            return True
            
        except Exception as e:
            print(f"发送trigger失败: {e}")
            return False
    
    def send_periodic_trigger(self, trigger_value=8, interval_sec=1.0, duration_sec=None):
        """
        每隔指定时间发送trigger信号
        
        Parameters:
        trigger_value: trigger值，默认8
        interval_sec: 发送间隔(秒)，默认1秒
        duration_sec: 总持续时间(秒)，如果为None则一直发送直到手动停止
        """
        print(f"开始每隔{interval_sec}秒发送trigger {trigger_value}")
        if duration_sec:
            print(f"总持续时间: {duration_sec}秒")
        else:
            print("按Ctrl+C停止发送")
        print("=" * 60)
        
        start_time = time.time()
        trigger_count = 0
        
        try:
            while True:
                # 发送trigger
                if self.send_trigger(trigger_value, duration_ms=50):
                    trigger_count += 1
                
                # 检查是否达到持续时间
                if duration_sec and (time.time() - start_time) >= duration_sec:
                    break
                
                # 等待下一个发送时间
                time.sleep(interval_sec)
                
        except KeyboardInterrupt:
            print(f"\n用户停止发送")
        
        elapsed_time = time.time() - start_time
        print("=" * 60)
        print(f"发送完成:")
        print(f"  总trigger数量: {trigger_count}")
        print(f"  总时间: {elapsed_time:.1f}秒")
        print(f"  平均间隔: {elapsed_time/max(trigger_count-1, 1):.2f}秒")
    
    def send_trigger_sequence(self, triggers, interval_ms=1000):
        """
        发送trigger序列
        
        Parameters:
        triggers: trigger值列表
        interval_ms: trigger间隔时间(毫秒)
        """
        print(f"开始发送trigger序列，共{len(triggers)}个trigger，间隔{interval_ms}ms")
        print("=" * 60)
        
        for i, trigger in enumerate(triggers):
            print(f"序列 {i+1}/{len(triggers)}: ", end="")
            self.send_trigger(trigger)
            
            if i < len(triggers) - 1:  # 最后一个trigger后不需要等待
                time.sleep(interval_ms / 1000.0)
        
        print("=" * 60)
        print("Trigger序列发送完成")
    
    def test_actiview_integration(self):
        """测试与Actiview的集成"""
        print("Actiview集成测试")
        print("请确保:")
        print("1. Actiview正在运行并处于采集状态")
        print("2. Trigger设备已正确连接")
        print("3. 在Actiview中能看到trigger通道")
        print()
        
        # 标准的实验trigger测试
        test_triggers = {
            'experiment_start': 10,
            'stimulus_1': 21,
            'stimulus_2': 22, 
            'stimulus_3': 23,
            'response': 100,
            'block_end': 200,
            'experiment_end': 255
        }
        
        print("发送实验相关的trigger:")
        for name, value in test_triggers.items():
            input(f"按Enter发送 {name} (trigger {value})...")
            self.send_trigger(value, duration_ms=50)  # 50ms持续时间确保被记录
            print()
    
    def interactive_test(self):
        """交互式trigger测试"""
        print("交互式Trigger测试")
        print("输入trigger值 (1-255)，输入 'q' 退出，输入 's' 进行序列测试")
        
        while True:
            user_input = input("\nTrigger值: ").strip().lower()
            
            if user_input == 'q':
                break
            elif user_input == 's':
                # 序列测试
                sequence = [10, 20, 30, 40, 50]
                self.send_trigger_sequence(sequence, interval_ms=2000)
            else:
                try:
                    trigger_value = int(user_input)
                    self.send_trigger(trigger_value, duration_ms=50)
                except ValueError:
                    print("请输入有效的数字、's'(序列测试)或'q'(退出)")
    
    def send_incremental_triggers(self, start_value=1, end_value=16, interval_sec=1.0):
        """
        每隔指定时间发送递增的trigger信号
        
        Parameters:
        start_value: 起始trigger值，默认1
        end_value: 结束trigger值，默认16
        interval_sec: 发送间隔(秒)，默认1秒
        """
        print(f"开始每隔{interval_sec}秒发送递增trigger")
        print(f"Trigger值范围: {start_value} 到 {end_value}")
        print("按Ctrl+C提前停止发送")
        print("=" * 60)
        
        start_time = time.time()
        trigger_count = 0
        
        try:
            for trigger_value in range(start_value, end_value + 1):
                print(f"发送第 {trigger_count + 1} 个trigger: ", end="")
                
                # 发送trigger
                if self.send_trigger(trigger_value, duration_ms=50):
                    trigger_count += 1
                
                # 如果不是最后一个trigger，等待间隔时间
                if trigger_value < end_value:
                    time.sleep(interval_sec)
                
        except KeyboardInterrupt:
            print(f"\n用户提前停止发送")
        
        elapsed_time = time.time() - start_time
        print("=" * 60)
        print(f"发送完成:")
        print(f"  总trigger数量: {trigger_count}")
        print(f"  总时间: {elapsed_time:.1f}秒")
        print(f"  平均间隔: {elapsed_time/max(trigger_count-1, 1):.2f}秒")
        print(f"  发送的trigger值: {list(range(start_value, start_value + trigger_count))}")

    def disconnect(self):
        """断开连接"""
        if self.ser and self.is_connected:
            self.ser.close()
            self.is_connected = False
            print(f"已断开trigger端口连接: {self.port}")

def main():
    print("Actiview Trigger测试程序")
    print("=" * 50)
    
    # 初始化trigger
    trigger = ActiviewTrigger()
    
    if not trigger.connect():
        print("无法连接到trigger设备，程序退出")
        return
    
    try:
        print("\n请选择测试模式:")
        print("1. Actiview集成测试")
        print("2. 交互式测试")
        print("3. 快速序列测试")
        print("4. 每隔1秒发送trigger 8")
        print("5. 自定义周期性发送")
        print("6. 每隔1秒发送递增trigger (1-16) [新功能]")
        
        choice = input("请选择 (1/2/3/4/5/6): ").strip()
        
        if choice == "1":
            trigger.test_actiview_integration()
        elif choice == "2":
            trigger.interactive_test()
        elif choice == "3":
            # 快速测试序列
            test_sequence = [1, 2, 10, 20, 50, 100, 200]
            trigger.send_trigger_sequence(test_sequence, interval_ms=1500)
        elif choice == "4":
            # 每隔1秒发送trigger 8
            print("开始每隔1秒发送trigger 8")
            print("按Ctrl+C停止发送")
            trigger.send_periodic_trigger(trigger_value=8, interval_sec=1.0)
        elif choice == "5":
            # 自定义周期性发送
            try:
                trigger_val = int(input("输入trigger值 (1-255): "))
                interval = float(input("输入发送间隔(秒): "))
                duration_input = input("输入总持续时间(秒，按回车表示一直发送): ").strip()
                duration = float(duration_input) if duration_input else None
                
                trigger.send_periodic_trigger(trigger_value=trigger_val, 
                                            interval_sec=interval, 
                                            duration_sec=duration)
            except ValueError:
                print("输入格式错误")
        elif choice == "6":
            # 每隔1秒发送递增trigger (1-16)
            print("开始每隔1秒发送递增trigger (1-16)")
            trigger.send_incremental_triggers(start_value=1, end_value=16, interval_sec=1.0)
        else:
            print("无效选择，运行默认测试...")
            trigger.test_actiview_integration()
            
    except KeyboardInterrupt:
        print("\n用户中断测试")
    finally:
        trigger.disconnect()

if __name__ == "__main__":
    main()