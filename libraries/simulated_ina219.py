import random

class INA219:
    def __init__(self, i2c_bus=1, addr=0x40):
        print("Simulating INA219 sensor. No actual I2C communication will occur.")
        self.bus_voltage = 8.0  # Simulate a reasonable battery voltage
        self.current_mA = 1000  # Simulate current in mA
        self.power_W = 8.0  # Simulate power in W

    def getBusVoltage_V(self):
        # Simulate voltage fluctuations
        self.bus_voltage = max(7.0, min(8.4, self.bus_voltage + (random.random() - 0.5) * 0.1))
        return self.bus_voltage

    def getCurrent_mA(self):
        # Simulate current fluctuations
        self.current_mA = max(100, min(2000, self.current_mA + (random.random() - 0.5) * 50))
        return self.current_mA

    def getPower_W(self):
        # Simulate power fluctuations
        self.power_W = max(5.0, min(15.0, self.power_W + (random.random() - 0.5) * 0.5))
        return self.power_W

    def getShuntVoltage_mV(self):
        return 0.0 # Not relevant for battery percentage

    def set_calibration_32V_2A(self):
        pass # Not needed for simulation


