import json
from hardwares.hardware_params import hardware_params
import sys
#sys.path.append('..')
import numpy as np
import matplotlib.pyplot as plt
from hardwares.hardware_params import hardware_params

class HarwareSimulator:

    def __init__(self, network_json, hardware_id):
        with open(network_json, 'r') as f:
            model_data = json.load(f)
        self.ops = model_data["ops"]
        self.load_act = model_data["load_act"]
        self.load_weight = model_data["load_weight"]
        self.memory_access = model_data["memory_access"]
        self.memory_consumption = model_data["memory_consumption"]
        self.memory_consumption_weight = model_data["memory_consumption_weight"]
        self.store_act = model_data["store_act"]
        self.w_bit = model_data["w_bit"]
        self.a_bit = model_data["a_bit"]
        self.hardware_id = hardware_id

    def get_network_analysis(self):
        bandwidth, max_OPS, onchip_buffer = self._get_hardware_info()
        memory_access = self.load_weight + self.load_act + self.store_act
        arithmetic_intensity, performance, bound = self._roofline_analyze(bandwidth, max_OPS, self.ops, memory_access)
        inference_time = self.ops / performance
        return inference_time, arithmetic_intensity, performance, bound

    def get_roofline(self):
        bandwidth, max_OPS, onchip_buffer = self._get_hardware_info()
        inference_time, arithmetic_intensity, performance, bound= self.get_network_analysis()
        fig=plt.figure(figsize=(5, 3))
        y_max = max_OPS
        turning_point = y_max / bandwidth

        plt.plot(
            [0, turning_point, turning_point * 3], [0, y_max, y_max], color="black"
        )
        plt.xlabel("Arithmetic Intensity (OPs/byte)")
        plt.ylabel("Performance (OPS)")

        plt.fill_between(
            [0, turning_point], [0, y_max], color="red", alpha=0.3, label="Memory Bound"
        )
        plt.text(turning_point * 0.55, y_max * 0.1, "memory-bound", ha="center", va="center")
        
        plt.fill_between(
            [turning_point,turning_point, turning_point * 3],
            [0,y_max, y_max],
            color="green",
            alpha=0.3,
            label="compute-bound",
        )
        plt.text(turning_point * 1.5, y_max * 0.1, "compute-bound")

        plt.hlines(y_max, 0, turning_point, color="black", linestyle="--")
        plt.vlines(turning_point, 0, y_max*2, color="black", linestyle="--")
        plt.annotate(
            "turning point",
            xy=(turning_point, y_max),
            xytext=(turning_point * 1.1, y_max * 0.8),
            arrowprops=dict(arrowstyle="->"),
        )
        plt.ylim(0, y_max * 1.1)
        plt.xlim(0, turning_point * 3)

        #plt.plot(arithmetic_intensity,performance,'ro') #TODO wrong scale

        # save pdf
        plt.savefig(f"./output/roofline_{self.hardware_id}.png", bbox_inches="tight")

    def _get_hardware_info(self):
        bandwidth = hardware_params[self.hardware_id]["bandwidth"]
        if self.w_bit <= 8 and self.a_bit <= 8:
            max_OPS = hardware_params[self.hardware_id]["INT8"]
        else:
            max_OPS = hardware_params[self.hardware_id]["FP16"]
        onchip_buffer = hardware_params[self.hardware_id]["onchip_buffer"]
        return bandwidth, max_OPS, onchip_buffer
    
    def _roofline_analyze(self, bandwidth, max_OPS, OPs, memory_access):
        # bandwidth is bytes/s
        # memory_access in byte
        # x axis is OPS/byte
        # y axis is OPS/s
        y_max = max_OPS
        memory_access_bytes = memory_access
        turning_point = y_max / bandwidth
        arithmetic_intensity = OPs / memory_access_bytes
        if arithmetic_intensity < turning_point:
            bound = "memory"
            performance = arithmetic_intensity * bandwidth
        else:
            bound = "compute"
            performance = y_max
        if performance==0:
            1==1
            pass
        return arithmetic_intensity, performance, bound