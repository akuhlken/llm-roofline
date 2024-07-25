
from hardware_params import hardware_params

def str_number(num):
    if num > 1e14:
        return f"{num/1e12:.0f}T"
    elif num > 1e12:
        return f"{num/1e12:.1f}T"
    elif num>1e11:
        return f"{num/1e9:.0f}G"
    elif num > 1e9:
        return f"{num/1e9:.1f}G"
    elif num > 1e8:
        return f"{num/1e6:.0f}M"
    elif num > 1e6:
        return f"{num/1e6:.1f}M"
    elif num > 1e5:
        return f"{num/1e3:.0f}K"
    elif num > 1e3:
        return f"{num/1e3:.1f}K"
    elif num >= 1:
        return f"{num:.1f}"
    else:
        return f"{num:.2f}"

def str_number_time(num):
    if num >= 1:
        return f"{num:.1f}"
    elif num > 1e-3:
        return f"{num*1e3:.1f}m"
    elif num > 1e-6:
        return f"{num*1e6:.1f}u"
    elif num > 1e-9:
        return f"{num*1e9:.1f}n"
    else:
        return f"{num:.0f}"
    
def get_bottleneck(bandwidth, max_OPS, OPs, memory_access):
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

def get_hardware_info(self, hardware, w_bit, a_bit, kv_bit):
    bandwidth = hardware_params[hardware]["bandwidth"]
    if w_bit <= 8 and a_bit <= 8 and kv_bit <= 8:
        max_OPS = hardware_params[hardware]["INT8"]
    else:
        max_OPS = hardware_params[hardware]["FP16"]
    onchip_buffer = hardware_params[hardware]["onchip_buffer"]
    return bandwidth, max_OPS, onchip_buffer

