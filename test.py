from hardware_simulator import HarwareSimulator
#import old.temp as temp
import json

# result = temp.huggingface_net_analysis("meta-llama/Llama-2-7b-hf", "intel_13900k")
# print(result["total_results"]) # time: 0.15061676
# result = temp.json_net_analysis("llamatest.json", "intel_13900k")

def write_model_json():
    data = {
        'num_attention_heads': 32,
        'hidden_size': 4096,
        'num_key_value_heads': 32,
        'norm_layers': ["attn_norm", "mlp_norm"],
        'num_hidden_layers': 32,
        'intermediate_size': 11008,
        'vocab_size': 32000
    }

    with open('llamatest.json', 'w') as f:
        json.dump(data, f)

def write_performance_json():
    data = {
        'ops': 13356498944.0,
        'load_act': 6217728.0,
        'load_weight': 13214154752.0,
        'memory_access': 13495261696.0,
        'memory_consumption': 13220629504.0,
        'memory_consumption_weight': 12952010752.0,
        'store_act': 5929472.0,
        'w_bit': 16,
        'a_bit': 16
    }

    with open('model.json', 'w') as f:
        json.dump(data, f)

write_performance_json()
sim = HarwareSimulator("./model.json", "nvidia_A6000")
sim.get_roofline()
results = sim.get_network_analysis()
print(results)