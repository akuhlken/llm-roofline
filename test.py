import hardware_simulator
import json

#result = hardware_simulator.huggingface_net_analysis("meta-llama/Llama-2-7b-hf", "intel_13900k")
result = hardware_simulator.json_net_analysis("llamatest.json", "intel_13900k")

def write_json_test():
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

print(result)