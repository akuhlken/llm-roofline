from ModelSimulator import ModelSimulator

sim = ModelSimulator("meta-llama/LLama-2-7b")
results = sim.simulate("intel_13900k", 512, 1)
print(results)
