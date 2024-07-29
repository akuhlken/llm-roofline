import onnx
model = onnx.load("/mnt/c/Users/akuhl/Downloads/gpt2-10.onnx")
# iterate through inputs of the graph
for input in model.graph.input:
    print(input.name, end=": ")
    # get type of input tensor
    tensor_type = input.type.tensor_type
    # check if it has a shape:
    if (tensor_type.HasField("shape")):
        # iterate through dimensions of the shape:
        for d in tensor_type.shape.dim:
            # the dimension may have a definite (integer) value or a symbolic identifier or neither:
            if (d.HasField("dim_value")):
                print(d.dim_value, end=", ")  # known dimension
            elif (d.HasField("dim_param")):
                print(d.dim_param, end=", ")  # unknown dimension with symbolic name <- this is what I get
            else:
                print("?", end=", ")  # unknown dimension with no name
    else:
        print("unknown rank", end="")
    print()