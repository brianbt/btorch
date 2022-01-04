# BTorch
BTorch is a PyTorch's useful utils library

## Requiements
PyTorch ≥ 1.10

## High Level Module (nn.Module)
You can use `btorch.nn` as normal `torch.nn` with more useful functions.  
You can define your model by subclassing it from `btorch.nn.Module` and everythings will be same as subclassing from `torch.nn.Module`.  
`btorch.nn.Module` provides you high-level training loop that you can define by yourself. Making the code more clean while maintain the flexibilityof PyTorch.  

The high-level methods are  
- .fit()  
- .train_net()  
- .train_epoch()  
- .test_epoch()  

Override them when necessary and train your model by just calling `.fit()`