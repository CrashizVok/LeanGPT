import torch
print(torch.cuda.is_available())           
print(torch.cuda.get_device_name(0))       
print(torch.device("cuda" if torch.cuda.is_available() else "cpu"))  
