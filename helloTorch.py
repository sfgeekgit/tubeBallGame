import torch
import numpy

def main():
    print("hello")
    #torch.device("cuda" if torch.cuda.is_available() else "cpu")  
    x = torch.tensor([1, 2, 3])
    print(x)

if __name__ == "__main__":
    main()
