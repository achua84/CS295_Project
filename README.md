# CS295_Project

### If you want to start working on getting results…

Use ```get_encodings.py```. Make sure ```all_encodings.pkl``` is in the same directory. Each encoding from lines 6-9 will be loaded as dictionaries at first. Then, ```all_encodings``` on line 17 will be where all the “Post-test” encodings are stored. You can use ```all_encodings``` for the classification tests. 

class_descriptiors is a dictionary where the key is the class and the value is the hypervector: ```{class: hypervector}```. 

Currently, the hypervector is of length 10,000 and there are 10 classes (cifar10) 

```all_encodings``` will be a list of tuples: ```(image_encoding, label)```, in which the image_encoding is length 10,000 and the label is either a 0 (ID) or 1 (OOD). 

### If you want to change the HD input…

Change the settings in ```helper.py```. The comments should say which variables you could change. 

After, run “pretrained_model.py” with 
```!python pretrained_model.py --layers 28 --widen-factor 10```

And make sure the ```runs/WideResNet-28-10/model_best.pth.tar``` is in the same directory as ```pretrained_model.py```. You’ll have to change ~line 182 (the filepath variable) in ```pretrained_model.py``` so it points to the tar file. 

### If you want to retrain the WideResNet…

Follow the GitHub https://github.com/xternalz/WideResNet-pytorch 

If you want to continue training our current model, you can use the last checkpoint that was used, in which you can add ```--start-epoch 176 --resume “runs/WideResNet-28-10/checkpoint.pth.tar”``` to the command, since that’s when training stopped. Currently, the max the code goes is 200 epochs, but you can change that with adding ```--epochs 300```, or something similar. 

### If you want to see which layers we use for feature maps…

Look at ```wideresnet.py```. All the lines that have ```helper.weights.append``` are the layers we’re using for feature maps. These include Conv2d, ReLU, BasicBlock, NetworkBlock, and shortcuts connections. 

