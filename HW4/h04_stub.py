# Alex Welk - CMSC438 - HW4 - 11/13/2024
# Description: In this program we are performing an adversarial attack on a series of images, this means we change the image every so slightly such that a model fails to predict the correct case.
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.models as models
from torch.hub import load

# write code for adversarial attack on an image (32x32, RGB, given as tensor [1,3,32,32], with meximum perturbation (epsilon) <= 8/255=0.0314)
# use the pre-trained cifar10_resnet20 model as shown below
def adversarial_attack(image, label, epsilon=8/255):
    # Clone and detach the input image to ensure it doesn't affect the computation graph
    image_adv = image.clone().detach().requires_grad_(True)

    # Load the pretrained model
    model = torch.hub.load('chenyaofo/pytorch-cifar-models', 'cifar10_resnet20', pretrained=True, verbose=False)
    model.eval()

    # compute output and loss
    output = model(image_adv)
    loss = F.cross_entropy(output, label)

    # Compute the gradient of the loss with respect to the image
    loss.backward()

    # Apply the FGSM formula to generate the adversarial example
    gradient_sign = image_adv.grad.sign()
    image_adv = image_adv + epsilon * gradient_sign



    return image_adv.detach()

if __name__ == "__main__":

    import numpy as np
    import matplotlib.pyplot as plt
    from torchvision import transforms
    from PIL import Image


    #from gradescope_solution import pgd_attack, fgsm_attack;
    #adversarial_attack = pgd_attack
    #adversarial_attack = fgsm_attack

    def prepare_image(image_path):
        image = Image.open(image_path)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010]
            )
        ])
        image_tensor = transform(image)
        return image_tensor.unsqueeze(0)
    
    def display_normalized_image(tensor_image, title = '', mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]):
        tensor = tensor_image.cpu().detach()
        
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        
        mean = torch.tensor(mean).view(3, 1, 1)
        std = torch.tensor(std).view(3, 1, 1)
        
        tensor = tensor * std + mean
        
        tensor = torch.clamp(tensor, 0, 1)
        
        img = tensor.permute(1, 2, 0).numpy()
        
        plt.figure(figsize=(8, 8))
        plt.imshow(img)
        plt.axis('off')
        plt.title(title)
        plt.show()
    
    # Define CIFAR10 classes
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
    
    # see three images that we will use in evaluating the adversarial attack code:
    images = ['automobile.png','dog.png','ship.png']
    class_ids = [1, 5, 8]
    
    model = torch.hub.load('chenyaofo/pytorch-cifar-models', 'cifar10_resnet20', pretrained=True, verbose=False, trust_repo=True)
    model.eval()
    
    
    def calculate_perturbation_epsilon(image_adv, image):
        return (image_adv - image).abs().max().item()
    
    for i in range(len(images)):
        print("####")
        
        # Process one image
        #image_path = f"in_data/{images[i]}"
        image_path = f"{images[i]}"
        image = prepare_image(image_path)
        
        true_class = class_ids[i]
        
        print(f"True class: {true_class} {classes[true_class]}")
        
        # Get original prediction
        with torch.no_grad():
            output = model(image)
            pred = output.argmax(dim=1).item()
            confidence = F.softmax(output, dim=1)[0][pred].item()
            print(f"Original prediction: {classes[pred]} ({confidence:.2%} confidence)")
        
        target_label = torch.tensor([true_class,])
        
        
        print("Expected input data shape and type for your adversarial_attack(image, label) function:")
        print(f"   image shape: {image.shape} and type: {type(image)}")
        print(f"   label shape: {target_label.shape} and type: {type(target_label)}")
    
        
        # Perform adversarial attack
        image_adv = adversarial_attack(image, target_label)
        
        
        with torch.no_grad():
            output_adv = model(image_adv)
            pred_adv = output_adv.argmax(dim=1).item()
            confidence_adv = F.softmax(output_adv, dim=1)[0][pred_adv].item()
            adv_epsilon = calculate_perturbation_epsilon(image_adv,image)
            print(f"Adversarial attack prediction: {classes[pred_adv]} ({confidence_adv:.2%} confidence, max change to image pixel: {255*adv_epsilon})")
            
            if (pred_adv == true_class):
                print("Adversarial attack failed, the true class is predicted!")
            
            if (adv_epsilon>8/255+0.001):
                print("Adversarial attack failed, the change to the image is too high (>8/255), at least at some pixels")
            display_normalized_image(image,f"original image: {classes[true_class]}")
            display_normalized_image(image_adv,f"modified image: {classes[pred_adv]}")
