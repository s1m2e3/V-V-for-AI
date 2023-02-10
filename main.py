import subprocess
import itertools


if __name__=="__main__":

    corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']
    severity = ["1","2","5","10"]
    corrupt_train = ["True","False"]
    corrupt_test = ["True","False"]
    
    arguments = []
    for element in itertools.product(*[corruptions,severity,corrupt_train,corrupt_test]):
        arguments.append(element)
    name ="python3 corrupt.py " 
    for quad in arguments:
        
        quad = " ".join(quad)
        final = name+quad
        subprocess.call(final)