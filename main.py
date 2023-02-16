import subprocess
import itertools


if __name__=="__main__":

    corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']
    corruptions = ["--corruption_name "+corrupted for corrupted in corruptions ]
    severity = ["2","3","4","5"]
    severity = ["--severity "+severities for severities in severity]
    corrupt_train = ["True","False"]
    corrupt_train = ["--apply_on_train " + train for train in corrupt_train]
    corrupt_test = ["True","False"]
    corrupt_test = ["--apply_on_test " + test for test in corrupt_test]
    arguments = []
    
    for element in itertools.product(*[corruptions,severity,corrupt_train,corrupt_test]):
        arguments.append(element)
    name ="python3 corrupt.py " 
    for quad in arguments:
        print("first_call")
        quad = " ".join(quad)
        final = name+quad
        subprocess.call(final)
        print("finish_call")