device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name_train = "python train.py --img 640 --cfg yolov5s.yaml --hyp hyp.no-augmentation.yaml --batch 16 --epochs 100 --data "+ name +" --weights yolov5s.pt --workers 24 --name yolo_road_det"+added
    os.chdir("../")
    
    name_infer = "python detect.py --source ../Road_Sign_Dataset"+added+"/images/test --weights runs/train/"+"yolo_road_det"+added+"/weights/best.pt --conf 0.25 --name yolo_road_det"+added
    name_val = "python val.py --data "+name +" --weights runs/train/yolo_road_det"+ added + "/weights/best.pt --img 640 --task test --save_txt True"
    subprocess.call(name_train)
    subprocess.call(name_infer)
    subprocess.call(name_val)
    print("finished_testing")