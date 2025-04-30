from ultralytics import YOLO

def main():
    model = YOLO("checkpoints/drink_scan_v10.pt")
    
    metrics = model.val(data=r'datasets/data.yaml', conf=0.8, iou=0.8, save_json=True)
    
    metrics.confusion_matrix.plot(
        normalize=True,
        names=model.names,
        save_dir='.',
    )

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()  
    main()
