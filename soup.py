import torch
import sys
sys.path.insert(0, "ultralytics-timm")

from ultralytics import YOLO

# Load models
model_paths = [
    '/home/sersasj/BYU---Locating-Bacterial-Flagellar-Motors-2025/runs_conv_base_more_augs/fold0/weights/epoch31_optimized.pt',

    '/home/sersasj/BYU---Locating-Bacterial-Flagellar-Motors-2025/runs_conv_base_more_augs/fold0/weights/epoch37_optimized.pt',
    '/home/sersasj/BYU---Locating-Bacterial-Flagellar-Motors-2025/runs_conv_base_more_augs/fold0/weights/epoch34_optimized.pt',
    '/home/sersasj/BYU---Locating-Bacterial-Flagellar-Motors-2025/runs_conv_base_more_augs/fold0/weights/epoch39_optimized.pt',


]

models = [YOLO(path) for path in model_paths]
state_dicts = [model.model.state_dict() for model in models]

avg_state_dict = {} 

# Iterate through keys and average the parameters
for key in state_dicts[0].keys():
    avg_state_dict[key] = sum(d[key] for d in state_dicts) / len(state_dicts)

soup_model = YOLO(model_paths[0]) 
soup_model.model.load_state_dict(avg_state_dict)
soup_model.model.eval()

soup_model.save('soup_conv_base_31_34_37_39_more_augs_yolo.pt')


test_image = '/home/sersasj/BYU---Locating-Bacterial-Flagellar-Motors-2025/train/tomo_01a877/slice_0155.jpg'
conf_threshold = 0.1  
def test_model(model, name, image_path, conf):
    print(f"\nTesting {name}...")
    try:
        results = model.predict(image_path, 
                              imgsz=960, 
                              conf=conf,
                              verbose=False)
        print(f"{name} results: {len(results[0].boxes)} boxes")
        # Uncomment to display results
        # results[0].show(labels=True)
        return results
    except Exception as e:
        print(f"Error testing {name}: {e}")
        return None

model1 = YOLO(model_paths[0])
model2 = YOLO(model_paths[1])
results1 = test_model(model1, "Model 1", test_image, conf_threshold)
results2 = test_model(model2, "Model 2", test_image, conf_threshold)

print("\nTesting soup model...")
soup_loaded = YOLO('soup_conv_base_31_34_37_39_more_augs_yolo.pt')
soup_results = test_model(soup_loaded, "Soup Model", test_image, conf_threshold)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
plt.imshow(results1[0].plot())
plt.axis('off')
plt.savefig('model1_results.jpg', bbox_inches='tight', pad_inches=0)
plt.close()

plt.figure(figsize=(10, 10))
plt.imshow(results2[0].plot())
plt.axis('off')
plt.savefig('model2_results.jpg', bbox_inches='tight', pad_inches=0)
plt.close()

plt.figure(figsize=(10, 10))
plt.imshow(soup_results[0].plot())
plt.axis('off')
plt.savefig('soup_model_results.jpg', bbox_inches='tight', pad_inches=0)
plt.close()


