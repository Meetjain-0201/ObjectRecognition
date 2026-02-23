import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

MODEL_PATH = "data/resnet18-v2-7.onnx"
IMG_DIR = "data/test_images/"
NET_SIZE = 224

OBJECTS = {
    "object1": ["obj1_1.jpeg", "obj1_2.jpeg", "obj1_3.jpeg"],
    "object2": ["obj2_1.jpeg", "obj2_2.jpeg", "obj2_3.jpeg"],
    "object3": ["obj3_1.jpeg", "obj3_2.jpeg", "obj3_3.jpeg"],
    "object4": ["obj4_1.jpeg", "obj4_2.jpeg", "obj4_3.jpeg"],
    "object5": ["obj5_1.jpeg", "obj5_2.jpeg", "obj5_3.jpeg"],
}

net = cv2.dnn.readNetFromONNX(MODEL_PATH)
print("ResNet18 loaded")

embeddings = []
labels = []

for label, fnames in OBJECTS.items():
    for fname in fnames:
        img = cv2.imread(IMG_DIR + fname)
        if img is None:
            print(f"Could not load {fname}")
            continue
        resized = cv2.resize(img, (NET_SIZE, NET_SIZE))
        blob = cv2.dnn.blobFromImage(resized, (1.0/255.0)*(1/0.226),
                                      (NET_SIZE, NET_SIZE),
                                      (124, 116, 104), True, False, cv2.CV_32F)
        net.setInput(blob)
        try:
            emb = net.forward("onnx_node!resnetv22_flatten0_reshape0")
        except:
            emb = net.forward()
        embeddings.append(emb.flatten())
        labels.append(label)
        print(f"Embedded: {label} - {fname}")

embeddings = np.array(embeddings)

# PCA to 2D
mean = np.mean(embeddings, axis=0)
centered = embeddings - mean
U, s, Vt = np.linalg.svd(centered, full_matrices=False)
proj = centered @ Vt[:2].T

# Plot
colors = {"object1":"red","object2":"blue","object3":"green","object4":"orange","object5":"purple"}
plt.figure(figsize=(8,6))
for i, (x, y) in enumerate(proj):
    lbl = labels[i]
    plt.scatter(x, y, color=colors[lbl], s=100)
    plt.annotate(lbl, (x, y), textcoords="offset points", xytext=(5,5), fontsize=8)

handles = [plt.Line2D([0],[0], marker='o', color='w', markerfacecolor=c, markersize=10, label=l)
           for l, c in colors.items()]
plt.legend(handles=handles)
plt.title("ResNet18 Embeddings projected to 2D (PCA)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.savefig("results/embedding_plot.png", dpi=150)
plt.show()
print("Saved to results/embedding_plot.png")
